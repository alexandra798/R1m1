"""RPN表达式求值器"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import logging
from .token_system import TokenType, TOKEN_DEFINITIONS

logger = logging.getLogger(__name__)


class RPNEvaluator:
    """评估RPN表达式的值"""

    @staticmethod
    def evaluate(token_sequence, data_dict):
        """
        评估RPN表达式

        Args:
            token_sequence: Token序列
            data_dict: 包含股票数据的字典，如{'open': array, 'close': array, ...}

        Returns:
            result: 计算结果的numpy数组或pandas Series
        """
        stack = []

        for token in token_sequence[1:]:  # 跳过BEG
            if token.name == 'END':
                break

            if token.type == TokenType.OPERAND:
                # 处理操作数
                if token.name in data_dict:
                    # 股票特征
                    stack.append(data_dict[token.name])
                elif token.name.startswith('const_'):
                    # 常数
                    const_value = float(token.name.split('_')[1])
                    # 创建与数据相同形状的常数数组
                    if isinstance(data_dict.get('close'), pd.Series):
                        stack.append(pd.Series(const_value, index=data_dict['close'].index))
                    else:
                        stack.append(np.full_like(data_dict['close'], const_value))
                elif token.name.startswith('delta_'):
                    # 时间窗口（作为整数使用）
                    delta_value = int(token.name.split('_')[1])
                    stack.append(delta_value)

            elif token.type == TokenType.OPERATOR:
                # 处理操作符
                if token.arity == 1:
                    # 一元操作
                    if len(stack) < 1:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None
                    operand = stack.pop()
                    result = RPNEvaluator.apply_unary_op(token.name, operand)
                    stack.append(result)

                elif token.arity == 2:
                    # 二元操作
                    if len(stack) < 2:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None
                    operand2 = stack.pop()
                    operand1 = stack.pop()
                    result = RPNEvaluator.apply_binary_op(token.name, operand1, operand2)
                    stack.append(result)

                elif token.arity == 3:
                    # 三元操作（如相关性）
                    if len(stack) < 3:
                        logger.error(f"Insufficient operands for {token.name}")
                        return None
                    operand3 = stack.pop()
                    operand2 = stack.pop()
                    operand1 = stack.pop()
                    result = RPNEvaluator.apply_ternary_op(token.name, operand1, operand2, operand3)
                    stack.append(result)

        # 返回栈顶元素
        if len(stack) == 1:
            return stack[0]
        elif len(stack) == 0:
            logger.error("Empty stack after evaluation")
            return None
        else:
            logger.warning(f"Stack has {len(stack)} elements after evaluation, expected 1")
            return stack[0]  # 返回栈顶元素

    @staticmethod
    def apply_unary_op(op_name, operand):
        """应用一元操作符"""
        if op_name == 'abs':
            return np.abs(operand)
        elif op_name == 'log':
            # 避免log(0)或负数
            if isinstance(operand, pd.Series):
                return np.log(np.maximum(operand, 1e-10))
            else:
                return np.log(np.maximum(operand, 1e-10))
        elif op_name == 'sign':
            return np.sign(operand)
        elif op_name == 'rank':
            # 计算排名百分位
            if isinstance(operand, pd.Series):
                return operand.rank(pct=True)
            else:
                return rankdata(operand, method='average') / len(operand)
        else:
            raise ValueError(f"Unknown unary operator: {op_name}")

    @staticmethod
    def apply_binary_op(op_name, operand1, operand2):
        """应用二元操作符"""
        if op_name == 'add':
            return operand1 + operand2
        elif op_name == 'sub':
            return operand1 - operand2
        elif op_name == 'mul':
            return operand1 * operand2
        elif op_name == 'div':
            # 安全除法
            if isinstance(operand1, pd.Series):
                return operand1.div(operand2).replace([np.inf, -np.inf], 0).fillna(0)
            else:
                return np.divide(operand1, operand2, out=np.zeros_like(operand1), where=operand2 != 0)
        elif op_name == 'greater':
            return (operand1 > operand2).astype(float)
        elif op_name == 'less':
            return (operand1 < operand2).astype(float)
        elif op_name.startswith('ts_'):
            # 时序操作
            return RPNEvaluator.apply_time_series_op(op_name, operand1, operand2)
        else:
            raise ValueError(f"Unknown binary operator: {op_name}")

    @staticmethod
    def apply_time_series_op(op_name, data, window):
        """应用时序操作符"""
        # 确保window是整数
        if isinstance(window, (pd.Series, np.ndarray)):
            window = int(window[0]) if len(window) > 0 else 5
        else:
            window = int(window)

        window = max(1, min(window, 100))  # 限制窗口大小

        if isinstance(data, pd.Series):
            if op_name == 'ts_mean':
                return data.rolling(window=window, min_periods=1).mean()
            elif op_name == 'ts_std':
                return data.rolling(window=window, min_periods=1).std().fillna(0)
            elif op_name == 'ts_max':
                return data.rolling(window=window, min_periods=1).max()
            elif op_name == 'ts_min':
                return data.rolling(window=window, min_periods=1).min()
            elif op_name == 'ts_sum':
                return data.rolling(window=window, min_periods=1).sum()
            else:
                raise ValueError(f"Unknown time series operator: {op_name}")
        else:
            # NumPy数组处理
            result = np.zeros_like(data)
            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                window_data = data[start_idx:i + 1]

                if op_name == 'ts_mean':
                    result[i] = np.mean(window_data)
                elif op_name == 'ts_std':
                    result[i] = np.std(window_data) if len(window_data) > 1 else 0
                elif op_name == 'ts_max':
                    result[i] = np.max(window_data)
                elif op_name == 'ts_min':
                    result[i] = np.min(window_data)
                elif op_name == 'ts_sum':
                    result[i] = np.sum(window_data)

            return result

    @staticmethod
    def apply_ternary_op(op_name, operand1, operand2, operand3):
        """应用三元操作符"""
        if op_name == 'corr':
            # 相关性：operand1和operand2的相关性，窗口大小为operand3
            window = int(operand3) if isinstance(operand3, (int, float)) else int(operand3[0])
            window = max(2, min(window, 100))

            if isinstance(operand1, pd.Series) and isinstance(operand2, pd.Series):
                return operand1.rolling(window=window, min_periods=2).corr(operand2)
            else:
                # NumPy实现
                result = np.zeros(len(operand1))
                for i in range(len(operand1)):
                    start_idx = max(0, i - window + 1)
                    if i - start_idx >= 1:  # 至少需要2个点
                        corr = np.corrcoef(operand1[start_idx:i + 1], operand2[start_idx:i + 1])[0, 1]
                        result[i] = corr if not np.isnan(corr) else 0
                return result

        elif op_name == 'cov':
            # 协方差
            window = int(operand3) if isinstance(operand3, (int, float)) else int(operand3[0])
            window = max(2, min(window, 100))

            if isinstance(operand1, pd.Series) and isinstance(operand2, pd.Series):
                return operand1.rolling(window=window, min_periods=2).cov(operand2)
            else:
                # NumPy实现
                result = np.zeros(len(operand1))
                for i in range(len(operand1)):
                    start_idx = max(0, i - window + 1)
                    if i - start_idx >= 1:
                        cov = np.cov(operand1[start_idx:i + 1], operand2[start_idx:i + 1])[0, 1]
                        result[i] = cov if not np.isnan(cov) else 0
                return result
        else:
            raise ValueError(f"Unknown ternary operator: {op_name}")

    @staticmethod
    def tokens_to_infix(token_sequence):
        """将RPN Token序列转换为中缀表达式字符串（用于可读性）"""
        stack = []

        for token in token_sequence[1:]:  # 跳过BEG
            if token.name == 'END':
                break

            if token.type == TokenType.OPERAND:
                # 操作数直接入栈
                stack.append(token.name)

            elif token.type == TokenType.OPERATOR:
                if token.arity == 1:
                    # 一元操作符
                    if len(stack) >= 1:
                        operand = stack.pop()
                        stack.append(f"{token.name}({operand})")

                elif token.arity == 2:
                    # 二元操作符
                    if len(stack) >= 2:
                        right = stack.pop()
                        left = stack.pop()

                        if token.name in ['add', 'sub', 'mul', 'div']:
                            # 算术操作符用中缀表示
                            op_symbol = {
                                'add': '+', 'sub': '-',
                                'mul': '*', 'div': '/'
                            }.get(token.name, token.name)
                            stack.append(f"({left} {op_symbol} {right})")
                        else:
                            # 其他用函数表示
                            stack.append(f"{token.name}({left}, {right})")

                elif token.arity == 3:
                    # 三元操作符
                    if len(stack) >= 3:
                        arg3 = stack.pop()
                        arg2 = stack.pop()
                        arg1 = stack.pop()
                        stack.append(f"{token.name}({arg1}, {arg2}, {arg3})")

        return stack[0] if stack else ""