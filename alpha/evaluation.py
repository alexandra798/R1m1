"""Alpha评估模块 - 支持RPN和传统公式"""
import pandas as pd
import numpy as np
import logging
import re
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.formula_generator import (
    safe_divide, ref, csrank,
    # Group 2 operators
    sign, abs_op, log, greater, less,
    rank, std, ts_max, ts_min, skew, kurt, mean, med, ts_sum,
    cov, corr, decay_linear, wma, ema
)

logger = logging.getLogger(__name__)


def is_rpn_formula(formula):
    """判断是否为RPN格式的公式"""
    # RPN公式通常包含这些特征
    rpn_indicators = ['BEG', 'END', 'add', 'sub', 'mul', 'div', 'ts_mean', 'ts_std']
    return any(indicator in formula for indicator in rpn_indicators)


def evaluate_rpn_formula(formula, data):
    """评估RPN格式的公式"""
    try:
        # 导入RPN相关模块
        from mcts.token_system import TOKEN_DEFINITIONS
        from mcts.rpn_evaluator import RPNEvaluator

        # 解析RPN字符串为Token序列
        token_names = formula.split()
        token_sequence = []

        for name in token_names:
            if name in TOKEN_DEFINITIONS:
                token_sequence.append(TOKEN_DEFINITIONS[name])
            else:
                # 尝试将其作为特征名处理
                logger.warning(f"Unknown token in RPN formula: {name}")
                return pd.Series(np.nan, index=data.index)

        # 转换数据为字典格式
        if hasattr(data, 'to_dict'):
            data_dict = data.to_dict('series')
        else:
            data_dict = data

        # 使用RPN求值器评估
        result = RPNEvaluator.evaluate(token_sequence, data_dict)

        # 确保返回Series
        if result is not None:
            if isinstance(result, pd.Series):
                return result
            else:
                return pd.Series(result, index=data.index)
        else:
            return pd.Series(np.nan, index=data.index)

    except Exception as e:
        logger.error(f"Error evaluating RPN formula '{formula}': {e}")
        return pd.Series(np.nan, index=data.index)


def sanitize_formula(formula):
    """
    清理和验证公式，防止不安全的操作

    Parameters:
    - formula: 要验证的公式字符串

    Returns:
    - sanitized_formula: 清理后的公式，如果不安全则返回None
    """
    # 检查是否包含不安全的操作
    unsafe_patterns = [
        r'__\w+__',  # 防止访问特殊方法
        r'import\s+',  # 防止导入模块
        r'exec\s*\(',  # 防止执行代码
        r'eval\s*\(',  # 防止嵌套eval
        r'open\s*\(',  # 防止文件操作
        r'file\s*\(',  # 防止文件操作
        r'input\s*\(',  # 防止输入操作
        r'raw_input\s*\(',  # 防止输入操作
    ]

    for pattern in unsafe_patterns:
        if re.search(pattern, formula, re.IGNORECASE):
            logger.warning(f"Unsafe formula detected: {formula}")
            return None

    # 验证公式只包含允许的字符
    allowed_chars = re.compile(r'^[a-zA-Z0-9_\s\+\-\*\/\(\)\.\,]+$')
    if not allowed_chars.match(formula):
        logger.warning(f"Formula contains invalid characters: {formula}")
        return None

    return formula


def evaluate_formula(formula, data):
    """
    评估公式在数据集上的结果
    支持RPN格式和传统格式

    Parameters:
    - formula: 要评估的公式字符串
    - data: 特征数据集

    Returns:
    - result: 评估后的特征
    """
    # 首先判断是否为RPN格式
    if is_rpn_formula(formula):
        return evaluate_rpn_formula(formula, data)

    # 以下为传统格式的处理
    try:
        # 清理和验证公式
        sanitized_formula = sanitize_formula(formula)
        if sanitized_formula is None:
            logger.error(f"Formula failed security validation: '{formula}'")
            return pd.Series(np.nan, index=data.index)

        # 创建安全的评估环境
        safe_dict = {}

        # 将DataFrame的每一列转换为Series并添加到字典中
        for col in data.columns:
            safe_dict[col] = data[col]

        # 限制可用的函数和变量
        allowed_functions = {
            'abs': abs,
            'max': max,
            'min': min,
            'sum': sum,
            'len': len,
            'safe_divide': safe_divide,
            # Group 1 operators
            'ref': ref,
            'csrank': csrank,
            # Group 2 operators - unary
            'sign': sign,
            'abs_op': abs_op,
            'log': log,
            # Group 2 operators - comparison
            'greater': greater,
            'less': less,
            # Group 2 operators - time series
            'rank': rank,
            'std': std,
            'ts_max': ts_max,
            'ts_min': ts_min,
            'skew': skew,
            'kurt': kurt,
            'mean': mean,
            'med': med,
            'ts_sum': ts_sum,
            # Group 2 operators - correlation
            'cov': cov,
            'corr': corr,
            # Group 2 operators - moving averages
            'decay_linear': decay_linear,
            'wma': wma,
            'ema': ema,
            'np': np,  # 允许numpy函数
            'pd': pd,  # 允许pandas函数
        }

        # 验证公式中的所有变量都存在
        formula_vars = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula)
        for var in formula_vars:
            if var not in safe_dict and var not in allowed_functions:
                logger.warning(f"Unknown variable '{var}' in formula: {formula}")
                return pd.Series(np.nan, index=data.index)

        # 将允许的函数添加到安全字典中
        safe_dict.update(allowed_functions)

        # 安全评估
        try:
            # 使用python引擎以获得更好的兼容性
            result = pd.eval(sanitized_formula, local_dict=safe_dict, engine='python')

            # 处理结果类型
            if isinstance(result, (int, float, np.number)):
                # 如果结果是标量，创建一个Series
                result = pd.Series(result, index=data.index)
            elif isinstance(result, np.ndarray):
                # 如果结果是numpy数组，转换为Series
                result = pd.Series(result, index=data.index)
            elif not isinstance(result, pd.Series):
                # 其他情况，尝试转换为Series
                try:
                    result = pd.Series(result, index=data.index)
                except:
                    logger.error(f"Cannot convert result to Series for formula: {formula}")
                    return pd.Series(np.nan, index=data.index)

            # 替换无限值为NaN
            result = result.replace([np.inf, -np.inf], np.nan)

            return result

        except Exception as e:
            # 如果pd.eval失败，尝试使用标准eval作为后备方案
            if "scalar" in str(e).lower():
                try:
                    # 创建一个更受限的环境用于标准eval
                    eval_dict = {"__builtins__": {}}
                    eval_dict.update(safe_dict)

                    # 执行评估
                    result = eval(sanitized_formula, eval_dict)

                    # 确保返回Series
                    if isinstance(result, pd.Series):
                        return result.replace([np.inf, -np.inf], np.nan)
                    else:
                        # 如果结果是标量或数组，转换为Series
                        return pd.Series(result, index=data.index).replace([np.inf, -np.inf], np.nan)

                except Exception as eval_error:
                    logger.error(f"Both pd.eval and eval failed for formula '{formula}': {eval_error}")
                    return pd.Series(np.nan, index=data.index)
            else:
                raise e

    except Exception as e:
        logger.error(f"Error evaluating formula '{formula}': {e}")
        return pd.Series(np.nan, index=data.index)