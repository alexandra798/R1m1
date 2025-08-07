import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import logging
from .token_system import (
    TOKEN_DEFINITIONS, TOKEN_TO_INDEX, INDEX_TO_TOKEN,
    TOTAL_TOKENS, TokenType, RPNValidator
)
from .rpn_evaluator import RPNEvaluator

logger = logging.getLogger(__name__)


class MDPState:
    """MDP环境的状态"""

    def __init__(self):
        self.token_sequence = [TOKEN_DEFINITIONS['BEG']]
        self.step_count = 0
        self.stack_size = 0

    def add_token(self, token_name):
        """添加一个Token到序列 - 真正修正版"""
        token = TOKEN_DEFINITIONS[token_name]
        self.token_sequence.append(token)
        self.step_count += 1

        # 更新栈大小
        if token.type == TokenType.OPERAND:
            # delta是时序操作符的参数，不影响栈大小
            if not token.name.startswith('delta_'):
                self.stack_size += 1
        elif token.type == TokenType.OPERATOR:
            # 所有操作符统一处理（包括时序操作符）
            # 时序操作符现在是arity=1，消耗1个产生1个
            self.stack_size = self.stack_size - token.arity + 1


    def encode_for_network(self):
        """编码状态用于神经网络输入"""
        max_length = 30
        encoding = np.zeros((max_length, TOTAL_TOKENS + 3))

        for i, token in enumerate(self.token_sequence[:max_length]):
            if i >= max_length:
                break

            token_idx = TOKEN_TO_INDEX[token.name]
            encoding[i, token_idx] = 1
            encoding[i, TOTAL_TOKENS] = i / max_length
            encoding[i, TOTAL_TOKENS + 1] = self.stack_size / 10.0
            encoding[i, TOTAL_TOKENS + 2] = self.step_count / max_length

        return encoding

    def to_formula_string(self):
        """将Token序列转换为可读的公式字符串"""
        return RPNEvaluator.tokens_to_infix(self.token_sequence)

    def copy(self):
        """深拷贝状态"""
        new_state = MDPState()
        new_state.token_sequence = self.token_sequence.copy()
        new_state.step_count = self.step_count
        new_state.stack_size = self.stack_size
        return new_state


class AlphaMiningMDP:
    """完整的马尔可夫决策过程环境"""

    def __init__(self):
        self.max_episode_length = 30
        self.current_state = None

    def reset(self):
        """开始新的episode"""
        self.current_state = MDPState()
        return self.current_state

    def step(self, action_token):
        """执行一个动作（选择一个Token）"""
        if not self.is_valid_action(action_token):
            return self.current_state, -1.0, True

        self.current_state.add_token(action_token)

        if action_token == 'END':
            done = True
        else:
            done = False

        if self.current_state.step_count >= self.max_episode_length:
            done = True

        return self.current_state, 0.0, done

    def is_valid_action(self, action_token):
        """检查动作是否合法"""
        valid_actions = RPNValidator.get_valid_next_tokens(self.current_state.token_sequence)
        return action_token in valid_actions

    def get_valid_actions(self):
        """获取当前状态的合法动作"""
        return RPNValidator.get_valid_next_tokens(self.current_state.token_sequence)


class RewardCalculator:
    """
    基于RiskMiner论文的奖励计算器

    核心公式：
    - 中间奖励: Reward_inter = IC - λ * (1/k) * Σ mutIC_i
    - 终止奖励: Reward_end = 合成alpha的IC
    """

    def __init__(self, alpha_pool, lambda_param=0.1):
        self.alpha_pool = alpha_pool  # 外部维护的alpha池
        self.lambda_param = lambda_param  # 论文指定λ=0.1
        self.pool_size = 100  # 论文指定K=100
        self.linear_model = None

    def calculate_intermediate_reward(self, state, X_data, y_data, evaluate_func=None):
        """
        计算中间奖励（论文公式5）
        """
        # 检查是否为合法的部分表达式
        if not RPNValidator.is_valid_partial_expression(state.token_sequence):
            return -0.1

        try:
            # 评估当前部分公式
            alpha_values = self._evaluate_state(state, X_data)

            if alpha_values is None:
                return -0.1

            # 计算IC
            ic = self._calculate_ic(alpha_values, y_data)

            # 如果池为空，直接返回IC
            if len(self.alpha_pool) == 0:
                return ic

            # 计算与池中alpha的平均mutIC
            mut_ic_sum = 0
            valid_count = 0

            for alpha in self.alpha_pool:
                if 'values' in alpha and alpha['values'] is not None:
                    mut_ic = self._calculate_mutual_ic(alpha_values, alpha['values'])
                    if not np.isnan(mut_ic):
                        mut_ic_sum += abs(mut_ic)
                        valid_count += 1

            # 按照论文公式计算
            if valid_count > 0:
                avg_mut_ic = mut_ic_sum / valid_count
                reward = ic - self.lambda_param * avg_mut_ic
            else:
                reward = ic

            logger.debug(f"Intermediate: IC={ic:.4f}, reward={reward:.4f}")
            return reward

        except Exception as e:
            logger.error(f"Error in intermediate reward: {e}")
            return -0.1

    def calculate_terminal_reward(self, state, X_data, y_data, evaluate_func=None):
        """
        计算终止奖励（合成alpha的IC）
        """
        # 验证是否正确终止
        if state.token_sequence[-1].name != 'END':
            return -1.0

        try:
            # 评估完整公式
            alpha_values = self._evaluate_state(state, X_data)

            if alpha_values is None:
                return -0.5

            # 计算个体IC
            individual_ic = self._calculate_ic(alpha_values, y_data)

            # 生成公式字符串
            formula_str = RPNEvaluator.tokens_to_infix(state.token_sequence)

            # 添加到池中
            new_alpha = {
                'formula': formula_str,
                'values': alpha_values,
                'ic': individual_ic,
                'weight': 1.0
            }

            # 检查是否已存在
            exists = any(a.get('formula') == formula_str for a in self.alpha_pool)
            if not exists:
                self.alpha_pool.append(new_alpha)

                # 维护池大小
                if len(self.alpha_pool) > self.pool_size:
                    self.alpha_pool.sort(key=lambda x: abs(x.get('ic', 0)), reverse=True)
                    self.alpha_pool = self.alpha_pool[:self.pool_size]

            # 计算合成IC（论文的核心）
            composite_ic = self._calculate_composite_ic(y_data)

            logger.info(f"Terminal: individual_IC={individual_ic:.4f}, composite_IC={composite_ic:.4f}")

            # 返回合成IC作为终止奖励
            return composite_ic

        except Exception as e:
            logger.error(f"Error in terminal reward: {e}")
            return -0.5

    def _evaluate_state(self, state, X_data):
        """评估状态对应的公式值 - 修复版"""
        try:
            # 将数据转换为字典格式
            if hasattr(X_data, 'to_dict'):
                data_dict = X_data.to_dict('series')
            else:
                data_dict = X_data

            # 判断是否为部分表达式（未以END结束）
            is_partial = (len(state.token_sequence) == 0 or
                          state.token_sequence[-1].name != 'END')

            # 使用RPN求值器评估，传递allow_partial参数
            result = RPNEvaluator.evaluate(
                state.token_sequence,
                data_dict,
                allow_partial=is_partial
            )

            if result is not None:
                if hasattr(result, 'values'):
                    return result.values
                else:
                    return np.array(result)
            return None

        except Exception as e:
            logger.error(f"Error evaluating state: {e}")
            return None

    def _calculate_ic(self, predictions, targets):
        """计算IC（Pearson相关系数）- 修正版"""
        try:
            # 处理predictions
            if hasattr(predictions, 'values'):
                predictions = predictions.values
            if hasattr(targets, 'values'):
                targets = targets.values

            # 确保是numpy数组
            predictions = np.array(predictions).flatten()
            targets = np.array(targets).flatten()

            # 检查长度
            if len(predictions) == 1 and len(targets) > 1:
                # predictions是标量，扩展为向量
                predictions = np.full(len(targets), predictions[0])
            elif len(targets) == 1 and len(predictions) > 1:
                # targets是标量（不应该发生）
                logger.error("Targets is scalar, this should not happen")
                return 0.0
            elif len(predictions) != len(targets):
                # 长度不匹配，取最小长度
                min_len = min(len(predictions), len(targets))
                predictions = predictions[:min_len]
                targets = targets[:min_len]

            # 移除NaN
            valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
            if valid_mask.sum() < 2:
                return 0.0

            corr, _ = pearsonr(predictions[valid_mask], targets[valid_mask])
            return corr if not np.isnan(corr) else 0.0

        except Exception as e:
            logger.error(f"Error calculating IC: {e}")
            return 0.0

    def _calculate_mutual_ic(self, alpha1_values, alpha2_values):
        """计算两个alpha的相互IC"""
        try:
            if hasattr(alpha1_values, 'values'):
                alpha1_values = alpha1_values.values
            if hasattr(alpha2_values, 'values'):
                alpha2_values = alpha2_values.values

            alpha1 = np.array(alpha1_values).flatten()
            alpha2 = np.array(alpha2_values).flatten()

            min_len = min(len(alpha1), len(alpha2))
            alpha1 = alpha1[:min_len]
            alpha2 = alpha2[:min_len]

            valid_mask = ~(np.isnan(alpha1) | np.isnan(alpha2))
            if valid_mask.sum() < 2:
                return 0.0

            corr, _ = pearsonr(alpha1[valid_mask], alpha2[valid_mask])
            return corr if not np.isnan(corr) else 0.0

        except Exception as e:
            logger.error(f"Error calculating mutual IC: {e}")
            return 0.0

    def _calculate_composite_ic(self, y_data):
        """
        计算合成alpha的IC（论文Algorithm 1）
        使用线性回归组合所有alpha
        """
        if len(self.alpha_pool) == 0:
            return 0.0

        try:
            # 筛选有效alpha
            valid_alphas = [a for a in self.alpha_pool
                            if 'values' in a and a['values'] is not None]

            if len(valid_alphas) == 0:
                return 0.0

            # 如果只有一个alpha，直接返回其IC
            if len(valid_alphas) == 1:
                return valid_alphas[0].get('ic', 0)

            # 构建特征矩阵
            feature_matrix = []
            for alpha in valid_alphas:
                values = alpha['values']
                if hasattr(values, 'values'):
                    values = values.values
                feature_matrix.append(np.array(values).flatten())

            feature_matrix = np.column_stack(feature_matrix)

            # 准备目标数据
            if hasattr(y_data, 'values'):
                y_array = y_data.values
            else:
                y_array = np.array(y_data).flatten()

            # 对齐长度
            min_len = min(len(feature_matrix), len(y_array))
            feature_matrix = feature_matrix[:min_len]
            y_array = y_array[:min_len]

            # 移除NaN
            valid_mask = ~(np.any(np.isnan(feature_matrix), axis=1) | np.isnan(y_array))

            if valid_mask.sum() < 10:
                # 数据太少，返回平均IC
                return np.mean([a.get('ic', 0) for a in valid_alphas])

            # 训练线性模型（论文的核心）
            self.linear_model = LinearRegression(fit_intercept=False)
            self.linear_model.fit(feature_matrix[valid_mask], y_array[valid_mask])

            # 更新权重
            weights = self.linear_model.coef_
            for i, alpha in enumerate(valid_alphas):
                if i < len(weights):
                    alpha['weight'] = weights[i]

            # 计算合成预测
            composite_predictions = self.linear_model.predict(feature_matrix[valid_mask])

            # 计算合成IC
            composite_ic = self._calculate_ic(composite_predictions, y_array[valid_mask])

            return composite_ic

        except Exception as e:
            logger.error(f"Error in composite IC: {e}")
            return np.mean([a.get('ic', 0) for a in self.alpha_pool if 'ic' in a])