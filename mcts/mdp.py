"""奖励密集型MDP实现"""
import numpy as np
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)


class RewardDenseMDP:
    """奖励密集型MDP"""

    def __init__(self, alpha_pool, lambda_param=0.1):
        self.alpha_pool = alpha_pool
        self.lambda_param = lambda_param
        self.episode_max_length = 30

    def calculate_intermediate_reward(self, formula, X, y, evaluate_func):
        """
        计算中间奖励（论文公式5）
        Reward_inter = IC - λ * (1/k) * Σ mutIC_i
        """
        # 计算IC
        feature = evaluate_func(formula, X)
        valid_indices = ~(feature.isna() | y.isna())

        if valid_indices.sum() < 2:
            return 0.0

        ic, _ = spearmanr(feature[valid_indices], y[valid_indices])
        if np.isnan(ic):
            ic = 0.0

        # 如果alpha池为空，直接返回IC
        if len(self.alpha_pool.alpha_pool) == 0:
            return ic

        # 计算与池中其他alpha的平均mutIC
        mutic_sum = 0
        for alpha in self.alpha_pool.alpha_pool:
            other_feature = evaluate_func(alpha['formula'], X)
            common_valid = ~(feature.isna() | other_feature.isna())

            if common_valid.sum() > 1:
                mutic, _ = spearmanr(
                    feature[common_valid],
                    other_feature[common_valid]
                )
                if not np.isnan(mutic):
                    mutic_sum += abs(mutic)

        avg_mutic = mutic_sum / len(self.alpha_pool.alpha_pool)

        # 计算奖励
        reward = ic - self.lambda_param * avg_mutic

        logger.debug(f"Intermediate reward for {formula}: IC={ic:.4f}, "
                     f"avg_mutIC={avg_mutic:.4f}, reward={reward:.4f}")

        return reward

    def calculate_terminal_reward(self, formula, X, y, evaluate_func):
        """
        计算终止奖励
        使用合成alpha的IC作为episode奖励
        """
        # 将公式添加到池中
        self.alpha_pool.add_to_pool({'formula': formula, 'score': 0})

        # 更新池
        self.alpha_pool.update_pool(X, y, evaluate_func)

        # 计算合成alpha的IC（简化版本）
        # 实际应该训练线性模型来合成，这里用平均IC代替
        total_ic = 0
        for alpha in self.alpha_pool.alpha_pool[:10]:  # 取前10个
            if 'adjusted_ic' in alpha:
                total_ic += alpha['adjusted_ic']

        composite_ic = total_ic / min(len(self.alpha_pool.alpha_pool), 10)

        logger.debug(f"Terminal reward (composite IC): {composite_ic:.4f}")

        return composite_ic