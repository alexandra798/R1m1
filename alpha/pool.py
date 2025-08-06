"""Alpha池管理模块"""
import numpy as np
import logging
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class AlphaPool:
    """Alpha池管理类"""

    def __init__(self, pool_size=100, lambda_param=0.5):
        self.pool_size = pool_size
        self.lambda_param = lambda_param
        self.alpha_pool = []
        self.ic_cache = {}
        self.mutic_cache = {}

    def add_to_pool(self, alpha):
        """
        添加alpha公式到池中
        如果池大小超过限制，删除最弱的alpha
        """
        self.alpha_pool.append(alpha)

        if len(self.alpha_pool) > self.pool_size:
            # 按调整后的奖励排序
            self.alpha_pool.sort(
                key=lambda x: x['score'] - self.lambda_param * self._calculate_mutual_ic(x),
                reverse=True
            )
            removed_alpha = self.alpha_pool.pop(-1)
            logger.info(f"Alpha removed from pool: {removed_alpha['formula']}")

    def _calculate_mutual_ic(self, alpha):
        """计算alpha与池中其他alpha的平均相互IC"""
        if len(self.alpha_pool) <= 1:
            return 0

        mutic_sum = 0
        count = 0
        for other_alpha in self.alpha_pool:
            if other_alpha['formula'] != alpha['formula']:
                pair_key = tuple(sorted([alpha['formula'], other_alpha['formula']]))
                if pair_key in self.mutic_cache:
                    mutic_sum += self.mutic_cache[pair_key]
                    count += 1

        return mutic_sum / max(count, 1)

    def cache_ic(self, formula, ic_value):
        """缓存IC值"""
        self.ic_cache[formula] = ic_value

    def cache_mutic(self, formula1, formula2, mutic_value):
        """缓存相互IC值"""
        self.mutic_cache[tuple(sorted([formula1, formula2]))] = mutic_value

    def update_pool(self, X_train, y_train, evaluate_formula_func):
        """
        动态更新alpha池
        """
        for alpha in self.alpha_pool:
            formula = alpha['formula']
            if formula in self.ic_cache:
                ic = self.ic_cache[formula]
            else:
                # 重新计算IC
                feature = evaluate_formula_func(formula, X_train)
                # 确保数据对齐后再使用.values
                common_index = feature.index.intersection(y_train.index)
                if len(common_index) > 0:
                    feature_aligned = feature.loc[common_index]
                    y_train_aligned = y_train.loc[common_index]
                    # 移除NaN值
                    valid_mask = ~(feature_aligned.isna() | y_train_aligned.isna())
                    if valid_mask.sum() > 1:
                        ic, _ = spearmanr(
                            feature_aligned[valid_mask].values, 
                            y_train_aligned[valid_mask].values
                        )
                    else:
                        ic = 0.0
                else:
                    ic = 0.0
                self.cache_ic(formula, ic)

            # 计算与其他alpha的相互IC
            mutic_sum = 0
            for other_alpha in self.alpha_pool:
                if other_alpha['formula'] != formula:
                    pair_key = tuple(sorted([formula, other_alpha['formula']]))
                    if pair_key in self.mutic_cache:
                        mutic = self.mutic_cache[pair_key]
                    else:
                        other_feature = evaluate_formula_func(other_alpha['formula'], X_train)
                        common_index = feature.index.intersection(other_feature.index)
                        if len(common_index) > 0:
                            feature_common = feature.loc[common_index]
                            other_feature_common = other_feature.loc[common_index]
                            # 移除NaN值
                            valid_mask = ~(feature_common.isna() | other_feature_common.isna())
                            if valid_mask.sum() > 1:
                                mutic, _ = spearmanr(
                                    feature_common[valid_mask].values,
                                    other_feature_common[valid_mask].values
                                )
                            else:
                                mutic = 0.0
                        else:
                            mutic = 0.0
                        self.cache_mutic(formula, other_alpha['formula'], mutic)
                    mutic_sum += mutic

            # 调整alpha的IC
            adjusted_ic = ic - (mutic_sum / len(self.alpha_pool))
            alpha['adjusted_ic'] = adjusted_ic

        # 按调整后的IC排序并修剪
        self.alpha_pool.sort(key=lambda x: x['adjusted_ic'], reverse=True)

        while len(self.alpha_pool) > self.pool_size:
            removed_alpha = self.alpha_pool.pop(-1)
            logger.info(f"Removed underperforming alpha: {removed_alpha['formula']}")

    def get_top_formulas(self, n=5):
        """获取前n个公式"""
        return [alpha['formula'] for alpha in self.alpha_pool[:n]]