"""MCTS搜索算法实现 - 支持Token系统和传统方式"""
import numpy as np
import logging
import warnings
from scipy.stats import spearmanr, ConstantInputWarning

# 新系统导入
from .trainer import RiskMinerTrainer
from .mdp_environment import AlphaMiningMDP, MDPState
from .mcts_searcher import MCTSSearcher
from .node import MCTSNode as NewMCTSNode

# 保留旧系统的部分功能用于兼容
from alpha.pool import AlphaPool
from .formula_generator import generate_formula

warnings.filterwarnings('ignore', category=ConstantInputWarning)
logger = logging.getLogger(__name__)


def run_mcts_with_token_system(X_train, y_train, num_iterations=200,
                               use_policy_network=True, num_simulations=50, device=None):
    """
    使用新的Token系统运行MCTS

    Args:
        X_train: 训练数据特征
        y_train: 训练数据标签
        num_iterations: 训练迭代次数
        use_policy_network: 是否使用策略网络
        num_simulations: 每次迭代的模拟次数
        device: torch设备(cuda或cpu)

    Returns:
        top_formulas: 最佳公式列表
    """
    logger.info("Starting MCTS with Token System")

    # 创建训练器
    trainer = RiskMinerTrainer(X_train, y_train, use_policy_network=use_policy_network, device=device)

    # 训练
    trainer.train(
        num_iterations=num_iterations,
        num_simulations_per_iteration=num_simulations
    )

    # 获取最佳公式
    top_formulas = trainer.get_top_formulas(n=5)

    # 转换为兼容格式（formula, score）
    result = []
    for formula in top_formulas:
        # 计算IC作为分数
        if trainer.alpha_pool:
            matching_alpha = next((a for a in trainer.alpha_pool if a['formula'] == formula), None)
            if matching_alpha:
                result.append((formula, matching_alpha['ic']))
            else:
                result.append((formula, 0.0))
        else:
            result.append((formula, 0.0))

    return result


# ========== 以下为兼容旧系统的函数 ==========



def run_mcts_with_risk_seeking(root, X_train, y_train, all_features,
                               num_iterations, evaluate_formula_func,
                               quantile_threshold=0.85, device=None):
    """运行带风险寻求策略的MCTS"""
    logger.info("Using Token-based MCTS with Risk Seeking")
    return run_mcts_with_token_system(
        X_train, y_train,
        num_iterations=num_iterations,
        use_policy_network=True,  # 使用策略网络
        num_simulations=50,
        device=device
    )