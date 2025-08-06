"""回测模块"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import logging


logger = logging.getLogger(__name__)


def eval_formula(formula, X):
    """
    评估公式（用于回测）
    """
    try:
        if 'delay' in formula:
            # 处理延迟函数
            try:
                parts = formula.split('(')[1].split(')')[0].split(',')
                if len(parts) != 2:
                    raise ValueError(f"Invalid delay formula format: {formula}")
                column, delay_str = parts[0].strip(), parts[1].strip()
                delay = int(delay_str)
                if column not in X.columns:
                    raise ValueError(f"Column '{column}' not found in dataset")
                return X[column].shift(delay)
            except (ValueError, IndexError) as e:
                logger.error(f"Invalid delay format in formula '{formula}': {e}")
                return pd.Series(np.nan, index=X.index)
        
        # 创建安全的评估环境
        safe_dict = X.copy()
        safe_dict['safe_divide'] = safe_divide
        return pd.eval(formula, local_dict=safe_dict)
    except Exception as e:
        logger.error(f"Error evaluating formula '{formula}': {e}")
        return pd.Series(np.nan, index=X.index)


def backtest_formulas(formulas, X_test, y_test):
    """
    回测已发现的公式

    Parameters:
    - formulas: 要测试的公式列表
    - X_test: 测试特征数据
    - y_test: 测试目标数据

    Returns:
    - results: 公式及其IC值的字典
    """
    results = {}

    for formula in formulas:
        # 评估公式
        feature = eval_formula(formula, X_test)

        # 对齐评估的特征与y_test
        valid_indices = ~(feature.isna() | y_test.isna())
        feature_clean = feature[valid_indices]
        y_test_clean = y_test[valid_indices]

        # 确保有足够的数据计算IC
        if len(feature_clean) > 1:
            ic, _ = spearmanr(feature_clean, y_test_clean)
            results[formula] = ic if not np.isnan(ic) else 0
        else:
            results[formula] = 0

    return results