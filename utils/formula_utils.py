"""公式工具模块"""
import pandas as pd
import numpy as np


def evaluate_formula(formula, data):
    """
    评估公式的通用函数
    """
    try:
        return pd.eval(formula, local_dict=data)
    except Exception as e:
        print(f"Error evaluating formula '{formula}': {e}")
        return pd.Series(np.nan, index=data.index)