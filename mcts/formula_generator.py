"""公式生成模块"""
import numpy as np
import pandas as pd


def safe_divide(x, y, default_value=0):
    """
    安全除法函数，避免除零错误
    
    Parameters:
    - x: 被除数
    - y: 除数
    - default_value: 除零时的默认值
    
    Returns:
    - 除法结果或默认值
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(y) < 1e-8, default_value, x / y)
        result = np.where(np.isnan(result) | np.isinf(result), default_value, result)
    return result


def ref(x, t):
    """
    Ref operator: The value of the variable x when assessed t days prior to today.
    
    Parameters:
    - x: pandas Series with data
    - t: int, number of days to look back
    
    Returns:
    - pandas Series with shifted values
    """
    if isinstance(x, pd.Series):
        return x.shift(t)
    elif isinstance(x, pd.DataFrame):
        return x.shift(t)
    else:
        raise TypeError("ref operator requires pandas Series or DataFrame")


def csrank(x):
    """
    CSRank operator: The rank of the current stock's feature value x 
    relative to the feature values of all stocks on today's date.
    
    Parameters:
    - x: pandas Series with MultiIndex (ticker, date)
    
    Returns:
    - pandas Series with cross-sectional ranks
    """
    if isinstance(x, pd.Series):
        # 如果有多级索引，按date分组进行排名
        if isinstance(x.index, pd.MultiIndex):
            # 假设第二级索引是date
            return x.groupby(level=1).rank(pct=True)
        else:
            # 如果没有多级索引，直接返回排名
            return x.rank(pct=True)
    else:
        raise TypeError("csrank operator requires pandas Series")


# Group 2 Operators

def sign(x):
    """Sign operator: Return 1 if x is positive, otherwise return 0."""
    return np.where(x > 0, 1, 0)


def abs_op(x):
    """Abs operator: The absolute value of x."""
    return np.abs(x)


def log(x):
    """Log operator: Natural logarithmic function on x."""
    # 处理非正值，避免log错误
    return np.where(x > 0, np.log(x), np.nan)


def greater(x, y):
    """Greater operator: Return 1 if x > y, otherwise 0."""
    return np.where(x > y, 1, 0)


def less(x, y):
    """Less operator: Return 1 if x < y, otherwise 0."""
    return np.where(x < y, 1, 0)


def rank(x, t):
    """Rank operator: The rank of the present feature value compared to its values from today going back up to t days."""
    if isinstance(x, pd.Series):
        return x.rolling(window=t, min_periods=1).apply(lambda w: pd.Series(w).rank(pct=True).iloc[-1])
    else:
        raise TypeError("rank operator requires pandas Series")


def std(x, t):
    """Std operator: The standard deviation of the feature x calculated for the past t days."""
    if isinstance(x, pd.Series):
        return x.rolling(window=t, min_periods=1).std().fillna(0)
    else:
        raise TypeError("std operator requires pandas Series")


def ts_max(x, t):
    """Max operator: The maximum value of the expression x calculated on the past t days."""
    if isinstance(x, pd.Series):
        return x.rolling(window=t, min_periods=1).max()
    else:
        raise TypeError("ts_max operator requires pandas Series")


def ts_min(x, t):
    """Min operator: The minimum value of the expression x calculated on the past t days."""
    if isinstance(x, pd.Series):
        return x.rolling(window=t, min_periods=1).min()
    else:
        raise TypeError("ts_min operator requires pandas Series")


def skew(x, t):
    """Skew operator: The skewness of the feature x in past t days prior to today."""
    if isinstance(x, pd.Series):
        return x.rolling(window=t, min_periods=1).skew().fillna(0)
    else:
        raise TypeError("skew operator requires pandas Series")


def kurt(x, t):
    """Kurt operator: The kurtosis of the feature x in past t days prior to today."""
    if isinstance(x, pd.Series):
        return x.rolling(window=t, min_periods=1).kurt().fillna(0)
    else:
        raise TypeError("kurt operator requires pandas Series")


def mean(x, t):
    """Mean operator: The mean of the feature x calculated over the past t days."""
    if isinstance(x, pd.Series):
        return x.rolling(window=t, min_periods=1).mean().fillna(0)
    else:
        raise TypeError("mean operator requires pandas Series")


def med(x, t):
    """Med operator: The median of the feature x calculated over the past t days."""
    if isinstance(x, pd.Series):
        return x.rolling(window=t, min_periods=1).median()
    else:
        raise TypeError("med operator requires pandas Series")


def ts_sum(x, t):
    """Sum operator: The total sum of the feature x calculated over the past t days."""
    if isinstance(x, pd.Series):
        return x.rolling(window=t, min_periods=1).sum()
    else:
        raise TypeError("ts_sum operator requires pandas Series")


def cov(x, y, t):
    """Cov operator: The covariance between two features x and y in the past t days."""
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        return x.rolling(window=t, min_periods=1).cov(y)
    else:
        raise TypeError("cov operator requires two pandas Series")


def corr(x, y, t):
    """Corr operator: The Pearson's correlation coefficient between two features x and y in past t days."""
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        return x.rolling(window=t, min_periods=1).corr(y)
    else:
        raise TypeError("corr operator requires two pandas Series")


def decay_linear(x, t):
    """Decay_linear operator: 线性衰减加权移动平均"""
    if isinstance(x, pd.Series):
        weights = np.arange(1, t + 1)
        weights = weights / weights.sum()
        result = x.rolling(window=t, min_periods=1).apply(
            lambda w: np.dot(w[~np.isnan(w)], weights[:len(w[~np.isnan(w)])]) if len(w[~np.isnan(w)]) > 0 else 0
        )
        result = result.replace([np.inf, -np.inf], np.nan)
        return result.fillna(0)
    else:
        raise TypeError("decay_linear operator requires pandas Series")


def wma(x, t):
    """WMA operator: The weighted moving average for the variable x calculated over the past t days."""
    if isinstance(x, pd.Series):
        weights = np.arange(1, t + 1)
        weights = weights / weights.sum()

        result = x.rolling(window=t, min_periods=1).apply(
            lambda w: np.dot(w[~np.isnan(w)], weights[:len(w[~np.isnan(w)])]) if len(w[~np.isnan(w)]) > 0 else 0
        )
        result = result.replace([np.inf, -np.inf], np.nan)
        return result.fillna(0)
    else:
        raise TypeError("wma operator requires pandas Series")


def ema(x, t):
    """EMA operator: The exponential moving average for the variable x calculated over the past t days."""
    if isinstance(x, pd.Series):
        return x.ewm(span=t, adjust=False).mean()
    else:
        raise TypeError("ema operator requires pandas Series")


def generate_formula(all_features):
    """
    生成随机公式

    Parameters:
    - all_features: 可用特征列表

    Returns:
    - formula: 生成的公式字符串
    """
    # 基础二元运算符
    binary_operators = ['+', '-', '*', '/']
    
    # 时间窗口选项
    time_deltas = [1, 5, 10, 20, 30, 40, 50]
    
    # 随机选择公式类型，增加更多类型
    formula_types = ['binary', 'unary', 'time_series', 'comparison', 'correlation', 'mixed']
    probabilities = [0.15, 0.15, 0.25, 0.1, 0.1, 0.25]
    formula_type = np.random.choice(formula_types, p=probabilities)
    
    if formula_type == 'binary':
        # 基础二元运算
        feature1 = np.random.choice(all_features)
        feature2 = np.random.choice(all_features)
        operator = np.random.choice(binary_operators)
        
        if operator == '/':
            formula = f"safe_divide({feature1}, {feature2})"
        else:
            formula = f"{feature1} {operator} {feature2}"
    
    elif formula_type == 'unary':
        # 单目运算符
        feature = np.random.choice(all_features)
        unary_op = np.random.choice(['sign', 'abs_op', 'log', 'csrank'])
        
        if unary_op == 'log':
            # Log需要特殊处理，确保输入为正
            formula = f"log(abs_op({feature}) + 1)"
        else:
            formula = f"{unary_op}({feature})"
    
    elif formula_type == 'time_series':
        # 时序运算符
        feature = np.random.choice(all_features)
        t = np.random.choice(time_deltas)
        ts_op = np.random.choice(['ref', 'rank', 'std', 'ts_max', 'ts_min', 'skew', 
                                  'kurt', 'mean', 'med', 'ts_sum', 'decay_linear', 'wma', 'ema'])
        
        if ts_op == 'ref':
            formula = f"ref({feature}, {t})"
        else:
            formula = f"{ts_op}({feature}, {t})"
    
    elif formula_type == 'comparison':
        # 比较运算符
        feature1 = np.random.choice(all_features)
        feature2 = np.random.choice(all_features)
        comp_op = np.random.choice(['greater', 'less'])
        formula = f"{comp_op}({feature1}, {feature2})"
    
    elif formula_type == 'correlation':
        # 相关性运算符
        feature1 = np.random.choice(all_features)
        feature2 = np.random.choice(all_features)
        t = np.random.choice(time_deltas)
        corr_op = np.random.choice(['cov', 'corr'])
        formula = f"{corr_op}({feature1}, {feature2}, {t})"
    
    else:  # mixed
        # 混合运算：组合不同类型的运算符
        mix_type = np.random.choice(['ts_binary', 'unary_binary', 'comparison_arithmetic'])
        
        if mix_type == 'ts_binary':
            # 时序运算与二元运算结合
            feature1 = np.random.choice(all_features)
            feature2 = np.random.choice(all_features)
            t = np.random.choice(time_deltas)
            ts_op = np.random.choice(['mean', 'std', 'ts_max', 'ts_min'])
            operator = np.random.choice(binary_operators)
            
            if operator == '/':
                formula = f"safe_divide({ts_op}({feature1}, {t}), {ts_op}({feature2}, {t}))"
            else:
                formula = f"{ts_op}({feature1}, {t}) {operator} {ts_op}({feature2}, {t})"
        
        elif mix_type == 'unary_binary':
            # 单目运算与二元运算结合
            feature1 = np.random.choice(all_features)
            feature2 = np.random.choice(all_features)
            unary_op = np.random.choice(['sign', 'abs_op', 'csrank'])
            operator = np.random.choice(binary_operators)
            
            if operator == '/':
                formula = f"safe_divide({unary_op}({feature1}), {unary_op}({feature2}))"
            else:
                formula = f"{unary_op}({feature1}) {operator} {unary_op}({feature2})"
        
        else:  # comparison_arithmetic
            # 比较运算与算术运算结合
            feature1 = np.random.choice(all_features)
            feature2 = np.random.choice(all_features)
            feature3 = np.random.choice(all_features)
            t = np.random.choice(time_deltas)
            
            # 例如：greater(volume, mean(volume, 20)) * close
            formula = f"greater({feature1}, mean({feature1}, {t})) * {feature2}"
    
    return formula