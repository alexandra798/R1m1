"""配置文件"""

# MCTS参数
MCTS_CONFIG = {
    "num_iterations": 200,  # MCTS search cycle执行200次
    "risk_seeking_exploration": 2.0,
    "quantile_threshold": 0.85,
    "learning_rate_beta": 0.01,  # quantile regression学习率
    "learning_rate_gamma": 0.001,  # 网络参数更新学习率
    "c_puct": None  # PUCT常数
}

# Alpha池参数
ALPHA_POOL_CONFIG = {
    "pool_size": 100,  # K=100
    "lambda_param": 0.1  # λ=0.1 (reward-dense MDP)
}

# GRU特征提取器参数
GRU_CONFIG = {
    "num_layers": 4,  # 4层结构
    "hidden_dim": 64  # 隐藏层维度64
}

# Policy头参数
POLICY_CONFIG = {
    "hidden_layers": 2,  # 两个隐藏层
    "hidden_neurons": 32  # 每层32个神经元
}

# 交叉验证参数
CV_CONFIG = {
    "n_splits": 8
}

# 数据路径
DATA_CONFIG = {
    "default_data_path": "/path/to/data.csv",
    "target_column": "label_shifted"
}