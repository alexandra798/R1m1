"""风险寻求策略网络模块"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """策略网络：GRU特征提取器 + MLP策略头"""

    def __init__(self, input_dim, gru_hidden_dim=64, gru_layers=4,
                 mlp_hidden_dim=32, mlp_layers=2, output_dim=100):
        super(PolicyNetwork, self).__init__()

        # GRU特征提取器（论文：4层，隐藏维度64）
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.1
        )

        # MLP策略头（论文：2个隐藏层，32个神经元）
        mlp_layers_list = []
        mlp_layers_list.append(nn.Linear(gru_hidden_dim, mlp_hidden_dim))
        mlp_layers_list.append(nn.ReLU())
        mlp_layers_list.append(nn.Dropout(0.1))

        for _ in range(mlp_layers - 1):
            mlp_layers_list.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim))
            mlp_layers_list.append(nn.ReLU())
            mlp_layers_list.append(nn.Dropout(0.1))

        mlp_layers_list.append(nn.Linear(mlp_hidden_dim, output_dim))
        mlp_layers_list.append(nn.Softmax(dim=-1))

        self.mlp = nn.Sequential(*mlp_layers_list)

    def forward(self, x):
        # GRU处理序列特征
        gru_out, _ = self.gru(x)
        # 取最后一个时间步的输出
        last_hidden = gru_out[:, -1, :]
        # MLP生成动作概率分布
        action_probs = self.mlp(last_hidden)
        return action_probs


class RiskSeekingPolicyOptimizer:
    """风险寻求策略优化器"""

    def __init__(self, policy_network, quantile_alpha=0.85,
                 learning_rate_beta=0.01, learning_rate_gamma=0.001):
        self.policy_network = policy_network
        self.quantile_alpha = quantile_alpha
        self.beta = learning_rate_beta  # 分位数回归学习率
        self.gamma = learning_rate_gamma  # 网络参数更新学习率

        self.optimizer = optim.Adam(policy_network.parameters(), lr=self.gamma)
        self.replay_buffer = deque(maxlen=10000)
        self.current_quantile = 0.0

    def update_quantile(self, reward):
        """更新分位数估计（论文公式11）"""
        self.current_quantile = self.current_quantile + self.beta * (
                1 - self.quantile_alpha - int(reward <= self.current_quantile)
        )
        return self.current_quantile

    def compute_policy_gradient(self, trajectory):
        """计算策略梯度（论文定理4.1）"""
        states, actions, rewards = trajectory
        cumulative_reward = sum(rewards)

        # 风险寻求：只在超过分位数阈值时更新
        if cumulative_reward > self.current_quantile:
            # 构建梯度方向
            log_probs = []
            for state, action in zip(states, actions):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = self.policy_network(state_tensor)
                log_prob = torch.log(action_probs[0, action])
                log_probs.append(log_prob)

            # 策略梯度损失
            policy_loss = -sum(log_probs)
            return policy_loss
        return None

    def train_on_batch(self, trajectories):
        """批量训练策略网络"""
        total_loss = 0
        update_count = 0

        for trajectory in trajectories:
            # 更新分位数
            _, _, rewards = trajectory
            cumulative_reward = sum(rewards)
            self.update_quantile(cumulative_reward)

            # 计算策略梯度
            loss = self.compute_policy_gradient(trajectory)
            if loss is not None:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                update_count += 1

        if update_count > 0:
            avg_loss = total_loss / update_count
            logger.info(f"Policy network updated. Avg loss: {avg_loss:.4f}, "
                        f"Current quantile: {self.current_quantile:.4f}")

        return total_loss / max(update_count, 1)

    def add_trajectory(self, trajectory):
        """添加轨迹到回放缓冲区"""
        self.replay_buffer.append(trajectory)

    def get_action_probabilities(self, state):
        """获取动作概率分布"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.policy_network(state_tensor)
        return action_probs.numpy()[0]