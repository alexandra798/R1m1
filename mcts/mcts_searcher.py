"""基于Token的MCTS搜索器"""
import numpy as np
import math
import logging
import torch
from .node import MCTSNode
from .token_system import RPNValidator, TOKEN_TO_INDEX
from .mdp_environment import MDPState

logger = logging.getLogger(__name__)


class MCTSSearcher:
    """MCTS搜索器 - 实现PUCT选择和树搜索"""

    def __init__(self, policy_network=None, c_puct=1.0, device=None):
        self.policy_network = policy_network
        self.c_puct = c_puct  # PUCT常数
        self.device = device if device else torch.device('cpu')

    def search_one_iteration(self, root_node, mdp_env, reward_calculator, X_data, y_data):
        """
        执行一次完整的MCTS迭代 - 修复版本
        """
        # 阶段1：选择（Selection）
        path = []
        current = root_node

        # 向下选择直到叶节点
        while current.is_expanded() and not current.is_terminal():
            # 使用PUCT选择最佳子节点
            current = current.get_best_child(self.c_puct)
            if current is None:
                break
            path.append(current)

        # 阶段2：扩展（Expansion）
        leaf_value = 0
        if not current.is_terminal() and current.visits >= 0:
            # 扩展节点
            leaf_value = self.expand(current, mdp_env)
            # 选择一个新扩展的子节点进行评估
            if current.children:
                # 随机选择一个子节点
                action = np.random.choice(list(current.children.keys()))
                current = current.children[action]
                path.append(current)

        # 阶段3：评估（Evaluation）
        if current.is_terminal():
            # 终止状态，计算终止奖励 - 修复：不传递None
            value = reward_calculator.calculate_terminal_reward(
                current.state, X_data, y_data
            )
        else:
            # 非终止状态，使用rollout或神经网络评估
            value = self.evaluate(current, mdp_env, reward_calculator, X_data, y_data)

        # 阶段4：回传（Backpropagation）
        self.backpropagate(path, value)

        # 返回轨迹（用于训练）
        trajectory = []
        for i, node in enumerate(path):
            if node.parent and node.action:
                # 计算即时奖励 - 修复：不传递None
                reward = reward_calculator.calculate_intermediate_reward(
                    node.state, X_data, y_data
                )
                trajectory.append((node.parent.state, node.action, reward))

        return trajectory

    def expand(self, node, mdp_env):
        """
        扩展节点，添加所有合法动作的子节点

        Returns:
            leaf_value: 叶节点的价值估计
        """
        if node.state is None:
            return 0

        # 获取所有合法动作
        valid_actions = RPNValidator.get_valid_next_tokens(node.state.token_sequence)

        if not valid_actions:
            return 0

        # 使用策略网络获取动作概率
        if self.policy_network:
            action_probs, value_estimate = self.get_policy_predictions(node.state, valid_actions)
        else:
            # 均匀分布
            action_probs = {action: 1.0 / len(valid_actions) for action in valid_actions}
            value_estimate = 0

        # 为每个合法动作创建子节点
        for action in valid_actions:
            # 创建新状态
            new_state = node.state.copy()
            new_state.add_token(action)

            # 添加子节点
            prior_prob = action_probs.get(action, 1.0 / len(valid_actions))
            node.add_child(action, new_state, prior_prob)

        return value_estimate

    def evaluate(self, node, mdp_env, reward_calculator, X_data, y_data):
        """
        评估叶节点的价值
        可以使用rollout或神经网络价值估计
        """
        if self.policy_network and hasattr(self.policy_network, 'value_head'):
            # 使用价值网络估计
            state_encoding = torch.FloatTensor(node.state.encode_for_network()).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, value = self.policy_network(state_encoding)
            return value.item()
        else:
            # 使用简单的rollout
            return self.rollout(node, mdp_env, reward_calculator, X_data, y_data)

    def rollout(self, node, mdp_env, reward_calculator, X_data, y_data, max_depth=10):
        """
        执行随机rollout来评估节点价值 - 修复版本
        """
        current_state = node.state.copy()
        total_reward = 0
        depth = 0

        while depth < max_depth and not current_state.token_sequence[-1].name == 'END':
            # 获取合法动作
            valid_actions = RPNValidator.get_valid_next_tokens(current_state.token_sequence)

            if not valid_actions:
                break

            # 随机选择动作
            action = np.random.choice(valid_actions)

            # 应用动作
            current_state.add_token(action)

            # 计算奖励 - 修复：不传递None
            if action == 'END':
                reward = reward_calculator.calculate_terminal_reward(
                    current_state, X_data, y_data
                )
            else:
                reward = reward_calculator.calculate_intermediate_reward(
                    current_state, X_data, y_data
                )

            total_reward += reward  # 无折扣
            depth += 1

            if action == 'END':
                break

        return total_reward

    def backpropagate(self, path, value):
        """
        向上回传价值
        """
        for node in reversed(path):
            node.update(value)
            if node.parent:
                node.parent.update(value)

    def get_policy_predictions(self, state, valid_actions):
        """
        使用策略网络获取动作概率分布和价值估计
        """
        if not self.policy_network:
            # 返回均匀分布
            probs = {action: 1.0 / len(valid_actions) for action in valid_actions}
            return probs, 0

        # 编码状态
        state_encoding = torch.FloatTensor(state.encode_for_network()).unsqueeze(0).to(self.device)

        # 创建合法动作掩码
        valid_actions_mask = torch.zeros(len(TOKEN_TO_INDEX), dtype=torch.bool)
        for action in valid_actions:
            valid_actions_mask[TOKEN_TO_INDEX[action]] = True
        valid_actions_mask = valid_actions_mask.unsqueeze(0).to(self.device)

        # 前向传播
        with torch.no_grad():
            action_probs, value = self.policy_network(state_encoding, valid_actions_mask)

        # 转换为字典
        probs = {}
        for action in valid_actions:
            idx = TOKEN_TO_INDEX[action]
            probs[action] = action_probs[0, idx].item()

        return probs, value.item()

    def get_best_action(self, root_node, temperature=1.0):
        """
        根据访问次数选择最佳动作

        Args:
            root_node: 根节点
            temperature: 温度参数，控制选择的随机性

        Returns:
            action: 选择的动作
        """
        if not root_node.children:
            return None

        actions, visits = root_node.get_visit_distribution()

        if temperature == 0:
            # 贪婪选择
            best_idx = np.argmax(visits)
            return actions[best_idx]
        else:
            # 根据访问次数分布采样
            visits = np.array(visits)
            if temperature != 1.0:
                visits = visits ** (1 / temperature)
            probs = visits / visits.sum()
            return np.random.choice(actions, p=probs)