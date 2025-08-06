"""MCTS节点类定义 - 支持状态而非公式字符串"""
import numpy as np
import math


class MCTSNode:
    """蒙特卡洛树搜索节点 - 基于状态的新版本"""

    def __init__(self, state=None, parent=None, action=None, prior_prob=1.0):
        """
        初始化MCTS节点

        Args:
            state: MDPState对象，表示当前状态
            parent: 父节点
            action: 从父节点到达此节点的动作（Token名称）
            prior_prob: 策略网络给出的先验概率
        """
        self.state = state
        self.parent = parent
        self.action = action  # 到达此节点的动作
        self.prior_prob = prior_prob  # P(s,a)

        self.children = {}  # {action: child_node}
        self.visits = 0
        self.total_value = 0.0
        self.q_value = 0.0  # Q(s,a) = total_value / visits

    def is_expanded(self):
        """检查节点是否已展开"""
        return len(self.children) > 0

    def is_terminal(self):
        """检查是否为终止节点"""
        if self.state is None:
            return False
        # 检查最后一个token是否为END
        if len(self.state.token_sequence) > 0:
            return self.state.token_sequence[-1].name == 'END'
        return False

    def is_fully_expanded(self):
        """检查节点是否已完全展开（所有合法动作都有子节点）"""
        if self.state is None:
            return False
        from .token_system import RPNValidator
        valid_actions = RPNValidator.get_valid_next_tokens(self.state.token_sequence)
        return all(action in self.children for action in valid_actions)

    def add_child(self, action, child_state, prior_prob=1.0):
        """添加子节点"""
        child = MCTSNode(
            state=child_state,
            parent=self,
            action=action,
            prior_prob=prior_prob
        )
        self.children[action] = child
        return child

    def update(self, value):
        """更新节点的访问次数和值"""
        self.visits += 1
        self.total_value += value
        self.q_value = self.total_value / self.visits

    def get_best_child(self, c_puct=1.0):
        """
        使用PUCT公式选择最佳子节点
        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        if not self.children:
            return None

        total_visits = sum(child.visits for child in self.children.values())
        sqrt_total = math.sqrt(total_visits)

        best_value = -float('inf')
        best_child = None

        for action, child in self.children.items():
            # Q值：平均奖励
            q_value = child.q_value

            # U值：探索奖励
            u_value = c_puct * child.prior_prob * sqrt_total / (1 + child.visits)

            # PUCT值
            puct_value = q_value + u_value

            if puct_value > best_value:
                best_value = puct_value
                best_child = child

        return best_child

    def get_visit_distribution(self):
        """获取子节点的访问次数分布（用于最终动作选择）"""
        actions = list(self.children.keys())
        visits = [self.children[a].visits for a in actions]
        return actions, visits

    def __repr__(self):
        """节点的字符串表示"""
        if self.state:
            formula = ' '.join([t.name for t in self.state.token_sequence[1:]])
            return f"MCTSNode(formula='{formula}', visits={self.visits}, Q={self.q_value:.4f})"
        return f"MCTSNode(root, visits={self.visits})"