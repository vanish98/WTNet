import torch
from torch import nn


class GateFusion(nn.Module):
    def __init__(self, input_dim):
        super(GateFusion, self).__init__()
        self.gate_weight = nn.Linear(input_dim, input_dim)
        self.last_state = torch.zeros(input_dim)

    def set_last_state(self, last_stat):
        # 用于设置初始化状态
        self.last_state = last_stat

    def forward(self, feature):
        gate_weight = torch.sigmoid(self.gate_weight(self.last_state))
        feature = gate_weight * self.last_state + (1 - gate_weight) * feature
        self.last_state = feature
        return feature
