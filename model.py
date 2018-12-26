import torch

import torch.nn as nn
import torch.nn.functional as F


class PolicyModel(nn.Module):

    def __init__(self, input_shape, output_shape, hidden_units=(256, 128, 64,)):
        super().__init__()

        self.fcs = nn.ModuleList()
        for f in hidden_units:
            self.fcs.append(nn.Linear(input_shape, f))
            input_shape = f
        self.fc_last = nn.Linear(input_shape, output_shape)

        self.fc_last.weight.data.uniform_(-3e-3, 3e-3)
        self.out_fn = torch.tanh

    def forward(self, X):
        for fc in self.fcs:
            X = F.relu(fc(X))
        return self.out_fn(self.fc_last(X))


class ValueModel(nn.Module):

    def __init__(self, input_shape, hidden_units=(256, 128, 64,)):
        super().__init__()

        self.fcs = nn.ModuleList()
        for f in hidden_units:
            self.fcs.append(nn.Linear(input_shape, f))
            input_shape = f
        self.fc_last = nn.Linear(input_shape, 1)

    def forward(self, X):
        for fc in self.fcs:
            X = F.relu(fc(X))
        return self.fc_last(X)
