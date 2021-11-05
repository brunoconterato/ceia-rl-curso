import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=0):
        """
        Construa uma rede neural fully connected

        Parameters
        ----------
        state_size (int): Dimensão do estado
        action_size (int): Dimensão da ação
        seed (int): random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 336)
        self.fc2 = nn.Linear(336, 512)
        self.fc3 = nn.Linear(512, action_size)

    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Essa variável armazena o índice desta célula para facilitar a exportação do seu agente treinado.
try:
    _q_cell = len(_ih) - 1
except:
    pass