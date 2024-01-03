import torch.nn as nn
import torch.nn.functional as F

from appsettings import MAX_PRIME_NUMBER


class PrimeNumberModel(nn.Module):
    def __init__(self):
        super(PrimeNumberModel, self).__init__()
        self.lin1 = nn.Linear(MAX_PRIME_NUMBER, 100)
        self.lin2 = nn.Linear(100, 100)
        self.lin3 = nn.Linear(100, 50)
        self.lin4 = nn.Linear(50, 10)
        self.lin5 = nn.Linear(10, 1)
        self.lin6 = nn.Linear(1, 1)

    def forward(self, input):
        x = F.relu(self.lin1(input))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        x = F.sigmoid(self.lin6(x))
        return x
