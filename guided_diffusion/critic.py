import torch
import torch.nn as nn

# Reward Predictor
class RewardPredictor(nn.Module):
    def __init__(self, Pool2dSize=4):
        super(RewardPredictor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((Pool2dSize, Pool2dSize))

        self.fc1 = nn.Linear(32 * Pool2dSize * Pool2dSize, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
