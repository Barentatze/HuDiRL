# import torch
# import torch.nn as nn
#
# # Reward Predictor
# class RewardPredictor(nn.Module):
#     def __init__(self, Pool2dSize=4):
#         super(RewardPredictor, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.pool = nn.AdaptiveAvgPool2d((Pool2dSize, Pool2dSize))
#
#         self.fc1 = nn.Linear(32 * Pool2dSize * Pool2dSize, 128)
#         self.fc2 = nn.Linear(128, 1)
#
#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.fc1(x))
#         return self.fc2(x)

import torch
import torch.nn as nn
import torchvision.models as models

class RewardPredictor(nn.Module):
    def __init__(self):
        super(RewardPredictor, self).__init__()
        # 加载 ResNet18，不加载预训练参数（你可以设为 pretrained=True）
        self.resnet = models.resnet18(pretrained=False)

        # 替换 ResNet 的全连接层为输出标量
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 1)  # 输出单个 reward 值

    def forward(self, x):
        return self.resnet(x).squeeze()

