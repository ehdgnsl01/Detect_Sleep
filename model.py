# model.py

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # conv3 이후 feature map 크기: (128, 3, 4) → flatten 크기 = 128×3×4 = 1536
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        # (batch, 1, 26, 34) → conv1 → ReLU → maxpool → (batch,32,13,17)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # (batch,32,13,17) → conv2 → ReLU → maxpool → (batch,64,6,8)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # (batch,64,6,8) → conv3 → ReLU → maxpool → (batch,128,3,4)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # (batch,128,3,4) → flatten → (batch,1536)
        x = x.reshape(x.size(0), -1)  # view 대신 reshape 사용
        # (batch,1536) → fc1 → ReLU → (batch,512)
        x = F.relu(self.fc1(x))
        # (batch,512) → fc2 → (batch,1)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    # 입력 채널=1, 높이=26, 너비=34
    summary(model, (1, 26, 34))
