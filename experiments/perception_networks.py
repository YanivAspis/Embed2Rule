import torch
import torch.nn as nn
from torch.nn import functional as F

class LeNet(torch.nn.Module):
    def __init__(self, output_size, num_flat_features = 256):
        super().__init__()
        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_flat_features, out_features=120),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=84, out_features=output_size),
            torch.nn.ReLU()
        )

    def forward(self, inputs):
        out = self.convs(inputs)
        out = out.view(out.size(0), -1)
        out = self.mlp(out)
        return out


class DigitConv(nn.Module):
    def __init__(self, output_size):
        super(DigitConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, output_size)

    def forward(self, x):
        batch_size, block_size = x.shape[0], x.shape[1]
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))

        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        # return x.view(batch_size, block_size, -1)
        return x
