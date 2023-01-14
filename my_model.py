import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(20736, 512)
        self.fc2 = nn.Linear(512, 6)

    def forward(self, x):
        x = self.relu((self.conv1(x)))
        x = self.pool(x)
        x = self.relu((self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu((self.conv3(x)))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        output = self.fc2(x)
        return output
