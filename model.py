import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=2)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(73984, 1024)
        self.fc2 = nn.Linear(1024, 6)

    def forward(self, x):
        x = self.relu((self.conv1(x)))
        x = self.relu((self.conv2(x)))
        x = self.pool(x)

        x = self.relu((self.conv3(x)))
        x = self.relu((self.conv4(x)))
        x = self.pool(x)

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        output = self.fc2(x)

        return output

    def predict(self, data):
        y_pred = self(data)
        return torch.argmax(y_pred, dim=1)