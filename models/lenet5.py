from torch.nn import Module
from torch import nn
from utils.builder import get_builder


class LeNet5(Module):
    def __init__(self, a = 6, b = 16, c = 120, d = 84):
        super(LeNet5, self).__init__()
        builder = get_builder()
        self.conv1 = builder.conv5x5(1, a)
        # self.conv1 = nn.Conv2d(1, 6, 5, bias=False)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = builder.conv5x5(a, b)
        # self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        # self.fc1 = nn.Linear(256, 120, bias=False)
        self.fc1 = builder.conv1x1(b*49, c)
        self.relu3 = nn.ReLU()
        # self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc2 = builder.conv1x1(c, d)
        self.relu4 = nn.ReLU()
        # self.fc3 = nn.Linear(84, 10, bias=False)
        self.fc3 = builder.conv1x1(d, 10)
        self.relu5 = nn.ReLU()
        self.b49 = b*49

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], self.b49, 1, 1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y.squeeze()