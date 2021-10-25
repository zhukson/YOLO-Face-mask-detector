import torch
import torch.nn as nn
import torchvision


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        resnet = torchvision.models.resnet34(pretrained=True)
        resnet_out_channel = resnet.fc.in_features
        # pretrained resnet model to extract basic features such as lines and edges
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        # convolutional layer of the last four layers in yolo model.
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        # Linear model
        self.Linear_model = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 7 * 7 * 13),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.Conv_layers(x)
        x = x.view(x.size()[0], -1)
        x = self.Linear_model(x)
        return x.reshape(-1, (5*2+3), 7, 7)

x = torch.randn((5,3,448,448))
net = Model()
print(net)
y = net(x)
print(y.size())
