import torch


class InceptionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()

        self.path1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )

        self.path2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.path3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2),
        )

        self.path4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return torch.cat([self.path1(x), self.path2(x), self.path3(x), self.path4(x)], 1)


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.setup_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
        )

        self.inception_1 = torch.nn.Sequential(
            InceptionModule(192, 64),
            torch.nn.Dropout(0.25),
        )
        self.inception_2 = torch.nn.Sequential(
            InceptionModule(256, 128),
            torch.nn.Dropout(0.25),
        )
        self.inception_3 = torch.nn.Sequential(
            InceptionModule(512, 256),
            torch.nn.Dropout(0.25),
        )

        self.output_layers = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=1),
            torch.nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Flatten(),
            torch.nn.Linear(1600, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.setup_layers(x)
        x = self.inception_1(x)
        x = self.inception_2(x)
        x = self.inception_3(x)
        return self.output_layers(x)
