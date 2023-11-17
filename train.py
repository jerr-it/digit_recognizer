import numpy as np
import torch
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from torchsummary import summary

import matplotlib.pyplot as plt

from convnet import ConvNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))


class KaggleDigits(Dataset):
    def __init__(self, path: str):
        transforms = v2.Compose([
            v2.RandomHorizontalFlip(0.5),
            v2.RandomRotation(15),
            v2.RandomAdjustSharpness(0.5),
        ])

        df = pd.read_csv(path)

        self.labels = df.iloc[:, 0].values
        self.data = df.iloc[:, 1:].values

        # Apply the transforms to the data
        self.data = np.array([transforms(x.reshape(28, 28)) for x in self.data])

        self.data = torch.from_numpy(self.data).float().reshape(-1, 1, 28, 28)

        self.labels = torch.from_numpy(self.labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


train_dataset = KaggleDigits('data/train.csv')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = ConvNet().to(device)
summary(model, (1, 28, 28))

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.95)

epochs = 10
losses = []
for epoch in range(epochs):
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        outputs = model(data)

        labels_one_hot = torch.zeros(labels.shape[0], 10).to(device)
        labels_one_hot[torch.arange(labels.shape[0]), labels] = 1

        loss = criterion(outputs, labels_one_hot)

        # loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')
            losses.append(loss.item())


torch.save(model, 'model.pth')

plt.plot(losses)
plt.savefig('loss.png')
plt.show()

print(f'Average loss: {sum(losses[-100:]) / 100}')
