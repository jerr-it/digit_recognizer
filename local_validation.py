import torch
from torch.utils.data import DataLoader, Dataset

from convnet import ConvNet
from torchsummary import summary

import pandas as pd

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

model = torch.load('model.pth').to(device)

summary(model, (1, 28, 28))


class KaggleValidateDigits(Dataset):
    def __init__(self, path: str):
        df = pd.read_csv(path)

        df = pd.concat([df.iloc[:, -1], df.iloc[:, :-1]], axis=1)

        self.labels = df.iloc[:, 0].values
        self.data = df.iloc[:, 1:].values

        self.data = torch.from_numpy(self.data).float().reshape(-1, 1, 28, 28)

        self.labels = torch.from_numpy(self.labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


test_dataset = KaggleValidateDigits('data/mnist_784.csv')
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

correct = 0
for i, (data, label) in enumerate(test_loader):
    data = data.to(device)

    prediction_batch = model(data)

    prediction_batch = torch.argmax(prediction_batch, dim=1)

    for j, prediction in enumerate(prediction_batch):
        if prediction.item() == label[j].item():
            correct += 1
        #else:
        #    # Display the image that was incorrectly classified
        #    plt.imshow(data[j].cpu().reshape(28, 28))
        #    plt.show()

print(f'Correct: {correct} / {len(test_dataset)}')
print(f'Accuracy: {correct / len(test_dataset)}')
