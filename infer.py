import torch
from torch.utils.data import DataLoader, Dataset

from convnet import ConvNet
from torchsummary import summary

import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

model = torch.load('model.pth').to(device)

summary(model, (1, 28, 28))


class KaggleTestDigits(Dataset):
    def __init__(self, path: str):
        df = pd.read_csv(path)

        self.data = df.iloc[:, :].values

        self.data = torch.from_numpy(self.data).float().reshape(-1, 1, 28, 28)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


test_dataset = KaggleTestDigits('data/test.csv')
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

results = pd.DataFrame(columns=['ImageId', 'Label'])

for i, data in enumerate(test_loader):
    data = data.to(device)

    prediction_batch = model(data)

    prediction_batch = torch.argmax(prediction_batch, dim=1)

    for j, prediction in enumerate(prediction_batch):
        results = pd.concat([results, pd.DataFrame({'ImageId': [i * 64 + j + 1], 'Label': [prediction.item()]})])


results.to_csv('results.csv', index=False)
