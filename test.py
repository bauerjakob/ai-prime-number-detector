from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch
import json

from appsettings import MAX_PRIME_NUMBER
from data_set.prime_number_dataset import PrimeNumberDataSet
from model.prime_number_model import PrimeNumberModel
from utils.prime_utils import tensor_to_number

full_dataset = PrimeNumberDataSet(MAX_PRIME_NUMBER)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_data_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1)

data = [tensor_to_number(input) for batch_idx, (input, target) in enumerate(test_data_loader)]
with open('output/testDataset.json', 'w', encoding='utf-8') as f:
    json.dump(list(data), f, ensure_ascii=False, indent=4)

model = PrimeNumberModel()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()
epochs = 3000

def train(epoch):
    global model, optimizer, criterion

    model.train()

    for batch_idx, (input, target) in enumerate(train_data_loader):

        inputVariable = Variable(input)
        targetVariable = Variable(target)

        optimizer.zero_grad()

        out = model(inputVariable)

        loss = criterion(out, targetVariable)

        loss.backward()

        optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, batch_idx * len(input), loss.data))

        torch.save(model, "output/primeNumberModel.pth")

for epoch in range(epochs):
    train(epoch)
