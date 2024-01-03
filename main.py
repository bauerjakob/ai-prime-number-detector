from model.prime_number_model import PrimeNumberModel
import torch
from torch.autograd import Variable

def getInput(size, number):
    ret = []
    for x in range(size):
        if x == number - 1:
            ret.append(float(1))
        else:
            ret.append(float(0))

    return Variable(torch.tensor(ret, dtype=torch.float32))

model = PrimeNumberModel()
model = torch.load("output/innAiModel.pth")

input = getInput(1000, 953)
prediction = model(input)
print(float(prediction))


