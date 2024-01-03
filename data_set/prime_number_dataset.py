from typing import List

from torch.utils.data import Dataset
import json
import torch

import math
import numpy as np



class PrimeNumberDataSet(Dataset):
    def __init__(self, size):
        self.size = size
        self.primes = self.getPrimes()
        print(self.primes)


    def getPrimes(self):
        prime = np.zeros(self.size)
        for m in range(2, self.size):
            list = []
            dummy = m
            if math.floor(math.sqrt(m)) > 3:
                dummy = math.floor(math.sqrt(m))
            for k in range(2, dummy):
                if (m % k == 0):
                    list.append(k)
            if (len(list) == 0):
                prime[m] = 1

        return prime


    def __getitem__(self, index):
        ret = []
        for x in range(self.size):
            if x == index -1:
                ret.append(float(1))
            else:
                ret.append(float(0))

        return torch.tensor(ret), torch.tensor([self.primes[index]], dtype=torch.float32)

    def __len__(self):
        return self.size

