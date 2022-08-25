import torch
import torchvision
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader

dataset = torchvision.datasets.MNIST(
    root='./data', transform=torchvision.transforms.ToTensor()
)

data, label = dataset
print(data.shape)
