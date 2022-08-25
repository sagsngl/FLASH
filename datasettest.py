from multiprocessing.dummy import freeze_support
import torch
import torchvision
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader


class WineDset(Dataset):
    def __init__(self):
        #data loading
        xy = np.loadtxt('wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]
        
    def __getitem__(self,index): #allows for indexng
        # dataset[0]
        return self.x[index], self.y[index]
    
    def __len__(self):
        # len(dataset)
        return self.n_samples
    
dset = WineDset()
#first_dset = dataset[0]
#features, labels = first_dset
#print(features, labels)

dataloader  = DataLoader(dataset=dset, batch_size=1, shuffle=True, num_workers=2)

# training loop
num_epochs = 2
total_samples = len(dset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)


for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        if (i+1) % 5 == 0:
            if __name__ == '__main__':
                freeze_support()
            print(f'epoch {epoch+1} of {num_epochs}, step {i+1} of {n_iterations}, inputs {inputs.shape}')
