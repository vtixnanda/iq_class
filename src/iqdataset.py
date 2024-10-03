import torch
from torch.utils.data import Dataset, DataLoader, random_split
import json
import numpy as np
import sys, os

class IQDataset(Dataset):
    def __init__(self, dictionary, iq_dir, indices, transform=None, target_transform=None):
        self.iq_labels = dictionary
        self.iq_dir = iq_dir
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

        # splits one data file into multiple samples, but associated with the same file
        self.splits = 640
        self.chunk = 1024

    def __len__(self):
        return len(self.iq_labels) * self.splits

    def __getitem__(self, ind):
        # identifies which file to load
        idx = int(np.floor(ind / self.splits))

        # identifies section of data to look at
        sect = ind % self.splits

        iq_data = os.path.join(self.iq_dir, self.iq_labels[self.indices[idx]][0])
        iq_meta = os.path.join(self.iq_dir, self.iq_labels[self.indices[idx]][1])
        iq = np.memmap(iq_data, mode="r", dtype=np.complex128)

        real = np.real(iq[sect*self.chunk : (sect+1)*self.chunk])
        imag = np.imag(iq[sect*self.chunk : (sect+1)*self.chunk])
        real = torch.tensor(real).float()
        imag = torch.tensor(imag).float()

        data = torch.stack((real, imag), dim=1)
        data = torch.transpose(data, 0, 1)

        with open(iq_meta, "r") as f:
            meta = json.loads(f.read())

        label = self.indices[idx]

        return data, idx