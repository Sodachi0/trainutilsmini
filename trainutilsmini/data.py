from typing import TypeAlias
from torch.utils.data import Dataset
import torch
import numpy as np

DataTensor: TypeAlias = list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor

class MiniEpoch(Dataset):
    def __init__(self, dataset, epoch_size):
        self.dataset = dataset
        
        self.size = epoch_size
        self.data_size = len(dataset)
        self.shuffle = np.arange(self.data_size)
        np.random.shuffle(self.shuffle)

        self.idx = 0
        self.data_idx = 0

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        idx = (self.data_idx + idx) % self.data_size
        x = self.dataset[self.shuffle[idx]]

        self.idx += 1
        if self.idx == self.size:
            self.data_idx += self.size
            self.data_idx %= self.data_size
            self.idx = 0

        return x
    
class KFoldDataset(Dataset):
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.data_size = len(dataset)
        self.k = k
        assert k > 1 and k <= self.data_size
        
        self.idxs = []

        q = self.data_size // k
        r = self.data_size % k

        self.steps = [q + 1] * r + [q] * (k -r)

        idx = 0
        for s in self.steps:
            self.idxs.append(idx)
            idx += s

        self.val_fold_idx = 0
        self.val_size = self.steps[self.val_fold_idx]
        self.train_size = self.data_size - self.val_size

        self.count = 0
        self.is_train = True
        self.stop = False

    def __len__(self):
        if self.is_train:
            return self.train_size
        return self.val_size

    def __getitem__(self, idx):
        if self.stop:
            self.stop = False
            raise StopIteration

        x = None
        self.count += 1
        if self.is_train:
            if idx >= self.idxs[self.val_fold_idx]:
                idx += self.steps[self.val_fold_idx]
            x = self.dataset[idx]
            if self.count == self.train_size:
                self.count = 0
                self.is_train = False
                self.stop = True
        else:
            x = self.dataset[self.idxs[self.val_fold_idx] + idx]
            if self.count == self.val_size:
                self.count = 0
                self.is_train = True
                self.val_fold_idx = (self.val_fold_idx + 1) % self.k
                self.val_size = self.steps[self.val_fold_idx]
                self.train_size = self.data_size - self.val_size
                self.stop = True
        return x
