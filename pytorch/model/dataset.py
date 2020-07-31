import numpy as np
import torch


class IrisDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.data = torch.from_numpy(X.astype('float32'))
        self.targets = torch.from_numpy(y.astype('float32')).long()

    def _ohe(self, targets):
        y = np.zeros((150, 3))
        for i, label in enumerate(targets):
            y[i, label] = 1.0
        return y

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)