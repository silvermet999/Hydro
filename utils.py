from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import torch

cuda = True if torch.cuda.is_available() else False

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


def dataset_function(dataset, batch_size, train=True):
    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)

    train_subset = Subset(dataset, range(0, train_size))
    val_subset = Subset(dataset, range(train_size, train_size + val_size))
    test_subset = Subset(dataset, range(train_size + val_size, total_size))

    if train:
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
        validation_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
        return train_loader, validation_loader

    else:
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
        return test_loader

def dataset_function_OOD(ID_dataset, OOD, batch_size, train=True):
    if train:
        train_loader = DataLoader(ID_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
        train_loader_ood = DataLoader(OOD, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
        return train_loader, train_loader_ood

    else:
        test_loader = DataLoader(ID_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
        ood_test_loader = DataLoader(OOD, batch_size=batch_size, shuffle=False)
        return test_loader, ood_test_loader