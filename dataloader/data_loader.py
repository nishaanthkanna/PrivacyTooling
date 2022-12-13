import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# Torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms



def get_dataloaders(dataset, batch_size):
    """Create train test dataloaders"""

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    if dataset == 'cifar10':
        training_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        
    elif dataset == 'cifar100':
        training_dataset = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    train_dl = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dl  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_dl, test_dl
        
