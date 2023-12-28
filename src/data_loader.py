# References : https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

# License: BSD
# Author: Sasank Chilamkurthy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import random

cudnn.benchmark = True
plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def GetDataLoaders(data_dir, batch_size=4, shuffle=True, num_workers=4, validation_split=0.2, test_split=0.1):
    train_dir = os.path.join(data_dir, 'train')
    train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])

    # Calculate the number of samples for each split
    num_samples = len(train_dataset)
    num_val = int(validation_split * num_samples)
    num_test = int(test_split * num_samples)
    num_train = num_samples - num_val - num_test

    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [num_train, num_val, num_test])

    # Create data loaders for each set
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    dataset_sizes = {
        'train': num_train,
        'val': num_val,
        'test': num_test
    }
    class_names = train_dataset.classes

    return dataloaders, class_names, dataset_sizes

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

if __name__ == '__main__':
    # Add your code here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = '.\data\miniImageNet'
    dataloaders, class_names, dataset_sizes  = GetDataLoaders(data_dir)
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)