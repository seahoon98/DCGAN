import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

'''
    PyTorch has two primitives to work with data: 
    torch.utils.data.DataLoader and torch.tuils.data.Dataset
    Dataset stores the samples and their corresponding labels
    DataLoader wraps an iterable around the Dataset
'''

'''
In this tutorial we will be using a TorchVision dataset
'''


def DownloadData(): 
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        )

    # Download testing data from open datasets.
    test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
            )

if __name__ == '__main__':
   # DownloadData()
    trainig_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
            )

