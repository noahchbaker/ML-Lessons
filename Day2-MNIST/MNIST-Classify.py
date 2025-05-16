import numpy as np
import pandas as pd
import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToTensor
import idx2numpy


transform = transforms.Compose ([
    transforms.Resize(size = (64,64)),
    transforms.ToTensor()
])

class MNISTDataset (Dataset):
    def __init__(self, dir, transform = None):
        self.imgSet = ImageFolder(root = dir, transform=transform)
