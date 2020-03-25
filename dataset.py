import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class PegasusDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train, transform, target_transform, download)

        valid_classes = [0, 2, 4, 7] # index of birds, horses, deers and planes

        pegasus_data = [self.data[i] for i in range(len(self.targets)) if self.targets[i] in valid_classes]

        pegasus_targets = [self.targets[i] for i in range(len(self.targets)) if self.targets[i] in valid_classes]

        self.data = pegasus_data
        self.targets = pegasus_targets
