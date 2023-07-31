
"""
contains functionality for creating PyTirch DataLoader's for image
classification data
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(train_dir: str,
                       test_dir: str,
                       transform: transforms.Compose,
                       batch_size: int):
    # Use ImageFolder to create datasets(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes
    
    # Turn images into DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True 
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True 
    )

    return train_dataloader, test_dataloader, class_names
