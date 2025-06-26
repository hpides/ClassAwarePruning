from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from typing import List
import random


def get_CIFAR10_dataloaders(
    train_batch_size: int,
    test_batch_size: int,
    use_data_Augmentation: bool = True,
    data_path: str = "./data/cifar",
    download: bool = False,
    train_shuffle: bool = True,
    selected_classes: List[int] = [],
    num_pruning_samples: int = 512,
):

    mean = [0.4940607, 0.4850613, 0.45037037]
    std = [0.20085774, 0.19870903, 0.20153421]

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    )

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    train_data_set = datasets.CIFAR10(
        data_path,
        transform=train_transform if use_data_Augmentation else test_transform,
        download=download,
        train=True,
    )
    test_data_set = datasets.CIFAR10(
        data_path, transform=test_transform, download=download, train=False
    )

    if selected_classes:
        train_data_set.data = train_data_set.data
        train_data_set.targets = train_data_set.targets
        indices = [
            i
            for i, label in enumerate(train_data_set.targets)
            if label in selected_classes
        ]
        random.shuffle(indices)  # Shuffle to get a random subset
        indices = indices[:num_pruning_samples]
        sampler = SubsetRandomSampler(indices)
        subset_train_data_loader = DataLoader(
            train_data_set,
            batch_size=train_batch_size,
            shuffle=False,
            sampler=sampler,
        )
        return subset_train_data_loader

    train_data_loader = DataLoader(
        train_data_set, batch_size=train_batch_size, shuffle=train_shuffle
    )
    test_data_loader = DataLoader(
        test_data_set, batch_size=test_batch_size, shuffle=True
    )

    return train_data_loader, test_data_loader
