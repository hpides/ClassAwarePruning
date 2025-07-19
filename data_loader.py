from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List


def get_CIFAR10_dataloaders(
    train_batch_size: int,
    test_batch_size: int,
    use_data_Augmentation: bool = True,
    data_path: str = "./data/cifar",
    download: bool = False,
    train_shuffle: bool = False,
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
        indices = indices[:num_pruning_samples]
        subset_dataset = Subset(train_data_set, indices)

        subset_train_data_loader = DataLoader(
            subset_dataset,
            batch_size=train_batch_size,
            shuffle=False,  # Keep shuffle=False for deterministic ordering
        )
        return subset_train_data_loader

    train_data_loader = DataLoader(
        train_data_set, batch_size=train_batch_size, shuffle=train_shuffle
    )
    test_data_loader = DataLoader(
        test_data_set, batch_size=test_batch_size, shuffle=False
    )

    return train_data_loader, test_data_loader
