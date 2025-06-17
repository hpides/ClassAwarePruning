from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_CIFAR10_dataloaders(
    train_batch_size,
    test_batch_size,
    use_data_Augmentation=True,
    data_path="./data/cifar",
    download=False,
    train_shuffle=True,
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

    train_data_loader = DataLoader(
        train_data_set, batch_size=train_batch_size, shuffle=train_shuffle
    )
    test_data_loader = DataLoader(
        test_data_set, batch_size=test_batch_size, shuffle=True
    )

    return train_data_loader, test_data_loader
