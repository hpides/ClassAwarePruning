import torch
from torch import nn
from data_loader import get_CIFAR10_dataloaders
from helpers import train_model
from torchvision import models


def main():
    model = models.vgg16(pretrained=True)  # Load a pre-trained VGG16 model
    model.classifier[6] = nn.Linear(4096, 10)  # Modify the last layer for CIFAR-10
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    batch_size = 256

    model.to(device)
    train_loader, test_loader = get_CIFAR10_dataloaders(
        train_batch_size=batch_size,
        test_batch_size=batch_size,
        use_data_Augmentation=True,
        data_path="./data/cifar",
        download=True,
        train_shuffle=True,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
    )


if __name__ == "__main__":
    main()
