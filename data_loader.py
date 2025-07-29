from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List


class DataLoaderFactory:

    def __init__(self, 
        train_batch_size: int,
        test_batch_size: int,
        use_data_Augmentation: bool = True,
        download: bool = True,
        train_shuffle: bool = True,
        selected_classes: List[int] = [],
        num_pruning_samples: int = 512,
    ):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.use_data_Augmentation = use_data_Augmentation
        self.download = download
        self.train_shuffle = train_shuffle
        self.selected_classes = selected_classes
        self.num_pruning_samples = num_pruning_samples

        self._initialize_datasets()

    def _initialize_datasets(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_dataloaders(self):
        train_data_loader = DataLoader(
            self.train_data_set, batch_size=self.train_batch_size, shuffle=self.train_shuffle
        )
        test_data_loader = DataLoader(
            self.test_data_set, batch_size=self.test_batch_size, shuffle=False
        )
        return train_data_loader, test_data_loader

    def _get_selected_indices(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_subset_dataloaders(self):
        if not self.selected_classes:
            return self.get_dataloaders()
        
        indices_train, indices_test = self._get_selected_indices()
        
        subset_dataset_train = Subset(self.train_data_set, indices_train)
        subset_dataset_test = Subset(self.test_data_set, indices_test)

        subset_train_data_loader = DataLoader(
            subset_dataset_train,
            batch_size=self.num_pruning_samples,
            shuffle=False,  # Keep shuffle=False for deterministic ordering
        )
        subset_test_data_loader = DataLoader(
            subset_dataset_test,
            batch_size=self.test_batch_size,
            shuffle=False,  # Keep shuffle=False for deterministic ordering
        )
        return subset_train_data_loader, subset_test_data_loader

class CIFAR10_DataLoaderFactory(DataLoaderFactory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize_datasets(self):
        mean = [0.4940607, 0.4850613, 0.45037037]
        std = [0.20085774, 0.19870903, 0.20153421]
        data_path = "./data/cifar"
        
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

        self.train_data_set = datasets.CIFAR10(
            data_path,
            transform=train_transform if self.use_data_Augmentation else test_transform,
            download=self.download,
            train=True,
        )
        self.test_data_set = datasets.CIFAR10(
            data_path, transform=test_transform, download=self.download, train=False
        )

    def _get_selected_indices(self):
        indices_train = [
            i
            for i, label in enumerate(self.train_data_set.targets)
            if label in self.selected_classes
        ]
        indices_test = [
            i
            for i, label in enumerate(self.test_data_set.targets)
            if label in self.selected_classes
        ]
        return indices_train, indices_test


class Imagenette_DataLoaderFactory(DataLoaderFactory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs) 

    def _initialize_datasets(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data_path = "./data/imagenette"

        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                #transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )

        test_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        self.train_data_set = datasets.Imagenette(data_path,
                transform=train_transform if self.use_data_Augmentation else test_transform, download=self.download, split="train",)
        self.test_data_set = datasets.Imagenette(data_path,
                transform=test_transform, download=self.download, split="val")       

    def _get_selected_indices(self):
        indices_train = [index for index, (_, label) in enumerate(self.train_data_set._samples) if label in self.selected_classes]
        indices_test = [index for index, (_, label) in enumerate(self.test_data_set._samples) if label in self.selected_classes]
        return indices_train, indices_test
    

dataloaderFactorys = {
    "cifar10": CIFAR10_DataLoaderFactory,
    "imagenette": Imagenette_DataLoaderFactory,
}