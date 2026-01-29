from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from typing import List
import random
from collections import defaultdict
import math


class RemappedSubset(Dataset):
    """Wrapper that remaps labels to consecutive indices."""

    def __init__(self, dataset, indices, original_classes):
        self.dataset = dataset
        self.indices = indices
        # Create mapping from original class indices to new consecutive indices
        self.label_map = {orig_class: new_idx for new_idx, orig_class in enumerate(sorted(original_classes))}
        print(f"Label remapping: {self.label_map}", flush=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data, original_label = self.dataset[self.indices[idx]]
        # Remap the label to consecutive index
        new_label = self.label_map[original_label]
        return data, new_label


class DataLoaderFactory:

    def __init__(self,
                 train_batch_size: int,
                 test_batch_size: int,
                 use_data_augmentation: bool = False,
                 download: bool = True,
                 train_shuffle: bool = True,
                 selected_classes: List[int] = [],
                 num_pruning_samples: int = 512,
                 use_imagenet_labels: bool | None = False,
                 subsample_ratio: float = None,
                 subsample_size_per_class: int = None,
                 ):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.use_data_Augmentation = use_data_augmentation
        self.download = download
        self.train_shuffle = train_shuffle
        self.selected_classes = selected_classes
        self.num_pruning_samples = num_pruning_samples
        self.use_imagenet_labels = use_imagenet_labels
        self.subsample_ratio = subsample_ratio
        self.subsample_size_per_class = subsample_size_per_class
        print(f"SAMPLING: subsample_ratio={subsample_ratio}, subsample_size_per_class={subsample_size_per_class}")

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

    def get_small_train_loader(self):
        indices = list(range(len(self.train_data_set)))
        random.seed(42)
        random.shuffle(indices)
        indices = indices[:self.num_pruning_samples]
        subset_dataset_train = Subset(self.train_data_set, indices)
        return DataLoader(
            subset_dataset_train,
            batch_size=self.train_batch_size,
            shuffle=False,  # Keep shuffle=False for deterministic ordering
        )

    def _get_selected_indices(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    '''
    def get_subset_dataloaders(self):
        indices_train, indices_test = self._get_selected_indices()  # Get indices that match the selected classes

        subset_dataset_train = Subset(self.train_data_set, indices_train)
        subset_dataset_test = Subset(self.test_data_set, indices_test)

        subset_train_data_loader = DataLoader(
            subset_dataset_train,
            batch_size=self.num_pruning_samples,
            shuffle=False,
        )
        subset_test_data_loader = DataLoader(
            subset_dataset_test,
            batch_size=self.test_batch_size,
            shuffle=False,  # Keep shuffle=False for deterministic ordering
        )
        return subset_train_data_loader, subset_test_data_loader
    '''

    def get_subset_dataloaders(self):
        indices_train, indices_test = self._get_selected_indices()
        #print(f"***** GOT INDICES: {indices_train} ***** and ***** {indices_test}")

        # Use RemappedSubset instead of Subset
        subset_dataset_train = RemappedSubset(self.train_data_set, indices_train, self.selected_classes)
        subset_dataset_test = RemappedSubset(self.test_data_set, indices_test, self.selected_classes)
        #print(f"***** CREATED SUBSET DATASETS: {subset_dataset_train} ***** and ***** {subset_dataset_test}")

        subset_train_data_loader = DataLoader(
            subset_dataset_train,
            batch_size=self.train_batch_size,
            shuffle=False,
        )
        subset_test_data_loader = DataLoader(
            subset_dataset_test,
            batch_size=self.test_batch_size,
            shuffle=False,
        )
        #print(f"***** CREATED SUBSET DATALOADERS: {subset_train_data_loader} ***** and ***** {subset_test_data_loader}")

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
        print(f"Selected classes are: {self.selected_classes}")
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
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224),
            ]
        )

        test_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        self.train_data_set = datasets.Imagenette(data_path,
                                                  transform=train_transform if self.use_data_Augmentation else test_transform,
                                                  download=self.download, split="train", )
        self.test_data_set = datasets.Imagenette(data_path,
                                                 transform=test_transform, download=self.download, split="val")

        if self.use_imagenet_labels:
            imagenette_classes = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
            self.train_data_set._samples = [
                (img, imagenette_classes[label]) for img, label in self.train_data_set._samples
            ]
            self.test_data_set._samples = [
                (img, imagenette_classes[label]) for img, label in self.test_data_set._samples
            ]

    def _get_selected_indices(self):
        indices_train = [index for index, (_, label) in enumerate(self.train_data_set._samples) if
                         label in self.selected_classes]
        indices_test = [index for index, (_, label) in enumerate(self.test_data_set._samples) if
                        label in self.selected_classes]
        random.seed(42)
        random.shuffle(indices_train)
        random.shuffle(indices_test)
        return indices_train, indices_test


class Imagenet_Dataloader_Factory(DataLoaderFactory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize_datasets(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224),
            ]
        )

        test_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        # Load the full datasets first
        full_train_dataset = datasets.ImageFolder(
            "/sc/dhc-cold/dsets/imagenet2012/train",
            transform=train_transform if self.use_data_Augmentation else test_transform
        )
        full_test_dataset = datasets.ImageFolder(
            "/sc/dhc-cold/dsets/imagenet2012/val",
            transform=test_transform
        )

        # Apply subsampling if specified
        if self.subsample_size_per_class is not None or self.subsample_ratio is not None:
            train_indices = self._get_subsample_indices(full_train_dataset, is_train=True)
            test_indices = self._get_subsample_indices(full_test_dataset, is_train=False)

            self.train_data_set = Subset(full_train_dataset, train_indices)
            self.test_data_set = Subset(full_test_dataset, test_indices)

            # Store the full dataset for accessing targets later
            self._full_train_dataset = full_train_dataset
            self._full_test_dataset = full_test_dataset
            self._train_subsample_indices = train_indices
            self._test_subsample_indices = test_indices
        else:
            self.train_data_set = full_train_dataset
            self.test_data_set = full_test_dataset
            self._full_train_dataset = full_train_dataset
            self._full_test_dataset = full_test_dataset
            self._train_subsample_indices = None
            self._test_subsample_indices = None

    def _get_subsample_indices(self, dataset, is_train=True):
        """
        Get indices for subsampling the dataset.

        Args:
            dataset: The full ImageFolder dataset
            is_train: Whether this is the training set (for logging)

        Returns:
            List of indices to keep
        """
        # Group indices by class
        class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.samples):
            class_to_indices[label].append(idx)

        selected_indices = []
        random.seed(42)  # For reproducibility

        for class_label, indices in class_to_indices.items():
            # Shuffle indices for this class
            class_indices = indices.copy()
            random.shuffle(class_indices)

            # Determine how many samples to keep
            if self.subsample_size_per_class is not None:
                # Use fixed number per class
                n_samples = min(self.subsample_size_per_class, len(class_indices))
            elif self.subsample_ratio is not None:
                # Use ratio of original size
                n_samples = max(1, int(len(class_indices) * self.subsample_ratio))
            else:
                n_samples = len(class_indices)

            # Select the samples
            selected_indices.extend(class_indices[:n_samples])

        # Shuffle all selected indices
        random.shuffle(selected_indices)

        return selected_indices

    def _get_selected_indices(self):
        """
        Get indices for the selected classes (used for pruning experiments).
        This should work whether or not subsampling is enabled.
        """
        if self._train_subsample_indices is not None:
            # We're working with a subsampled dataset
            # Need to get targets from the full dataset using subsample indices
            train_labels = [self._full_train_dataset.targets[i] for i in self._train_subsample_indices]
            test_labels = [self._full_test_dataset.targets[i] for i in self._test_subsample_indices]

            # Find indices within the subsampled dataset that match selected_classes
            indices_train = [
                i for i, label in enumerate(train_labels)
                if label in self.selected_classes
            ]
            indices_test = [
                i for i, label in enumerate(test_labels)
                if label in self.selected_classes
            ]
        else:
            # Working with full dataset
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
        #print(f"train indices: {indices_train}")
        #print(f"test indices: {indices_test}")

        random.seed(42)
        random.shuffle(indices_train)
        random.shuffle(indices_test)

        print(f"Selected indices for pruning:")
        print(f"  Train: {len(indices_train)} samples from selected classes {self.selected_classes}")
        print(f"  Test: {len(indices_test)} samples from selected classes {self.selected_classes}")

        return indices_train, indices_test


class GTSRB_DataLoaderFactory(DataLoaderFactory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize_datasets(self):
        mean = [0.3403, 0.3121, 0.3214]
        std = [0.2724, 0.2608, 0.2669]
        data_path = "./data/gtsrb"

        train_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
            ]
        )

        test_transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        self.train_data_set = datasets.GTSRB(
            data_path,
            transform=train_transform if self.use_data_Augmentation else test_transform,
            download=self.download,
            split="train",
        )
        self.test_data_set = datasets.GTSRB(
            data_path, transform=test_transform, download=self.download, split="test"
        )

    def _get_selected_indices(self):
        indices_train = [
            i
            for i, sample in enumerate(self.train_data_set._samples)
            if sample[1] in self.selected_classes
        ]
        indices_test = [
            i
            for i, sample in enumerate(self.test_data_set._samples)
            if sample[1] in self.selected_classes
        ]
        random.seed(42)
        random.shuffle(indices_train)
        random.shuffle(indices_test)
        return indices_train, indices_test


dataloaderFactories = {
    "cifar10": CIFAR10_DataLoaderFactory,
    "imagenette": Imagenette_DataLoaderFactory,
    "imagenet": Imagenet_Dataloader_Factory,
    "gtsrb": GTSRB_DataLoaderFactory,
}