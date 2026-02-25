from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from typing import List
import random
from collections import defaultdict
from torch.utils.data import random_split
import torch


def get_limited_samples_dataloader(dataloader, samples_per_class, num_classes):
    """
    Extract a subset of dataloader with limited samples per class.

    Args:
        dataloader (DataLoader): Original DataLoader
        samples_per_class (int): Number of samples to keep per class
        num_classes (int): Number of classes in the dataset

    Returns:
        New DataLoader with limited samples per class
    """
    # Dictionary to track samples per class
    class_counts = {i: 0 for i in range(num_classes)}
    selected_indices = []

    # Iterate through the dataset (not dataloader) to get individual samples
    dataset = dataloader.dataset

    for idx in range(len(dataset)):
        # Get the label for this sample
        _, label = dataset[idx]

        # If we haven't collected enough samples for this class yet
        if class_counts[label] < samples_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1

        # Check if we've collected enough samples for all classes
        if all(count >= samples_per_class for count in class_counts.values()):
            break

    # Create a new subset with selected indices
    limited_dataset = torch.utils.data.Subset(dataset, selected_indices)

    # Create new dataloader with the limited dataset
    limited_dataloader = DataLoader(
        limited_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
    )

    return limited_dataloader


class RemappedSubset(Dataset):
    def __init__(self, dataset, indices, original_classes):
        """
        Wrapper class that remaps labels to consecutive indices (e.g., [204, 042, 059] --> [0, 1, 2]).

        Args:
            dataset (Dataset): Dataset whose labels need to be remapped.
            indices (List[int]): Selected indices for remapping.
            original_classes (List[int]): List of original classes for tracking back remapped labels.
        """
        self.dataset = dataset
        self.indices = indices
        # Create mapping from original class indices to new consecutive indices
        self.label_map = {orig_class: new_idx for new_idx, orig_class in enumerate(original_classes)}
        print(f"Label remapping: {self.label_map}", flush=True)

    def __len__(self):
        """Returns the number of samples in the subset."""
        return len(self.indices)

    def __getitem__(self, idx):
        """Retrieves a sample from the subset and returns it with its remapped label."""
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
                 num_pruning_samples: int = None,
                 use_imagenet_labels: bool | None = False,
                 subsample_ratio: float = None,
                 subsample_size_per_class: int = None,
                 val_split: float = 0.1,
                 ):
        """
        Factory for creating DataLoader objects.

        Args:
            train_batch_size (int): Batch size of the train set.
            test_batch_size (int): Batch size of the test set.
            use_data_augmentation (boolean): True if we want to use data augmentation
            download (boolean): True if we want to download the necessary dataset files.
            train_shuffle (boolean): True if we want to shuffle the training set.
            selected_classes (List[int]): List of selected classes for remapped labels.
            num_pruning_samples (int): Samples for a limited dataloder (e.g., for OCAP).
            use_imagenet_labels (boolean): Use labels of ImageNet, mainly used for ImageNette remapping.
            subsample_ratio (float): Ratio of samples per class to use from original dataset.
            subsample_size_per_class (int): Absolute number of samples per class to use.
            val_split (float): Ratio of train set to be used for validation.
        """
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
        self.val_split = val_split
        print(f"SAMPLING: subsample_ratio={subsample_ratio}, subsample_size_per_class={subsample_size_per_class}")

        self._initialize_datasets()

    def _initialize_datasets(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_dataloaders(self):
        """Returns train, val and test split."""
        train_data_loader = DataLoader(
            self.train_data_set, batch_size=self.train_batch_size, shuffle=self.train_shuffle
        )
        val_data_loader = DataLoader(
            self.val_data_set, batch_size=self.test_batch_size, shuffle=False
        )
        test_data_loader = DataLoader(
            self.test_data_set, batch_size=self.test_batch_size, shuffle=False
        )
        return train_data_loader, val_data_loader, test_data_loader

    def _get_selected_indices(self):
        raise NotImplementedError("This method should be implemented by subclasses.")


    def get_subset_dataloaders(self):
        """Returns train, val and test split with subsets of the data (e.g., for ImageNet)."""
        indices_train, indices_val, indices_test = self._get_selected_indices()

        # Remap indices
        subset_dataset_train = RemappedSubset(self.train_data_set, indices_train, self.selected_classes)
        subset_dataset_val = RemappedSubset(self.val_data_set, indices_val, self.selected_classes)
        subset_dataset_test = RemappedSubset(self.test_data_set, indices_test, self.selected_classes)

        # Create dataloaders with subsets of the whole data
        subset_train_data_loader = DataLoader(
            subset_dataset_train,
            batch_size=self.train_batch_size,
            shuffle=False,
        )
        subset_val_data_loader = DataLoader(
            subset_dataset_val,
            batch_size=self.test_batch_size,
            shuffle=False,
        )
        subset_test_data_loader = DataLoader(
            subset_dataset_test,
            batch_size=self.test_batch_size,
            shuffle=False,
        )

        pruning_dataloader = None
        if self.num_pruning_samples is not None:
            pruning_dataloader = get_limited_samples_dataloader(
                subset_train_data_loader,
                samples_per_class=self.num_pruning_samples,
                num_classes=len(self.selected_classes)
            )

        return subset_train_data_loader, subset_val_data_loader, subset_test_data_loader, pruning_dataloader

    def _split_train_val(self, indices):
        """
        Shuffle indices and split into train/val according to val_split.

        Args:
            indices (List[int]): List of the indices to be split.

        Returns:
            Tuple(List[int], List[int], List[int]): Indices for train and val, respectively.
        """
        random.seed(42)
        random.shuffle(indices)
        val_size = int(len(indices) * self.val_split)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        return train_indices, val_indices


class Imagenet_Dataloader_Factory(DataLoaderFactory):
    def __init__(self, **kwargs):
        """Factory for creating ImageNet dataloaders."""
        super().__init__(**kwargs)

    def _initialize_datasets(self):
        """
        Initializes dataset by applying the necessary transformations and subsets.
        """
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

        # Always store full datasets
        self._full_train_dataset = full_train_dataset
        self._full_test_dataset = full_test_dataset

        # Determine train/test indices
        if self.subsample_size_per_class is not None or self.subsample_ratio is not None:
            train_indices = self._get_subsample_indices(full_train_dataset)
            test_indices = self._get_subsample_indices(full_test_dataset)
        else:
            train_indices = list(range(len(full_train_dataset)))
            test_indices = None

        # Split train into train/val
        train_indices, val_indices = self._split_train_val(train_indices)

        # Create dataset splits
        self.train_data_set = Subset(full_train_dataset, train_indices)
        self.val_data_set = Subset(full_train_dataset, val_indices)
        self.test_data_set = (
            Subset(full_test_dataset, test_indices)
            if test_indices is not None
            else full_test_dataset
        )

        # Store indices for later access
        self._train_subsample_indices = train_indices
        self._val_subsample_indices = val_indices
        self._test_subsample_indices = test_indices

    def _get_subsample_indices(self, dataset):
        """
        Get indices for subsampling the dataset.

        Args:
            dataset (Dataset): The full dataset.

        Returns:
            List of subsampled indices to keep.
        """
        # Group indices by class
        class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.samples):
            class_to_indices[label].append(idx)

        selected_indices = []
        random.seed(42)

        for class_label, indices in class_to_indices.items():
            # Shuffle indices for this class
            class_indices = indices.copy()
            random.shuffle(class_indices)

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
        Get indices for the selected classes.
        Works for full dataset and subsampled dataset.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                Indices for train, val and test split.
        """

        selected = set(self.selected_classes)

        def extract_labels(dataset, subset_indices=None):
            """
            Extract labels from a dataset.
            If subset_indices is provided, labels are taken from the full dataset
            using those indices (for subsampled case).
            """
            if subset_indices is None:
                return dataset.targets
            return [dataset.targets[i] for i in subset_indices]

        def filter_indices(labels):
            """Return positions whose label is in selected classes."""
            return [i for i, label in enumerate(labels) if label in selected]

        # ---- TRAIN / VAL LABEL SOURCES ----
        if self._train_subsample_indices is not None:
            train_labels = extract_labels(
                self._full_train_dataset,
                self._train_subsample_indices,
            )
            val_labels = extract_labels(
                self._full_train_dataset,
                self._val_subsample_indices,
            )
        else:
            train_labels = extract_labels(self.train_data_set)
            val_labels = extract_labels(self.val_data_set)

        # ---- TEST LABEL SOURCE ----
        if self._test_subsample_indices is not None:
            test_labels = extract_labels(
                self._full_test_dataset,
                self._test_subsample_indices,
            )
        else:
            test_labels = extract_labels(self.test_data_set)

        # ---- FILTER ----
        indices_train = filter_indices(train_labels)
        indices_val = filter_indices(val_labels)
        indices_test = filter_indices(test_labels)

        # ---- SHUFFLE (deterministic) ----
        rng = random.Random(42)
        rng.shuffle(indices_train)
        rng.shuffle(indices_val)
        rng.shuffle(indices_test)

        print("Selected indices for pruning:")
        print(f"  Train: {len(indices_train)} samples from selected classes {self.selected_classes}")
        print(f"  Val: {len(indices_val)} samples from selected classes {self.selected_classes}")
        print(f"  Test: {len(indices_test)} samples from selected classes {self.selected_classes}")

        return indices_train, indices_val, indices_test


class CIFAR10_DataLoaderFactory(DataLoaderFactory):
    def __init__(self, **kwargs):
        """Factory for creating CIFAR10 dataloaders."""
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
        """Factory for creating ImageNette dataloaders."""
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



class GTSRB_DataLoaderFactory(DataLoaderFactory):
    def __init__(self, **kwargs):
        """Factory for creating GTSRB dataloaders."""
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
    "imagenet": Imagenet_Dataloader_Factory,
    "cifar10": CIFAR10_DataLoaderFactory,
    "imagenette": Imagenette_DataLoaderFactory,
    "gtsrb": GTSRB_DataLoaderFactory,
}
