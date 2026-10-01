from abc import ABC, abstractmethod

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class DatasetStrategy(ABC):
    @abstractmethod
    def get_datasets(self, data_dir: str) -> tuple[Dataset, Dataset]: pass

class FashionMNISTStrategy(DatasetStrategy):
    name = "Fashion-MNIST"

    classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    mean = (0.2860,)
    std = (0.3530,)

    in_features = 28 * 28 * 1
    in_channels = 1

    cnn_flatten_dim = 128 * 3 * 3
    num_classes = 10

    def get_datasets(self, data_dir: str) -> tuple[Dataset, Dataset]:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]
        )

        train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=test_transform, download=True)
        return train_dataset, test_dataset

class CIFAR10Strategy(DatasetStrategy):
    name = "CIFAR-10"

    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    in_features = 32 * 32 * 3
    in_channels = 3

    cnn_flatten_dim = 128 * 4 * 4
    num_classes = 10

    def get_datasets(self, data_dir: str) -> tuple[Dataset, Dataset]:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]
        )

        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)
        return train_dataset, test_dataset