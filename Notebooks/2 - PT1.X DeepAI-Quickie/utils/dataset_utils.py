# Copyright 2022 Vittorio Mazzia. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from abc import ABC, abstractmethod

import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


class DatasetLoader(ABC):
    """Generic PyTorch Dataset Loader"""

    def __init__(self):
        self.train_set = None
        self.test_set = None

        self.train_transform = None
        self.test_transform = None

        self.download_path = None

    @staticmethod
    def get_loader(
        dataset: torchvision.datasets, batch_size: int, shuffle: bool = True
    ) -> DataLoader:
        """Provides a DataLoader of the given dataset

        Args:
            dataset (torchvision.datasets): input dataset
            batch_size (int): DataLoader batch size
            shuffle (bool, optional): DataLoader shuffle option. Defaults to True.

        Returns:
            DataLoader: output DataLoader
        """
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return train_loader

    @abstractmethod
    def download_dataset(self):
        raise NotImplementedError

    def get_train_loader(self, batch_size: int) -> DataLoader:
        return self.get_loader(self.train_set, batch_size, True)

    def get_test_loader(self, batch_size: int) -> DataLoader:
        return self.get_loader(self.test_set, batch_size, False)


class MNISTLoader(DatasetLoader):
    """MNIST PyTorch Dataset Loader"""

    def __init__(
        self,
        train_transform: torchvision.transforms,
        test_transform: torchvision.transforms,
        download_path: str = "./tmp",
    ):
        """Initialize MNIST Dataset Loader

        Args:
            train_transform (torchvision.transforms): transformations to be applied to training set
            test_transform (torchvision.transforms): transformations to be applied to test set
            download_path (str, optional): download path. Defaults to './tmp'.
        """
        super(MNISTLoader, self).__init__()

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.download_path = download_path

    def download_dataset(self) -> None:
        """Download dataset to the given path"""
        self.train_set = MNIST(
            self.download_path,
            train=True,
            download=True,
            transform=self.train_transform,
        )

        self.test_set = MNIST(
            self.download_path,
            train=False,
            download=True,
            transform=self.test_transform,
        )


class TrainingDatasetFF(torch.utils.data.Dataset):
    """Utility class to store positive and negative examples to train
    with FF algorithm.
    """

    def __init__(self, dataset_generator: DataLoader) -> None:
        """Initialize TrainingDatasetFF

        Args:
            dataset_generator (DataLoader): DataLoader to store
        """
        with torch.no_grad():
            self.dataset = [
                batch
                for X_pos, X_neg in dataset_generator
                for batch in zip(X_pos, X_neg)
            ]

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
