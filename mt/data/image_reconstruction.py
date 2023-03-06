# Copyright 2019 Ondrej Skopek.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Tuple, Any
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from scipy.io import mmread

from .vae_dataset import VaeDataset
from ..mvae.distributions import EuclideanUniform


class ToDefaultTensor(transforms.Lambda):

    def __init__(self) -> None:
        super().__init__(lambda x: x.to(torch.get_default_dtype()))


def flatten_transform(img: torch.Tensor) -> torch.Tensor:
    return img.view(-1)


class ImageDynamicBinarization:

    def __init__(self, train: bool, invert: bool = False) -> None:
        self.uniform = EuclideanUniform(0, 1)
        self.train = train
        self.invert = invert

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape((-1,))  # Reshape per element, not batched yet.
        if self.invert:
            x = 1 - x
        if self.train:
            x = x > self.uniform.sample(x.shape)  # dynamic binarization
        else:
            x = x > 0.5  # fixed binarization for eval
        x = x.to(torch.get_default_dtype())
        return x


class ImageNormalization:

    def __init__(self, train: bool) -> None:
        self.train = train

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape((-1,))  # Reshape per element, not batched yet.
        x = x.to(torch.get_default_dtype())
        if x.max().item() != 0:
            x = x / x.max()
        return x


class MnistVaeDataset(VaeDataset):

    def __init__(self, batch_size: int, data_folder: str) -> None:
        super().__init__(batch_size, img_dims=(-1, 1, 28, 28), in_dim=784)
        self.data_folder = data_folder

    def _get_dataset(self, train: bool, transform: Any) -> torch.utils.data.Dataset:
        return datasets.MNIST(self.data_folder, train=train, download=True, transform=transform)

    def _load_mnist(self, train: bool) -> DataLoader:
        transformation = transforms.Compose(
            [transforms.ToTensor(),
             ToDefaultTensor(),
             transforms.Lambda(ImageNormalization(train=train))])
             # transforms.Lambda(ImageDynamicBinarization(train=train))
        return DataLoader(dataset=self._get_dataset(train, transform=transformation),
                          batch_size=self.batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=train)

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = self._load_mnist(train=True)
        test_loader = self._load_mnist(train=False)
        return train_loader, test_loader

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(x_mb_, x_mb, reduction="none")


class OmniglotVaeDataset(VaeDataset):

    def __init__(self, batch_size: int, data_folder: str) -> None:
        super().__init__(batch_size, img_dims=(-1, 1, 28, 28), in_dim=784)
        self.data_folder = data_folder

    def _get_dataset(self, train: bool, transform: Any) -> torch.utils.data.Dataset:
        return datasets.Omniglot(self.data_folder, background=train, download=True, transform=transform)

    def _load_omniglot(self, train: bool) -> DataLoader:
        transformation = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            ToDefaultTensor(),
            transforms.Lambda(ImageNormalization(train=train))
            # transforms.Lambda(ImageDynamicBinarization(train=train, invert=True))
        ])
        return DataLoader(dataset=self._get_dataset(train, transform=transformation),
                          batch_size=self.batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=train)

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = self._load_omniglot(train=True)
        test_loader = self._load_omniglot(train=False)
        return train_loader, test_loader

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(x_mb_, x_mb, reduction="none")


class Cifar10VaeDataset(VaeDataset):

    def __init__(self, batch_size: int, data_folder: str) -> None:
        super().__init__(batch_size, img_dims=(-1, 3, 32, 32), in_dim=3072)
        self.data_folder = data_folder

    def _load_cifar(self, train: bool) -> DataLoader:
        transformation = transforms.Compose([
            transforms.ToTensor(),
            ToDefaultTensor(),
            transforms.Lambda(flatten_transform),
        ])
        return DataLoader(dataset=datasets.CIFAR10(self.data_folder,
                                                   train=train,
                                                   download=True,
                                                   transform=transformation),
                          batch_size=self.batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=train)

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = self._load_cifar(train=True)
        test_loader = self._load_cifar(train=False)
        return train_loader, test_loader

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(x_mb_, x_mb, reduction="none")


class Cifar100VaeDataset(VaeDataset):

    def __init__(self, batch_size: int, data_folder: str) -> None:
        super().__init__(batch_size, img_dims=(-1, 3, 32, 32), in_dim=3072)
        self.data_folder = data_folder

    def _load_cifar(self, train: bool) -> DataLoader:
        transformation = transforms.Compose([
            transforms.ToTensor(),
            ToDefaultTensor(),
            transforms.Lambda(flatten_transform),
        ])
        return DataLoader(dataset=datasets.CIFAR100(self.data_folder,
                                                   train=train,
                                                   download=True,
                                                   transform=transformation),
                          batch_size=self.batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=train)

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = self._load_cifar(train=True)
        test_loader = self._load_cifar(train=False)
        return train_loader, test_loader

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(x_mb_, x_mb, reduction="none")


class SCVaeDataset(VaeDataset):

    def __init__(self, batch_size: int, in_dim: int, file_path: str) -> None:
        super().__init__(batch_size, img_dims=None, in_dim=in_dim)
        self.file_path = file_path

    def _load_gene(self, train: bool) -> DataLoader:
        transformation = transforms.Compose([
            transforms.ToTensor(),
            ToDefaultTensor(),
            transforms.Lambda(ImageNormalization(train=train)),
        ])
        gene_dataset = GeneExpressionDataset(csv_file=self.file_path, train=train, transform=transformation)
        return DataLoader(dataset=gene_dataset,
                          batch_size=self.batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=train)

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = self._load_gene(train=True)
        test_loader = self._load_gene(train=False)
        return train_loader, test_loader

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(x_mb_, x_mb, reduction="none")


class GeneExpressionDataset(Dataset):
    """Customized gene expression dataset."""

    def __init__(self, csv_file, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            train (bool): Whether to load train data or test data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.gene_features = pd.read_csv(csv_file)
        self.transform = transform
        self.train = train
        if self.train:
            self.offset = 0
        else:
            self.offset = 0
            # self.offset = int((self.gene_features.shape[0]) * 0.8)

    def __len__(self):
        if self.train:
            # training use the first 80% datapoints
            # return int((self.gene_features.shape[0]) * 0.8)
            return self.gene_features.shape[0]
        else:
            # return self.gene_features.shape[0] - int((self.gene_features.shape[0]) * 0.8)
            return self.gene_features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        gene_count_feature = np.array(self.gene_features.iloc[idx + self.offset, 1:-1])
        gene_count_feature = gene_count_feature.astype('float').reshape(-1, 1)
        label = int(self.gene_features.iloc[idx + self.offset, -1])

        if self.transform is not None:
            gene_count_feature = self.transform(gene_count_feature)

        return gene_count_feature, label


class SCPhereVaeDataset(VaeDataset):

    def __init__(self, batch_size: int, in_dim: int, mtx_path: str, label_path: str) -> None:
        super().__init__(batch_size, img_dims=None, in_dim=in_dim)
        self.mtx_path = mtx_path
        self.label_path = label_path

    def _load_cell(self, train: bool) -> DataLoader:
        transformation = transforms.Compose([
            transforms.ToTensor(),
            ToDefaultTensor(),
            transforms.Lambda(ImageNormalization(train=train)),
        ])
        sc_dataset = SCExpressionDataset(mtx_file=self.mtx_path, label_file=self.label_path, train=train, transform=transformation)
        return DataLoader(dataset=sc_dataset,
                          batch_size=self.batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=train)

    def create_loaders(self) -> Tuple[DataLoader, DataLoader]:
        train_loader = self._load_cell(train=True)
        test_loader = self._load_cell(train=False)
        return train_loader, test_loader

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(x_mb_, x_mb, reduction="none")
    

class SCExpressionDataset(Dataset):
    """Customized gene expression dataset."""

    def __init__(self, mtx_file, label_file, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            train (bool): Whether to load train data or test data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # read the data file
        self.cell_features = mmread(mtx_file).A.T
        # read the label information
        with open(label_file, 'r') as fr:
            label_lines = fr.read().splitlines()
        assert len(label_lines) == self.cell_features.shape[0], f"data size {self.cell_features.shape[0]} and label size {len(label_lines)} does not match!"
        
        label_set = []
        self.cell_labels = []
        for line in label_lines:
            if line not in label_set:
                label_set.append(line)
            self.cell_labels.append(label_set.index(line))
        self.cell_labels = np.array(self.cell_labels)
        
        self.transform = transform
        self.train = train
        # since we just want the embedding here, we would set train and test 
        # to be on the same dataset
        if self.train:
            self.offset = 0
        else:
            self.offset = 0
            # self.offset = int((self.gene_features.shape[0]) * 0.8)

    def __len__(self):
        if self.train:
            # training use the first 80% datapoints
            # return int((self.gene_features.shape[0]) * 0.8)
            return self.cell_features.shape[0]
        else:
            # return self.gene_features.shape[0] - int((self.gene_features.shape[0]) * 0.8)
            return self.cell_features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        gene_count_feature = np.array(self.cell_features[idx + self.offset, :]).astype('float').reshape(-1, 1)
        label = self.cell_labels[idx + self.offset]

        if self.transform is not None:
            gene_count_feature = self.transform(gene_count_feature)

        return gene_count_feature, label