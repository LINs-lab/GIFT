import os
import random
import data.utils.serialize
import numpy as np
import lmdb
import time
import pickle
import cv2
from PIL import Image
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Ignore warnings caused by the specific Torch version
import warnings

warnings.filterwarnings("ignore")


class ImageFolder(torchvision.datasets.ImageFolder):

    def __init__(self, classes, ipc, mem=False, shuffle=False, **kwargs):
        """
        Custom ImageFolder class with additional features.

        Parameters:
        - classes (list): List of class labels or folder names.
        - ipc (int): Number of images to sample per class.
        - mem (bool, optional): If True, load images into memory. Default is False.
        - shuffle (bool, optional): If True, shuffle images within each class. Default is False.
        - **kwargs: Additional arguments to pass to the base class constructor (torchvision.datasets.ImageFolder).

        Attributes:
        - mem (bool): Flag indicating whether to load images into memory.
        - image_paths (list): List of file paths for all sampled images.
        - samples (list): List of loaded image samples (if mem=True).
        - targets (list): List of target labels for each image.

        Note: Inherits from torchvision.datasets.ImageFolder.

        """
        super(ImageFolder, self).__init__(**kwargs)
        self.mem = mem
        self.image_paths = []  # List to store file paths for all sampled images
        self.samples = []  # List to store loaded image samples (if mem=True)
        self.targets = []  # List to store target labels for each image

        # Iterate through each class
        for c in range(len(classes)):
            dir_path = os.path.join(self.root, str(classes[c]).zfill(5))
            file_ls = os.listdir(dir_path)

            # Shuffle the file list if specified
            if shuffle:
                random.shuffle(file_ls)

            if ipc == -1:
                num_samples = len(file_ls)
            else:
                num_samples = ipc

            # Sample ipc images from the class
            for i in range(num_samples):
                if i >= len(file_ls):
                    index = i - len(file_ls) * (i // len(file_ls))
                else:
                    index = i

                # Construct the full path to the image
                self.image_paths.append(os.path.join(dir_path, file_ls[index]))

                # Load the image into memory if specified
                if self.mem:
                    self.samples.append(
                        self.loader(os.path.join(dir_path, file_ls[index]))
                    )

                # Record the target label
                self.targets.append(c)



    def __getitem__(self, index):
        """
        Custom implementation of __getitem__ method.

        Parameters:
        - index (int): Index of the desired item.

        Returns:
        - sample (torch.Tensor): Transformed image sample.
        - target (int): Target label for the image.
        """
        # Load the image from memory or file based on the mem attribute
        if self.mem:
            sample = self.samples[index]
        else:
            sample = self.loader(self.image_paths[index])

        # Apply transformations to the image sample
        sample = self.transform(sample)
        return sample, self.targets[index]

    def __len__(self):
        """
        Custom implementation of __len__ method.

        Returns:
        - int: Number of items in the dataset.
        """
        return len(self.targets)


class LMDBPTClass(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, is_image=True):
        """
        Custom Dataset class for reading data from an LMDB database.

        Parameters:
        - root (str): Root directory of the LMDB database.
        - transform (callable, optional): A function/transform to apply to the loaded data. Default is None.
        - target_transform (callable, optional): A function/transform to apply to the target (label). Default is None.
        - is_image (bool, optional): If True, assumes data is image-like. Default is True.

        Attributes:
        - root (str): Root directory of the LMDB database.
        - transform (callable): Transformation to apply to the loaded data.
        - target_transform (callable): Transformation to apply to the target (label).
        - is_image (bool): Flag indicating whether the data is image-like.
        - env (lmdb.Environment): LMDB environment for efficient data retrieval.
        - length (int): Number of samples in the dataset.
        - keys (list): List of keys representing samples in the LMDB database.

        Note: Inherits from torch.utils.data.Dataset.
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.is_image = is_image

        # Initialize placeholder for env and length.
        self.env = None
        self.length = self._get_tmp_length()
        self.keys = []

    def _open_lmdb(self):
        """
        Open the LMDB environment.

        Returns:
        - lmdb.Environment: LMDB environment object.
        """
        return lmdb.open(
            self.root,
            subdir=os.path.isdir(self.root),
            readonly=True,
            lock=False,
            readahead=False,
            max_readers=1,
            meminit=False,
        )

    def _get_tmp_length(self):
        """
        Get the temporary length of the dataset (used during initialization).

        Returns:
        - int: Temporary length of the dataset.
        """
        env = self._open_lmdb()
        with env.begin(write=False) as txn:
            length = txn.stat()["entries"]
            if txn.get(b"__keys__") is not None:
                length -= 1
        # Clean up.
        del env
        return length

    def _get_length(self):
        """
        Get the actual length of the dataset.

        Updates the 'length' attribute.
        """
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
            if txn.get(b"__keys__") is not None:
                self.length -= 1

    def _prepare_cache(self):
        """
        Prepare a cache file with keys for efficient data retrieval.
        """
        cache_file = self.root + "_cache_"
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor() if key != b"__keys__"]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def _decode_from_image(self, x):
        """
        Decode image data from binary format.

        Parameters:
        - x (bytes): Binary image data.

        Returns:
        - PIL.Image.Image: Decoded image.
        """
        image = cv2.imdecode(x, cv2.IMREAD_COLOR).astype("uint8")
        return Image.fromarray(image, "RGB")

    def _decode_from_array(self, x):
        """
        Decode array data from binary format.

        Parameters:
        - x (bytes): Binary array data.

        Returns:
        - PIL.Image.Image: Decoded array as an image.
        """
        return Image.fromarray(x.reshape(3, 32, 32).transpose((1, 2, 0)), "RGB")

    def __getitem__(self, index, apply_transform=True):
        """
        Get the item at the specified index.

        Parameters:
        - index (int): Index of the item to retrieve.
        - apply_transform (bool, optional): If True, apply data and target transformations. Default is True.

        Returns:
        - tuple: A tuple containing the loaded data and its target (label).
        """
        if self.env is None:
            # Open LMDB environment.
            self.env = self._open_lmdb()
            # Prepare cache file.
            self._prepare_cache()

        # Setup.
        env = self.env
        with env.begin(write=False) as txn:
            bin_file = txn.get(self.keys[index])

        image, target = serialize.loads(bin_file)

        if apply_transform:
            if self.is_image:
                image = self._decode_from_image(image)
            else:
                image = self._decode_from_array(image)

            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return image, target

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
        - int: Number of samples in the dataset.
        """
        return self.length

    def __repr__(self):
        """
        Get a string representation of the dataset.

        Returns:
        - str: String representation of the dataset.
        """
        return f"{self.__class__.__name__} ({self.root})"


class LMDBPT(torch.utils.data.Dataset):
    """A class to load an LMDB file for extremely large datasets.

    Args:
        root (str): Either the root directory for the database files or an absolute path pointing to the file.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        is_image (bool, optional): If True, assumes data is image-like. Default is True.
    """

    def __init__(self, root, transform=None, target_transform=None, is_image=True):
        """
        Initialize LMDBPT dataset.

        Parameters:
        - root (str): Either the root directory for the database files or an absolute path pointing to the file.
        - transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        - target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        - is_image (bool, optional): If True, assumes data is image-like. Default is True.
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.lmdb_files = self._get_valid_lmdb_files()

        # For each class, create an LMDBPTClass dataset
        self.dbs = []
        for lmdb_file in self.lmdb_files:
            self.dbs.append(
                LMDBPTClass(
                    root=lmdb_file,
                    transform=transform,
                    target_transform=target_transform,
                    is_image=is_image,
                )
            )

        # Build up indices
        self.indices = np.cumsum([len(db) for db in self.dbs])
        self.length = self.indices[-1]
        self._build_indices()
        self._prepare_target()

    def _get_valid_lmdb_files(self):
        """Get valid LMDB files based on the given root."""
        if not self.root.endswith(".lmdb"):
            files = []
            for l in os.listdir(self.root):
                if "_" in l and "-lock" not in l:
                    files.append(os.path.join(self.root, l))
            return files
        else:
            return [self.root]

    def _build_indices(self):
        self.from_to_indices = enumerate(zip(self.indices[:-1], self.indices[1:]))

    def _get_matched_index(self, index):
        if len(list(self.from_to_indices)) == 0:
            return 0, index

        for ind, (from_index, to_index) in self.from_to_indices:
            if from_index <= index and index < to_index:
                return ind, index - from_index

    def __getitem__(self, index, apply_transform=True):
        """
        Get the item at the specified index.

        Parameters:
        - index (int): Index of the item to retrieve.
        - apply_transform (bool, optional): If True, apply data and target transformations. Default is True.

        Returns:
        - tuple: A tuple containing the loaded data and its target (label).
        """
        block_index, item_index = self._get_matched_index(index)
        image, target = self.dbs[block_index].__getitem__(item_index, apply_transform)
        return image, target

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
        - int: Number of samples in the dataset.
        """
        return self.length

    def __repr__(self):
        """
        Get a string representation of the dataset.

        Returns:
        - str: String representation of the dataset.
        """
        fmt_str = f"Dataset {self.__class__.__name__}\n"
        fmt_str += f"    Number of datapoints: {self.__len__()}\n"
        fmt_str += f"    Root Location: {self.root}\n"
        tmp = "    Transforms (if any): "
        transform = self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        fmt_str += f"{tmp}{transform}\n"
        tmp = "    Target Transforms (if any): "
        target_transform = self.target_transform.__repr__().replace(
            "\n", "\n" + " " * len(tmp)
        )
        fmt_str += f"{tmp}{target_transform}"
        return fmt_str

    def _prepare_target(self):
        cache_file = self.root + "_targets_cache_"
        if os.path.isfile(cache_file):
            self.targets = pickle.load(open(cache_file, "rb"))
        else:
            self.targets = [
                self.__getitem__(idx, apply_transform=False)[1]
                for idx in range(self.length)
            ]
            pickle.dump(self.targets, open(cache_file, "wb"))


class SubsetLMDBPT(LMDBPT):
    """
    A subclass of LMDBPT representing a subset of the original LMDBPT dataset.

    Args:
        dataset (LMDBPT): The original LMDBPT dataset.
        indices (list): List of indices to include in the subset.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.length = len(indices)

    def __getitem__(self, index, apply_transform=True):
        # Redirect to the original dataset using the subset index
        subset_index = self.indices[index]
        return self.dataset[subset_index]

    def __len__(self):
        return self.length


def load_subset_lmdb_dataset(dataset, ipc, classes, targets):
    """
    Get a subset LMDB dataset based on the given ipc and classes.

    Args:
        dataset (LMDBPT): The original LMDB dataset.
        ipc (int): Number of data points to extract per class.
        classes (list): List of class labels to extract.

    Returns:
        LMDBPT: Subset LMDB dataset.
    """
    subset_indices = []

    # Collect indices for the specified classes with random sampling
    for class_label in classes:
        class_indices = [
            idx for idx, target in enumerate(targets) if target == class_label
        ]
        if ipc == -1:
            subset_indices.extend(class_indices)
        else:
            subset_indices.extend(random.sample(class_indices, ipc))

    # Create a subset dataset using the collected indices
    subset_dataset = SubsetLMDBPT(dataset, subset_indices)

    return subset_dataset



def load_dataset(
    dataset="imagenet-10",
    ipc=-1,
    classes=[],
    root="./data",
    train=True,
    transform=None,
    shuffle=True,
):
    """
    Get the specified dataset for training or validation.

    Parameters:
    - dataset (str): Name of the dataset, e.g., "imagenet-10" or "imagenet-1k-lmdb".
    - ipc (int): Index of the ImageNet class to use (only applicable for "imagenet-1k-lmdb").
    - classes (list): List of class names to include in the dataset.
    - root (str): Root directory where the dataset is located.
    - train (bool): If True, return the training set; otherwise, return the validation set.
    - transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.

    Returns:
    - dataset: The requested dataset, either as an ImageFolder or a custom LMDB dataset.

    Note:
    - For "imagenet-1k-lmdb", the LMDB dataset is loaded and a subset is extracted based on specified classes and ipc.
    - For other datasets, ImageFolder is used to load the dataset from the specified directory.
    """

    # Converts the dataset to lowercase.
    dataset = dataset.lower()

    # Construct the directory path based on the dataset and split (train or val)
    if train:
        dataset_dir = os.path.join(root, dataset, "train")
    else:
        dataset_dir = os.path.join(root, dataset, "val")

    if classes == []:
        classes = get_classes(dataset)

    if transform == None:
        resize = transforms.Compose([])
        if dataset != "tinyimagenet" and "imagenet" in dataset:
            resize = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224)]
            )
        transform = transforms.Compose(
            [resize, transforms.ToTensor(), load_normalize(dataset)]
        )

    # Handle the case of "imagenet-1k-lmdb" separately
    if dataset == "imagenet-1k-lmdb":
        dataset_dir = dataset_dir + ".lmdb"
        dataset = LMDBPT(root=dataset_dir, transform=transform)
        dataset = load_subset_lmdb_dataset(dataset, ipc, classes, dataset.targets)
    else:
        # For other datasets, use torchvision's ImageFolder
        dataset = ImageFolder(
            root=dataset_dir,
            classes=classes,
            ipc=ipc,
            mem=False,
            shuffle=shuffle,
            transform=transform,
        )

    nclass = len(classes)
    dataset.nclass = nclass

    return dataset


def get_classes(dataset="imagenet-1k"):

    # Converts the dataset to lowercase.
    dataset = dataset.lower()

    full_dataset_nclass = {
        "cifar10": 10,
        "cifar100": 100,
        "fashionmnist": 10,
        "imagenet-10": 10,
        "imagenet-100": 100,
        "imagenet-1k": 1000,
        "imagenet-1k-lmdb": 1000,
        "imagenet-fruits": 10,
        "imagenet-nette": 10,
        "imagenet-woof": 10,
        "imagenet-fruits": 10,
        "mnist": 10,
        "tinyimagenet": 200,
    }
    return list(range(full_dataset_nclass[dataset]))


def load_normalize(dataset="cifar10"):
    dataset = dataset.lower()
    data_stats = {
        "cifar": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
        "imagenet": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "cifar10": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
        "cifar100": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
        "svhn": {"mean": [0.4377, 0.4438, 0.4728], "std": [0.1980, 0.2010, 0.1970]},
        "mnist": {"mean": [0.1307], "std": [0.3081]},
        "fashion": {"mean": [0.2861], "std": [0.3530]},
    }

    if "imagenet" in dataset:
        dataset = "imagenet"

    return transforms.Normalize(
        mean=data_stats[dataset]["mean"], std=data_stats[dataset]["std"]
    )


def load_denormalize(dataset="cifar10"):
    dataset = dataset.lower()
    data_stats = {
        "cifar": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
        "imagenet": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "cifar10": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
        "cifar100": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
        "svhn": {"mean": [0.4377, 0.4438, 0.4728], "std": [0.1980, 0.2010, 0.1970]},
        "mnist": {"mean": [0.1307], "std": [0.3081]},
        "fashion": {"mean": [0.2861], "std": [0.3530]},
    }

    if "imagenet" in dataset:
        dataset = "imagenet"

    denormalize = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[
                    1 / data_stats[dataset]["std"][0],
                    1 / data_stats[dataset]["std"][1],
                    1 / data_stats[dataset]["std"][2],
                ],
            ),
            transforms.Normalize(
                mean=[
                    -data_stats[dataset]["mean"][0],
                    -data_stats[dataset]["mean"][1],
                    -data_stats[dataset]["mean"][2],
                ],
                std=[1.0, 1.0, 1.0],
            ),
        ]
    )

    return denormalize
