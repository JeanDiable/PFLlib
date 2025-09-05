import os
import random
import sys

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, save_file, separate_data, split_data

random.seed(1)
np.random.seed(1)
num_clients = 2
dir_path = "Cifar10/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    # Always regenerate when called explicitly; if you prefer caching, uncomment the check
    # if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
    #     return

    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=dir_path + "rawdata", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=dir_path + "rawdata", train=False, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False
    )

    # Load full tensors once
    train_images, train_labels = next(iter(trainloader))
    test_images, test_labels = next(iter(testloader))

    # Convert to numpy (N, C, H, W) and int64 labels
    train_images = train_images.cpu().detach().numpy()
    test_images = test_images.cpu().detach().numpy()
    train_labels = train_labels.cpu().detach().numpy().astype(np.int64)
    test_labels = test_labels.cpu().detach().numpy().astype(np.int64)

    dataset_image = np.concatenate([train_images, test_images], axis=0)
    dataset_label = np.concatenate([train_labels, test_labels], axis=0)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data(
        (dataset_image, dataset_label),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=5,
    )
    train_data, test_data = split_data(X, y)
    save_file(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        num_classes,
        statistic,
        niid,
        balance,
        partition,
    )


if __name__ == "__main__":
    niid = True if (len(sys.argv) > 1 and sys.argv[1] == "noniid") else False
    balance = True if (len(sys.argv) > 2 and sys.argv[2] == "balance") else False
    partition = sys.argv[3] if (len(sys.argv) > 3 and sys.argv[3] != "-") else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
