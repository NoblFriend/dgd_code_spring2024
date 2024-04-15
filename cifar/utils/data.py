
import torch

import torchvision
import torchvision.transforms as transforms


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def get_train_dataloader():
    return torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                ),
            ])
        ),
        batch_size=128,
        shuffle=False,
        num_workers=0
    )



def get_test_dataloader():
    return torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                ),
            ])
        ),
        batch_size=128,
        shuffle=False,
        num_workers=0
    )


def get_train_data():
    return torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(
                size=32,
                padding=4
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ])
    )
