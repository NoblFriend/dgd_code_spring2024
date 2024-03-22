
import torch

import torchvision
import torchvision.transforms as transforms


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


trainloader = torch.utils.data.DataLoader(
    dataset=torchvision.datasets.CIFAR10(
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
    ),
    batch_size=128,
    shuffle=False,
    num_workers=2
)

testloader = torch.utils.data.DataLoader(
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
    num_workers=2
)