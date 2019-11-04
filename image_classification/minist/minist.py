# zhangshulin
# zhangslwork@yeah.net
# 2019-10-11


import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


def build_minist_transforms():
    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomResizedCrop((28, 28), scale=(0.9, 1.1),
                                     ratio=(0.9, 1.1)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img / 255)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img / 255)
    ])

    return train_transform, test_transform


def build_minist_dataset(path):
    train_transform, test_transform = build_minist_transforms()

    trainset = datasets.MNIST(path, True, train_transform, download=True)
    testset = datasets.MNIST(path, False, test_transform, download=True)

    return trainset, testset


def build_minist_loader(path, batch, workers=2):
    trainset, testset = build_minist_dataset(path)

    trainloader = DataLoader(trainset, batch_size=batch, shuffle=True,
                             num_workers=workers)
    testloader = DataLoader(testset, batch_size=batch, shuffle=False,
                            num_workers=workers)

    return trainloader, testloader
