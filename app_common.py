import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os

from utils import calculate_filters
from inversemodel import InverseNet, InverseNetDenoise


CIFAR10_ROOT = os.path.join(os.path.dirname(__file__), 'data')


def seed_initial(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_cifar10_loaders(train_batch_size=None, test_batch_size=100, test_shuffle=False):
    transform = transforms.Compose([transforms.ToTensor()])

    train_loader = None
    if train_batch_size is not None:
        trainset = torchvision.datasets.CIFAR10(
            root=CIFAR10_ROOT,
            train=True,
            download=True,
            transform=transform,
        )
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=2,
        )

    testset = torchvision.datasets.CIFAR10(
        root=CIFAR10_ROOT,
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=test_shuffle,
        num_workers=2,
    )
    return train_loader, test_loader


def build_model(
    model_name,
    compression_ratio,
    channel,
    device,
    use_denoise=False,
    x_min=-3.0,
    x_max=3.0,
):
    if model_name == 'Inverse':
        num_filters, real_cr = calculate_filters(compression_ratio)
        inverse_cls = InverseNetDenoise if use_denoise else InverseNet
        model_tag = 'InverseNetDenoise' if use_denoise else 'InverseNet'
        print(
            f'==> Building model.. {model_tag} '
            f'\nThe real compression rate is {real_cr:.3f} || '
            f'The number of filters is {num_filters}'
        )
        model = inverse_cls(M=num_filters, channel=channel, x_min=x_min, x_max=x_max)
    else:
        raise ValueError(f'Unknown model: {model_name}')

    model = model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: {}M'.format(n_parameters / 1e6))
    return model
