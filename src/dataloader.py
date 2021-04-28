import cutout
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as datasets

"""All necessary data loaders."""

def transform_sample():
    return transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) )])

def train_loader_noncropped(path, batch_size, pin_memory=False, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize((256, 256)),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                                 cutout.Cutout(16)
                             ])),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=2)

def test_loader_noncropped(path, batch_size, pin_memory=False, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize((256, 256)),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=2)

def train_loader_cropped(path, batch_size):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize((224, 224)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                                 cutout.Cutout(16)
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)

def test_loader_cropped(path, batch_size):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2)

def train_loader_sample(path, sample_size):
    
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Resize((224, 224)),
                                 transforms.ToTensor()
                             ])),
        batch_size=sample_size,
        shuffle=True,
        num_workers=2)