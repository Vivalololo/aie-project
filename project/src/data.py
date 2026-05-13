from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


IMG_SIZE = 224

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def load_datasets(data_dir="./data/dataset/chest_xray"):
    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
    val_ds = datasets.ImageFolder(f"{data_dir}/val", transform=eval_tf)
    test_ds = datasets.ImageFolder(f"{data_dir}/test", transform=eval_tf)
    return train_ds, val_ds, test_ds


def get_transforms():
    return train_tf, eval_tf


def make_loaders_model(
        ds_train,
        ds_val,
        ds_test,
        batch_size: int,
        seed: int = 42,
):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(ds_train))
    rng.shuffle(indices)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

    return train_loader, val_loader, test_loader