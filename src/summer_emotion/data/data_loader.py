import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def get_data_loader(data_dir="data", batch_size=64, val_split=0.2) -> tuple:
    """
    Loading dataset, splitting it for train/val/test sets, transforms it and return DataLoader
    """
    # Transform 
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),    # Data conversion to gray scale
        transforms.Resize((48,48)),                     # Resizing to 48x48
        transforms.ToTensor()                           # Transforming into tensor with values 0-1
    ])

    # Paths
    train_path = f"{data_dir}/train"
    test_path = f"{data_dir}/test"

    # Loading datasets 
    train_val_dataset = datasets.ImageFolder(train_path, transform=transform)
    test_dataset = datasets.ImageFolder(test_path, transform=transform)

    # Splitting dataset into training and validation set
    train_size = int((1.0 - val_split) * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size

    # Seed blocking -> with every new execution of the script, network will be splitted the same as before
    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)

    # Data Loaders
    # shuffle=True for training, so the network will not learn the order of images
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_val_dataset.classes