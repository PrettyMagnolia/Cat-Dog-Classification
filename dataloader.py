from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CatDogDataset


def load_data(data_dir, batch_size, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_data = CatDogDataset(root=data_dir + '/train', transform=transform)
    val_data = CatDogDataset(root=data_dir + '/val', transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
