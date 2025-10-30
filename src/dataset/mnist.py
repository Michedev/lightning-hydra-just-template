import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms


class FlattenedMNIST(Dataset):
    def __init__(self, root: str = "./data", train: bool = True, download: bool = True):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.mnist = MNIST(
            root=root,
            train=train,
            download=download,
            transform=self.transform
        )
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        # Flatten the image from (1, 28, 28) to (784,)
        image = image.view(-1)
        return image, label
