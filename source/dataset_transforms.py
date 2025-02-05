import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as Trans
from PIL import Image
import matplotlib.pyplot as plt
import random

mean = 0.5
std = 0.2

# Dataset class
class Dataset(Dataset):
    def __init__(self, image_path, label_path, transform=False):
        self.images = Image.open(image_path)
        self.labels = Image.open(label_path)
        self.transform = transform

    def __len__(self):
        return self.images.n_frames


    def __getitem__(self, idx):
        # Access specific frame
        self.images.seek(idx)
        self.labels.seek(idx)



        # Convert to grayscale
        image = self.images.convert("L")
        label = self.labels.convert("L")

        if self.transform:

          # Random horizontal flipping
          if random.random() > 0.5:
              image = Trans.hflip(image)
              label = Trans.hflip(label)

        # Random vertical flipping
          if random.random() > 0.5:
              image = Trans.vflip(image)
              label = Trans.vflip(label)

        # Transform to tensor
        image = Trans.to_tensor(image)
        label = Trans.to_tensor(label)

        if self.transform:
          # normalize the image
            image = Trans.normalize(image, mean, std)

        return image, label
    
# Data transforms
basic_transform = transforms.Compose([
    transforms.ToTensor()
])

basic_transform_resize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([572, 572])
])

advanced_transform_image = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[mean])
])

advanced_transform_label = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
])
def computeMeanStd(dataloader):
    mean = 0.0
    std = 0.0
    n_samples = 0

    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, -1)
        mean += images.mean(1).sum(0)
        std += images.std(1).sum(0)
        n_samples += batch_samples

    mean /= n_samples
    std /= n_samples

    return mean.item(), std.item()
