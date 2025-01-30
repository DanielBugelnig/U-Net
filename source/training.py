import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as Trans
from PIL import Image
import matplotlib.pyplot as plt

from architecture import UNet

# Clearing CUDA memory
torch.cuda.empty_cache()

## colab
# from google.colab import drive
# drive.mount('/content/drive')


# Dataset class
class Dataset(Dataset):
    def __init__(self, image_path, label_path, image_transform=None, label_transform=None):
        self.images = Image.open(image_path)
        self.labels = Image.open(label_path)
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __len__(self):
        return self.images.n_frames

    def __getitem__(self, idx):
        # Access specific frame
        self.images.seek(idx)
        self.labels.seek(idx)

        # Convert to grayscale
        image = self.images.convert("L")
        label = self.labels.convert("L")

        # Apply transformations
        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        return image, label



mean = 0.5
std = 0.2


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

def train(model, dataloader, criterion, optimizer, nrOfEpochs):
    model.train()
    for epoch in range(nrOfEpochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            labels = Trans.center_crop(labels, [388, 388])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{nrOfEpochs}, Loss: {train_loss:.4f}")

def test(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0

    total_TP = total_TN = total_FP = total_FN = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            outputs_bin = (outputs > 0.5).int()
            labels = Trans.center_crop(labels, [388, 388])
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            total_TP += ((outputs_bin == 1) & (labels == 1)).sum().item()
            total_TN += ((outputs_bin == 0) & (labels == 0)).sum().item()
            total_FP += ((outputs_bin == 1) & (labels == 0)).sum().item()
            total_FN += ((outputs_bin == 0) & (labels == 1)).sum().item()

    test_loss = running_loss / len(dataloader)
    accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
    precision = total_TP / (total_TP + total_FP) 
    recall = total_TP / (total_TP + total_FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":


    # for google colab, vanilla dataset ISBI 2012
    train_image_path = "/content/drive/My Drive/Machine_Learning/ISBI-2012-challenge/train-volume.tif"
    train_label_path = "/content/drive/My Drive/Machine_Learning/ISBI-2012-challenge/train-labels.tif"
    test_image_path = "/content/drive/My Drive/Machine_Learning/ISBI-2012-challenge/test-volume.tif"
    test_label_path = "/content/drive/My Drive/Machine_Learning/ISBI-2012-challenge/test-labels.tif"

    # for google colab, mirrored dataset ISBI 2012
    train_image_path = "/content/drive/My Drive/Machine_Learning/ISBI-2012-challenge-mirrored/train-mirror.tif"
    train_label_path = "/content/drive/My Drive/Machine_Learning/ISBI-2012-challenge-mirrored/train-labels.tif"
    test_image_path = "/content/drive/My Drive/Machine_Learning/ISBI-2012-challenge-mirrored/test-mirror.tif"
    test_label_path = "/content/drive/My Drive/Machine_Learning/ISBI-2012-challenge-mirrored/test-labels.tif"


    # for local, vanilla dataset ISBI 2012
    train_image_path = "../ISBI-2012-challenge/train-volume.tif"
    train_label_path = "../ISBI-2012-challenge/train-labels.tif"
    test_image_path = "../ISBI-2012-challenge/test-volume.tif"
    test_label_path = "/../ISBI-2012-challenge/test-labels.tif"

    # for local, mirrored dataset ISBI 2012
    train_image_path = "../ISBI-2012-mirrored/train-mirror.tif"
    train_label_path = "../ISBI-2012-mirrored/train-labels.tif"
    test_image_path = "../ISBI-2012-mirrored/test-mirror.tif"
    test_label_path = "../ISBI-2012-mirrored/test-labels.tif"



    # Initial transform for computing mean and std
    computing_mean_std_dataset = Dataset(train_image_path, train_label_path, image_transform=basic_transform, label_transform=basic_transform)
    computing_mean_std_dataset_loader = DataLoader(computing_mean_std_dataset, batch_size=1, shuffle=True)


    mean, std = computeMeanStd(computing_mean_std_dataset_loader)

    print(f"Mean: {mean}, Std: {std}")


    # Datasets and loaders
    # Vanilla Dataset: use basic_transform_resize
    # Mirrored Dataset: use advanced_transform_image and advanced_transform_label or : basic_transform for both
    train_dataset = Dataset(train_image_path, train_label_path, image_transform=basic_transform_resize, label_transform=basic_transform_resize)
    test_dataset = Dataset(test_image_path, test_label_path, image_transform=basic_transform_resize, label_transform=basic_transform_resize)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(1, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Training
    print("Starting training")
    train(model, train_loader, criterion, optimizer, nrOfEpochs=10)

    # Testing
    print("Starting evaluation")
    test(model, test_loader, criterion)

    torch.save(model.state_dict(), "results/model.pth")
    #torch.save(model.state_dict(), "/content/drive/My Drive/Machine_Learning/results/model1.pth")
    print("Model saved")