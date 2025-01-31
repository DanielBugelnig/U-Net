import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as Trans
from PIL import Image
import matplotlib.pyplot as plt
from architecture import UNet

torch.cuda.empty_cache()
mean=0
std=0

#Loading data
#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
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

def test(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0

    # Initialize accumulators for metrics
    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0

    with torch.no_grad():
        for images, labels in dataloader:
            # Move data to the device
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            
            #binary threshold at 0.5
            outputs_bin = (outputs > 0.5).int()
            
            labels = Trans.center_crop(labels, [388, 388]) 
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Calculate metrics
            total_TP += ((outputs_bin == 1) & (labels == 1)).sum().item()
            total_TN += ((outputs_bin == 0) & (labels == 0)).sum().item()
            total_FP += ((outputs_bin == 1) & (labels == 0)).sum().item()
            total_FN += ((outputs_bin == 0) & (labels == 1)).sum().item()

    # Final metrics calculation
    test_loss = running_loss / len(dataloader)
    accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN) # total correct / total samples
    precision = total_TP / (total_TP + total_FP) # how many of the predicted positives are actually positive
    recall = total_TP / (total_TP + total_FN) # how many of the actual positives are predicted as positive
    f1 = 2 * (precision * recall) / (precision + recall)

    # Print results
    print(f"Test loss: {test_loss}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    #model evaluation of model_van_30 (30 epochs, no transformations and weight initialization)
    # Test loss: 0.39729388753573097 --> the lower the better
    # Accuracy: 0.8178928458753008  --> 
    # Precision: 0.91446923903471
    # Recall: 0.8323932903366669
    # F1: 0.8715031045581975

  



if  __name__ == "__main__":


  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = UNet(1,1).to(device)
  model.load_state_dict(torch.load("../results/model_van_30.pth" ,map_location=device, weights_only=True))
  criterion = nn.BCEWithLogitsLoss() #binary cross entropy loss with sigmoid


  # Initial transform for computing mean and std
  computing_mean_std_dataset = Dataset(train_image_path, train_label_path, image_transform=basic_transform, label_transform=basic_transform)
  computing_mean_std_dataset_loader = DataLoader(computing_mean_std_dataset, batch_size=1, shuffle=True)


  mean, std = computeMeanStd(computing_mean_std_dataset_loader)

  print(f"Mean: {mean}, Std: {std}")


  # Datasets and loaders
  # Vanilla Dataset: use basic_transform_resize
  # Mirrored Dataset: use advanced_transform_image and advanced_transform_label or : basic_transform for both
  test_dataset = Dataset(test_image_path, test_label_path, image_transform=basic_transform, label_transform=basic_transform)
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
  
  #Running code:
  test(model, test_loader, criterion)


