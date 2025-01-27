import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as Trans
from PIL import Image
import matplotlib.pyplot as plt

from architecture import UNet

torch.cuda.empty_cache()

#Loading data
#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class Dataset(Dataset):
  def __init__(self, image_path, label_path, transform=None):
    self.images = Image.open(image_path)
    self.labels = Image.open(label_path)
    self.transform = transform

  def __len__(self):
    return self.images.n_frames

  def __getitem__(self, idx):
    #find specific frame
    self.images.seek(idx)
    self.labels.seek(idx)
    #grayscale conversion (if necessary)
    image = self.images.convert("L")
    label = self.labels.convert("L")
    if self.transform:
      image = self.transform(image)
      label = self.transform(label)
    return image, label

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([572,572])])


transform_enhanced = transforms.Compose([
      transforms.Resize((572, 572)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(degrees=15),
      transforms.ColorJitter(brightness=0.2, contrast=0.2),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5], std=[0.5])
  ])


def computeMeanStd(dataloader):
  # Compute mean and std
  mean = 0.0
  std = 0.0
  n_samples = 0

  for images, _ in dataloader:
      batch_samples = images.size(0)  # batch size (number of images)
      images = images.view(batch_samples, -1)  # Flatten the images
      mean += images.mean(1).sum(0)  # Mean of each image
      std += images.std(1).sum(0)    # Std of each image
      n_samples += batch_samples

  mean /= n_samples
  std /= n_samples

  return mean, std


#print(train_dataset.shape)
#Training
#https://pytorch.org/tutorials/beginner/introyt/trainingyt.html




def train(model, dataloader, criterion, optimizer, nrOfEpochs):
  model.train()
  for i in range(nrOfEpochs):
    running_loss = 0.0
    for images, labels in dataloader:
      images, labels = images.to(device), labels.to(device)
      #print(images.shape)
      optimizer.zero_grad()
      outputs=model(images)
      #outputs = (outputs > 0.5).int()
      labels = Trans.center_crop(labels, [388,388])
      #assert outputs.shape == labels.shape, f"Shape mismatch: {outputs.shape} vs {labels.shape}"
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    train_loss = running_loss/len(dataloader)
    print(f"Epoch {i+1}/{nrOfEpochs}\nLoss:{train_loss}")


#Evaluation
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
  


if __name__ == "__main__":
  
  train_dataset = Dataset("../ISBI-2012-challenge/train-volume.tif", "../ISBI-2012-challenge/train-labels.tif", transform)
  test_dataset = Dataset("../ISBI-2012-challenge/test-volume.tif", "../ISBI-2012-challenge/test-labels.tif", transform)
  #Batch size?
  train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

  mean, std = computeMeanStd(train_loader)
  print(f"Mean: {mean}, Std: {std}")
  input("Closing")
  

  if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
  else:
    device = torch.device("cpu")   # Use CPU

  model = UNet(1,1).to(device)
  optimizer= optim.Adam(model.parameters(),0.001)
  criterion = nn.BCEWithLogitsLoss() #binary cross entropy loss with sigmoid


  #Running code:
  print("Start Training")
  train(model, train_loader, criterion, optimizer, 10)

  print("Start Evaluation")
  test(model, test_loader, criterion)

  torch.save(model.state_dict(), "results/model1.pth")











