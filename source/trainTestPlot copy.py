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


train_dataset = Dataset("ISBI-2012-challenge/train-volume.tif", "ISBI-2012-challenge/train-labels.tif", transform)
test_dataset = Dataset("ISBI-2012-challenge/test-volume.tif", "ISBI-2012-challenge/test-labels.tif", transform)
#Batch size?
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#print(train_dataset.shape)
#Training
#https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
else:
    device = torch.device("cpu")   # Use CPU

model = UNet(1,1).to(device)
optimizer= optim.Adam(model.parameters(),0.001)
criterion = nn.BCEWithLogitsLoss() #binary cross entropy loss with sigmoid



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
  with torch.no_grad():
    for images, labels in dataloader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      #outputs = (outputs > 0.5).int()
      labels = Trans.center_crop(labels, [388,388])
      loss = criterion(outputs,labels)
      running_loss+=loss.item() 
      #visualisation
      for j in range(dataloader.batch_size):
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(images[j].cpu().numpy().squeeze(), cmap='viridis')
        plt.subplot(2,2,2)
        plt.imshow(labels[j].cpu().numpy().squeeze(), cmap='viridis')
        plt.subplot(2,2,3)
        outputs[j]=(outputs[j]>0.5).int()
        plt.imshow(outputs[j].cpu().detach().numpy().squeeze(), cmap='viridis')
        plt.subplot(2,2,4)
        diff=(outputs[j]!=labels[j]).int()
        plt.imshow(diff.cpu().detach().numpy().squeeze(),cmap='viridis')
        plt.show()
        '''
        TO DO
        Output image to binary values

        TP = 
        TN =
        FP =
        FN =

        accuracy = (TP+TN)/(TP+TN+FP+FN)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 = 2*(precision*recall)/(precision+recall)

        print(f"Accuracy:{accuracy}")
        print(f"Precision:{precision}")
        print(f"Recall:{recall}")
        print(f"F1 Score:{f1}")
        '''
  test_loss = running_loss/len(dataloader)
  print(f"Test loss:{test_loss}")
  

#Running code:
print("Start Training")
train(model, train_loader, criterion, optimizer, 10)

print("Start Evaluation")
test(model, test_loader, criterion)

torch.save(model.state_dict(), "results/model1.pth")
