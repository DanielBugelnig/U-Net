import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as Trans
from PIL import Image
import matplotlib.pyplot as plt

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

transform = transforms.Compose([transforms.ToTensor()])

#original
#train_dataset = Dataset("ISBI-2012-challenge/train-volume.tif", "ISBI-2012-challenge/train-labels.tif", transform)
#test_dataset = Dataset("ISBI-2012-challenge/test-volume.tif", "ISBI-2012-challenge/test-labels.tif", transform)

#mirrored
train_dataset = Dataset("ISBI-2012-challenge/train-mirror.tif", "ISBI-2012-challenge/train-labels.tif", transform)
test_dataset = Dataset("ISBI-2012-challenge/test-mirror.tif", "ISBI-2012-challenge/test-labels.tif", transform)


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#print(train_dataset.images.n_frames)

#print(train_dataset.shape)

#Training
#https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(1,1).to(device)
optimizer= optim.Adam(model.parameters(),0.00001)
criterion = nn.BCEWithLogitsLoss() #binary cross entropy loss with sigmoid



def train(model, dataloader, criterion, optimizer, nrOfEpochs):
  model.train()
  for i in range(nrOfEpochs):
    running_loss = 0.0
    for images, labels in dataloader:
      images, labels = images.to(device), labels.to(device);
      #print(images.shape)
      optimizer.zero_grad()
      outputs=model(images)
      #outputs = (outputs > 0.5).float()
      labels = Trans.center_crop(labels, [388,388])
      #assert outputs.shape == labels.shape, f"Shape mismatch: {outputs.shape} vs {labels.shape}"
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    '''
    #for visualization of the training data
    for j in range(dataloader.batch_size):
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(images[j].cpu().numpy().squeeze(), cmap='viridis')
        plt.subplot(2,2,2)
        plt.imshow(labels[j].cpu().numpy().squeeze(), cmap='viridis')
    '''
    train_loss = running_loss/len(dataloader)
    print(f"Epoch {i+1}/{nrOfEpochs}\nLoss:{train_loss}")


#Evaluation
def test(model, dataloader, criterion):
  model.eval()
  running_loss = 0.0
  avgAcc=0.0
  avgPrec=0.0
  avgRec=0.0
  avgF1=0.0
  with torch.no_grad():
    for images, labels in dataloader:
      images, labels = images.to(device), labels.to(device);
      outputs = model(images)
      labels = Trans.center_crop(labels, [388,388])
      loss = criterion(outputs,labels)
      running_loss+=loss.item()
      outputs = (outputs > 0.5).int()
      #visualisation and evaluation
      for j in range(dataloader.batch_size):
        '''
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(images[j].cpu().numpy().squeeze(), cmap='viridis')
        plt.subplot(2,2,2)
        plt.imshow(labels[j].cpu().numpy().squeeze(), cmap='viridis')
        plt.subplot(2,2,3)
        #outputs[j]=(outputs[j]>0.5).int()
        plt.imshow(outputs[j].cpu().detach().numpy().squeeze(), cmap='viridis')
        plt.subplot(2,2,4)
        diff=(outputs[j]!=labels[j]).int()
        plt.imshow(diff.cpu().detach().numpy().squeeze(),cmap='viridis')
        plt.show()
        '''
        #Output image to binary values
        #outputs[j] = (outputs > 0.5).float()
        labels = labels.int()
        TP = ((labels[j])*(outputs[j])).sum()
        TN = ((1-labels[j])*(1-outputs[j])).sum()
        FP = ((1-labels[j])*(outputs[j])).sum()
        FN = ((labels[j])*(1-outputs[j])).sum()

        accuracy = (TP+TN)/(TP+TN+FP+FN)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 = 2*(precision*recall)/(precision+recall)
        '''
        print(f"Accuracy:{accuracy}")
        print(f"Precision:{precision}")
        print(f"Recall:{recall}")
        print(f"F1 Score:{f1}\n")
        '''
        avgAcc+=accuracy
        avgPrec+=precision
        avgRec+=recall
        avgF1+=f1
 
  print(f"Avg Accuracy:{avgAcc/len(dataloader)}")
  print(f"Avg Precision:{avgPrec/len(dataloader)}")
  print(f"Avg Recall:{avgRec/len(dataloader)}")
  print(f"Avg F1 Score:{avgF1/len(dataloader)}\n")

  test_loss = running_loss/len(dataloader)
  print(f"Test loss:{test_loss}")
  return avgAcc/len(dataloader), avgPrec/len(dataloader), avgRec/len(dataloader), avgF1/len(dataloader)

x = [0]*10
acc = [0]*10
prec = [0]*10
rec = [0]*10
f1 = [0]*10

#torch.load('unet_50ep_lr0001_mirr.pth', map_location=torch.device('cpu'))

#Running code:
for i in range(10):
  train(model, train_loader, criterion, optimizer, 10)
  x[i] = 10*i
  acc[i], prec[i], rec[i], f1[i] = test(model, test_loader, criterion)

#Plot acc, prec, rec, f1
plt.figure()
plt.plot(x, acc, label='Accuracy')
plt.plot(x, prec, label='Precision')
plt.plot(x, rec, label='Recall')
plt.plot(x, f1, label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Metrics over Epochs')
plt.legend()
plt.show()


#Save model - change name to: unet _ nr of epochs _ learning rate 0.xxx as lrxxx _ which dataset
torch.save(model.state_dict(), 'unet_100ep_lr00001_mirr.pth')
