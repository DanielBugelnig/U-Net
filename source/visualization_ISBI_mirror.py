# Script for visualizing and testing the U-net model
# Explanation of code in visualization_ReadMe.md

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
import torchvision.models as models
import torchvision.transforms.functional as Trans

from architecture import UNet

# Count images in TIF file
def countImages(image):
    # Count images
    image_count = 0
    try: 
        while True:       
            image.seek(image_count)
            image_count +=1
    except EOFError:
        pass

    print(f"Number of pages: {image_count}\n")
    return image_count

# Display all Images of the dataset
def displayImages(image, title):
    plt.figure(figsize=(20,20))
    for i in range(countImages(image)):
        try:
            image.seek(i)
            plt.subplot(6,5, i+1)
            plt.imshow(image)
            plt.axis("off")
            plt.title(f"Page {i+1}")
        except EOFError:
            break

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show(block=False)

# Prediction function for a single image
def predict_single(image, model):
    model.eval()
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        pred = model(image)
    return pred
    

# function only for testing purposes of visualization
def evaluate_test(prediction, expected):
    probabilities = torch.sigmoid(prediction)
    probabilities = prediction
    #predicted_segmentation = (probabilities > 0.5).int()  #convert to binary mask (1,1,H,W)
    predicted_segmentation = probabilities   #convert to binary mask (1,1,H,W)


    _,H,W = predicted_segmentation.shape
    H=388
    W=388   
    expected_segmentation = Trans.center_crop(expected, [H,W]) # crop the expected label to size of model output (from 512 to 388)
    predicted_segmentation = Trans.center_crop(predicted_segmentation, [H,W]) # crop the expected label to size of model output (from 512 to 388)

    diff = (expected_segmentation != predicted_segmentation).int()
    return predicted_segmentation, expected_segmentation, diff  #return both binary masks

# correct evaluation function
def evaluate(prediction, expected):
    probabilities = torch.sigmoid(prediction)
    predicted_segmentation = (probabilities > 0.5).int()  #convert to binary mask (1,1,H,W)

    _,_,H,W = predicted_segmentation.shape  
    expected_segmentation = Trans.center_crop(expected, [H,W]) # crop the expected label to size of model output (from 512 to 388)

    diff = (expected_segmentation != predicted_segmentation).int() # difference mask
    return predicted_segmentation, expected_segmentation, diff  #return both binary masks


# selecting device and loading model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(1,1).to(device)
#model.load_state_dict(torch.load("../results/model_120ep_dropout10_mirror_transf_normalized.pth" ,map_location=device, weights_only=True))
model.load_state_dict(torch.load("../results/model_120ep_dropout10_mirror_transf_normalized.pth" ,map_location=device, weights_only=True))



# ----------------------------------------------------------------------
# Displaying the complete  train/test dataset
#Training images
train_image = Image.open("../ISBI-2012-mirrored/train-mirror.tif") 
#displayImages(train_image, "Training images")
# Training labels
train_label = Image.open("../ISBI-2012-mirrored/train-labels.tif")
#displayImages(train_label, "Training labels")

# #Test images
test_image = Image.open("../ISBI-2012-mirrored/test-mirror.tif")

# displayImages(test_image, "Test images")
# Training labels
test_label = Image.open("../ISBI-2012-mirrored/test-labels.tif")
# displayImages(test_label, "Test labels")
#input("press enter for close")

#---------------------------------------------------------------------


# --------------------------------------------------------------------
# Visualization
# What we want to display: Image, Prediction; Groundtruth, Difference
# Image size = 512x512,
# Prediction size = 388x388
# select a random image of test set
randomID = random.randint(1, countImages(test_image))
test_image.seek(randomID)
test_label.seek(randomID)

plt.figure(figsize=(30,30))
'''
# plotting 30 images, labels
for i in range(0,30):
    test_image.seek(i)
    test_label.seek(i)
    plt.subplot(6,10, i+1)
    plt.imshow(test_image,cmap='viridis')
    plt.title(f"Mirrored Test set{i}")
    plt.axis("off")

    plt.subplot(6,10, i+31)
    plt.imshow(test_label, cmap='viridis')
    plt.title(f"Label test set{i}")
    plt.axis("off")
plt.show()
input("press enter for close")
'''


# Test image
plt.subplot(2,2, 1)
plt.imshow(test_image,cmap='viridis')
plt.title(f"Test image")
plt.axis("off")



# Test label
plt.subplot(2,2, 2)
plt.imshow(test_label, cmap='viridis')
plt.title(f"Test label")
plt.axis("off")




# Transform the test image/label to tensor
transform = transforms.Compose([transforms.ToTensor()]) # convert PIL image to (C,H,W)
 # Convert to grayscale
test_image = test_image.convert("L")
image_tensor = transform(test_image)
label_tensor = transform(test_label)
#image_tensor = image_tensor.unsqueeze(0) #  Shape (1,C,H,W)
#torch.set_printoptions(threshold=torch.inf)
print(f"Tensor size image_tensor: {image_tensor.shape}") # (1,1,512,512)
print(image_tensor)
print(f"Tensor size label tensor: {label_tensor.shape}") # (1,1,512,512)
#print(label_tensor)

# Run the U-net for one image
#prediction = predict_single(test_image, model)
prediction = predict_single(image_tensor, model)

prediction_mask, label_mask, diff = evaluate(prediction, label_tensor)
print(f"Tensor size prediction mask: {prediction_mask.shape}") 
print(f"Tensor size label mask: {label_mask.shape}") 
prediction_mask.squeeze_(0).squeeze_(0)
print(f"Tensor size prediction mask: {prediction_mask.shape}")
label_mask.squeeze_(0)
diff.squeeze_(0).squeeze_(0)
plt.subplot(2,2,2)
plt.imshow(prediction_mask, cmap='viridis')
plt.title(f"Prediction")
cbar = plt.colorbar()
cbar.set_label("0: membrane, 1: membrane")
plt.axis("off")
plt.subplot(2,2,3)
plt.imshow(label_mask, cmap='viridis')
cbar = plt.colorbar()
cbar.set_label("0: membrane, 1: cell")
plt.title(f"GT")
plt.axis("off")
plt.subplot(2,2,4)
plt.imshow(diff, cmap='viridis')
cbar = plt.colorbar()
cbar.set_label("0: correct, 1: wrong")
plt.title(f"Difference")
plt.axis("off")
# add colorbar

plt.show()