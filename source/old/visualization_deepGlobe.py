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
    predicted_segmentation = (probabilities > 0.5).int()  #convert to binary mask (1,1,H,W)

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
model.load_state_dict(torch.load("../results/model2.pth" ,map_location=device, weights_only=True))



# ----------------------------------------------------------------------
# Displaying the complete  train/test dataset
#Training images
train_image = Image.open("../Deep_Global_Land/DeepGlobeLand_572_mirror/test/86805_sat_augmented.tif")
#displayImages(train_image, "Training images")
# Training labels
train_label = Image.open("../ISBI-2012-challenge/train-labels.tif")
#displayImages(train_label, "Training labels")

# #Test images
test_image = Image.open("../ISBI-2012-challenge/test-volume.tif")
# displayImages(test_image, "Test images")
# Training labels
test_label = Image.open("../ISBI-2012-challenge/test-labels.tif")
# displayImages(test_label, "Test labels")
#input("press enter for close")

#---------------------------------------------------------------------



# --------------------------------------------------------------------
# Visualization
# What we want to display: Image, Prediction; Groundtruth, Difference
# Image size = 512x512,
# Prediction size = 388x388
# select a random image of test set
count_images = countImages(test_image)
print(count_images)