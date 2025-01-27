# Creating U-Net architecture 

# page 4 of the paper: (https://arxiv.org/pdf/1505.04597.pdf)

# Network Architecture
# The network architecture is illustrated in Figure 1. It consists of a contracting
# path (left side) and an expansive path (right side). The contracting path follows
# the typical architecture of a convolutional network. It consists of the repeated
# application of two 3x3 convolutions (unpadded convolutions), each followed by
# a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2
# for downsampling. At each downsampling step we double the number of feature
# channels. Every step in the expansive path consists of an upsampling of the
# feature map followed by a 2x2 convolution (“up-convolution”) that halves the
# number of feature channels, a concatenation with the correspondingly cropped
# feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in
# every convolution. At the final layer a 1x1 convolution is used to map each 64-
# component feature vector to the desired number of classes. In total the network
# has 23 convolutional layers.
# To allow a seamless tiling of the output segmentation map (see Figure 2), it
# is important to select the input tile size such that all 2x2 max-pooling operations
# are applied to a layer with an even x- and y-size.

# The reduced output size within a single tile (e.g., 388x388 for a 572x572 input) ensures that the predictions are based on full context, 
# avoiding incomplete or invalid segmentations near the borders. 




#pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F #for ReLu
import torchvision.transforms.functional as Trans
from torchinfo import summary

class UNet(nn.Module):
    def __init__(self, input_number, output_number):
        super(UNet, self).__init__()
        self.input_number = input_number
        self.output_number = output_number


        # Encoder
        # input: 4d tensor: batch_size x input_number x 572x572 --> input number=1 for grayscale image, 3 for RGB image
        # assuming 572x572 image
        self.conv1 = nn.Conv2d(self.input_number, 64, kernel_size=3, padding=0) # 64x570x570
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=0) # 64x568x568
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 64x284x284

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0) # 128x282x282
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=0) # 128x280x280
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 128x140x140

        self.conv5 = nn.Conv2d(128,256, kernel_size=3, padding=0) # 256x138x138
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=0) # 256x136x136
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 256x68x68

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=0) # 512x66x66
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=0) # 512x64x64
        self.maxPool4 = nn.MaxPool2d(kernel_size=2, stride=2) # 512x32x32

        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, padding=0) # 1024x30x30
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3, padding=0) # 1024x28x28

        # Decoder
        # Upsampling by a factor of 2, --> stride=2, kernel_size=2
        # 2x2 up convolution halves the feature channels

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # 512x56x56 --> output size formular(input) # stride * (input-1) + kernel_size -2 *padding + output_padding = 2*(28-1) + 2 - 2*0 + 0 = 56
        self.conv1b = nn.Conv2d(1024, 512, kernel_size=3, padding=0) # 512x54x54 other 512 features come from encoder site
        self.conv2b = nn.Conv2d(512, 512, kernel_size=3, padding=0) # 512x52x52

        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, 2) # 512x104x104
        self.conv3b = nn.Conv2d(512, 256, 3, padding=0) #256x102x102
        self.conv4b = nn.Conv2d(256, 256, 3, padding=0) # 256x100x100

        self.upconv3 = nn.ConvTranspose2d(256, 128, 2,2,) #256x200x200
        self.conv5b = nn.Conv2d(256, 128, 3, padding=0) #128x198x198
        self.conv6b = nn.Conv2d(128, 128, 3, padding=0) #128x196x196

        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, 2) #128x392x392
        self.conv7b = nn.Conv2d(128, 64, 3, padding=0) #64x390x390
        self.conv8b = nn.Conv2d(64,64,3, padding=0) # 64x388x388
        self.final_conv = nn.Conv2d(64, self.output_number, kernel_size=1, padding=0) #2x388x388
    
    def cropConcat(self, encoder, decoder):
        # crops the encoder tensor and concatenate its with the decoder tensor
        _,_,H,W = decoder.shape
        cropped_enc = Trans.center_crop(encoder, [H,W]) # crops the encoder tensor in the centre
        return torch.cat((cropped_enc, decoder), dim=1) # concatenates at the feature dimension

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x1 = F.relu(self.conv2(x))
        x = self.maxPool1(x1)

        x = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x = self.maxPool2(x2)

        x = F.relu(self.conv5(x)) 
        x3 = F.relu(self.conv6(x))
        x = self.maxPool3(x3)

        x = F.relu(self.conv7(x))
        x4 = F.relu(self.conv8(x))
        x = self.maxPool4(x4)

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        
        # Decoder
        x = self.upconv1(x)  # size 512x56x56

        x = self.cropConcat(x4, x) # concatination1 size 1024x56x56
        x = F.relu(self.conv1b(x))
        x = F.relu(self.conv2b(x))
        x = self.upconv2(x)

        x = self.cropConcat(x3,x)
        x = F.relu(self.conv3b(x))
        x = F.relu(self.conv4b(x))
        x = self.upconv3(x)

        x = self.cropConcat(x2,x)
        x = F.relu(self.conv5b(x))
        x = F.relu(self.conv6b(x))
        x = self.upconv4(x)

        x = self.cropConcat(x1,x)
        x = F.relu(self.conv7b(x))
        x = F.relu(self.conv8b(x))
        x = self.final_conv(x)
        return x

 
model = UNet(1,1)
summary(model, input_size=(1, 1, 572, 572))  # Example input size
