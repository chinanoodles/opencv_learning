import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()  # Initialize the nn.Module part of this class
        
        # 1 input image channel (grayscale), 10 output channels/feature maps
        # 3x3 square convolution kernel
        # ( w-f )/s+1
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)

        # conv1 = ( w-f )/s+1 (224 - 3)/1 +1 = 222
        # pool1 = ( w-f )/s+1 (222 - 3)/2 +1 = 111
        # conv2 = ( w-f )/s+1 (111 - 3)/1 +1 = 109
        # pool2 = ( w-f )/s+1 (109 - 3)/2 +1 = 54       
        # conv3 = ( w-f )/s+1 (54 - 3)/1 +1 = 52
        # pool3 = ( w-f )/s+1 (52 - 3)/2 +1 = 26
        # conv4 = ( w-f )/s+1 (26 - 3)/1 +1 = 24
        # pool4 = ( w-f )/s+1 (24 - 3)/2 +1 = 12


        # Max-pooling layer
        self.pool = nn.MaxPool2d(2, 2)        
        self.fc1 = nn.Linear(256 * 12 * 12, 512)  # Fully connected layer
        self.fc2 = nn.Linear(512, 136)  # Fully connected layer
        # Dropout layer to avoid overfitting
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        # Apply fully connected layer
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


