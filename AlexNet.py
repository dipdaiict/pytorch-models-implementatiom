## ---------- AlexNet Architecture -------------------- ##
## AlexNet is a convolutional neural network (CNN) architecture designed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in 2012. 
## The model was proposed in the research paper “ImageNet Classification with Deep Convolutional Neural Networks” and won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012.
## Research paper: https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

import torch 
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(6*6*256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
    def forward(self, image):
        x = self.pool1(F.relu(self.conv1(image)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool3(F.relu(self.conv5(x)))
        x = x.view(-1, 6 * 6 * 256)
        # x = F.dropout(x)            # Dropout Implementation: by default p = 0.5
        x = F.relu(self.fc1(x))
        # x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
if __name__ == '__main__':
    model = AlexNet(num_classes=1000)
    print(f"Model Architecture:\n {model}")
    fake_input = torch.randn(1, 3, 227, 227)  # Shape: [batch_size, channels, height, width]
    outputs = model(fake_input)
    print("-----------------------------")
    print(f"Output Shape: {outputs.size()}")