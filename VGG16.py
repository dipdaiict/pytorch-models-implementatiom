# VGG Very Deep Convolutional Networks (VGGNet)

import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self, features, num_classes):
        """
        Initialize the VGG network.

        Parameters
        ----------
        features : nn.Module
            A PyTorch module containing the convolutional layers of the VGG network.
        num_classes : int
            The number of classes in the classification task.

        Attributes
        ----------
        features : nn.Sequential
            A PyTorch Sequential module containing the convolutional layers of the VGG network.
        classifier_fc : nn.Sequential
            A PyTorch Sequential module containing the fully connected layers of the VGG network.
        """
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.classifier_fc = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        image : torch.Tensor
            Input tensor of shape (batch_size, 3, height, width)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(image)
        x = x.view(x.size(0), -1)  # Flatten the feature maps into a vector
        x = self.classifier_fc(x)
        return x
    
if __name__ == '__main__':
    # Create an instance of the VGG network
    vgg = VGG(features=None, num_classes=1000)
    print(f"Model Architecture:\n {vgg}")
    
    # Generate a random input tensor
    fake_input = torch.randn(2, 3, 224, 224)  # Shape: [batch_size, channels, height, width]
    
    # Pass the input through the network
    outputs = vgg(fake_input)
    print("-----------------------------")
    print(f"Output Shape: {outputs.size()}")