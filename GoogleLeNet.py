import torch
import torch.nn as nn

# Here red mean Reduce Part:
# During Ensuring the Model is Working or Not Please Carefully Check below Conditions:
# If you are tarining than set model.train() and aux_classifier = True &
# If you are evaluating/testing the model then you should use model.eval() and aux_classifier = False and Remove Print Statament Output1 & Output2.
# Research Paper: Going deeper with convolutions: https://arxiv.org/pdf/1409.4842

# Inception Block:
class Inception(nn.Module):
    def __init__(self, in_channels, k_1x1, k_3x3_red, k_3x3, k_5x5_red, k_5x5, pool_proj):
        super(Inception, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=k_1x1, kernel_size=1),
            nn.ReLU(inplace=True))
        
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=k_3x3_red, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=k_3x3_red, out_channels=k_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=k_5x5_red, kernel_size=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=k_5x5_red, out_channels=k_5x5, kernel_size=5, padding=1),
            nn.ReLU(inplace=True))
        
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1, padding=1),
            nn.ReLU(inplace=True))
        
    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)

class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxClassifier, self).__init__()
        self.pool1 = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=True))
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 128, out_features=1024),
            nn.ReLU(inplace=True))
        self.drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)
     
    def forward(self, x):
        x = self.pool1(x)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x    
    
class GoogleLeNet(nn.Module):
    def __init__(self, num_classes, aux_classifier=True):
        super(GoogleLeNet, self).__init__()
        self.aux_classifier = aux_classifier
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),)
        
        self.block3a = Inception(in_channels=192, k_1x1=64, k_3x3_red=96, k_3x3=128, k_5x5_red=16, k_5x5=32, pool_proj=32)
        self.block3b = Inception(in_channels=256, k_1x1=128, k_3x3_red=128, k_3x3=192, k_5x5_red=32, k_5x5=96, pool_proj=64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block4a = Inception(in_channels=480, k_1x1=192, k_3x3_red=96, k_3x3=208, k_5x5_red=16, k_5x5=48, pool_proj=64)
        if self.aux_classifier:
            self.aux0 = AuxClassifier(in_channels=512, num_classes=num_classes)
        self.block4b = Inception(in_channels=512, k_1x1=160, k_3x3_red=112, k_3x3=224, k_5x5_red=24, k_5x5=64, pool_proj=64)
        self.block4c = Inception(in_channels=512, k_1x1=128, k_3x3_red=128, k_3x3=256, k_5x5_red=24, k_5x5=64, pool_proj=64)
        self.block4d = Inception(in_channels=512, k_1x1=112, k_3x3_red=144, k_3x3=288, k_5x5_red=32, k_5x5=64, pool_proj=64)
        if self.aux_classifier:
            self.aux1 = AuxClassifier(in_channels=528, num_classes=num_classes)
        self.block4e = Inception(in_channels=528, k_1x1=256, k_3x3_red=160, k_3x3=320, k_5x5_red=32, k_5x5=128, pool_proj=128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block5a = Inception(in_channels=832, k_1x1=256, k_3x3_red=160, k_3x3=320, k_5x5_red=32, k_5x5=128, pool_proj=128)
        self.block5b = Inception(in_channels=832, k_1x1=384, k_3x3_red=192, k_3x3=384, k_5x5_red=48, k_5x5=128, pool_proj=128)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.drop = nn.Dropout(p=0.4)
        self.classifier = nn.Linear(in_features=16384, out_features=num_classes)
        
    def forward(self, x):
        output0, output1 = None, None  
        x = self.head(x)
        x = self.block3a(x)
        x = self.block3b(x)
        x = self.pool1(x)
        x = self.block4a(x)
        if self.training and self.aux_classifier:
            output0 = self.aux0(x)
        x = self.block4b(x)
        x = self.block4c(x)
        x = self.block4d(x)
        if self.training and self.aux_classifier:
            output1 = self.aux1(x)
        x = self.block4e(x)
        x = self.pool2(x)
        x = self.block5a(x)
        x = self.block5b(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.classifier(x)
        if self.training:
            return output0, output1, x
        else:
            return x
    
if __name__ == '__main__':     
    fake_input = torch.randn(1, 3, 224, 224) 
    num_classes = 1000
    print(f"Model Architecture:")
    model = GoogleLeNet(num_classes=num_classes, aux_classifier=True) # Training So We are aux_classifier is equal to True & Evaluation Mode aux_classifier is False.
    print(model)    
    print("--------------------------------")
    model.train()
    output0, output1, output2 = model(fake_input)
    
    # Comment out Output 0 & 1 when you are in testing model architecture........
    print("Output 0 shape:", output0.shape)  # Shape of first auxiliary classifier output # In Training So We are Get the Output0 in Evaluation Mode we get the error.
    print("Output 1 shape:", output1.shape)  # Shape of second auxiliary classifier output # In Training So We are Get the Output1 in Evaluation Mode we get the error.
    print("Output 2 shape:", output2.shape)  # Shape of final classifier output
