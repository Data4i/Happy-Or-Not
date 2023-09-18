import torch
import torch.nn as nn
import torchvision.transforms as T

class BaselineExpressionModelV0(nn.Module):
    def __init__(self, mean, std, num_classes: int):
        super(BaselineExpressionModelV0, self).__init__()
        self.mean = mean
        self.std = std
        self.num_classes = num_classes
        
        
        self.normalize = T.Compose([
            T.Normalize(mean = self.mean, std = self.std),
        ])
        
        self.features = nn.Sequential(
            # First Layer
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 11, stride = 4, padding = 2),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            
            # Second Layer 
            nn.Conv2d(in_channels = 64, out_channels = 192, kernel_size = 5, padding = 2),
            nn.BatchNorm2d(num_features = 192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            
            # Third Layer
            nn.Conv2d(in_channels = 192, out_channels = 384, kernel_size = 5, padding = 1),
            nn.BatchNorm2d(num_features = 384),
            nn.ReLU(),
            
            # Fourth Layer
            nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 5, padding = 1),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
            
            # Fifth Layer
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 5, padding = 2),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        
        self.classifier = nn.Sequential(
            # Sixth Layer
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(), 
            
            # Seventh Layer
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            
            # Eighth Layer import nn
            nn.Linear(4096, self.num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x 
            

class HnModelV1(nn.Module):
    def __init__(self, num_classes, mean, std):
        super(HnModelV1, self).__init__()
        self.num_classes = num_classes
        self.mean = mean
        self.std = std
        
        self.normalize = T.Compose([
            T.Normalize(mean=self.mean, std = self.std),           
        ])
        
        self.features = nn.Sequential(
            # Input Layer
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride = 4, padding=1),
            nn.BatchNorm2d(num_features = 96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Conv 2 Layer
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            
            # Conv 3 Layer
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features = 384),
            nn.ReLU(),
            
            # Conv 4 Layer
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features = 384),
            nn.ReLU(),
            
            # Conv 5 Layer
            nn.Conv2d(in_channels=384, out_channels = 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            # First fully connected Layer
            nn.Linear(256 * 4, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            
            nn.Linear(in_features=4096, out_features = num_classes)
        )
        

    def forward(self, x):
        x = self.normalize(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x