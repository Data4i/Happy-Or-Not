import os
import torch
from torch.utils.data import Dataset, DataLoader
import  torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

class HNDataset(Dataset):
    def __init__(self, root_dir, transforms = None, happy_transforms = None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.happy_transforms = happy_transforms
        self.class_to_index = {'happy': 0, 'not_happy': 1}
        self.image_paths = self._load_image_paths()
        
    def get_class_index(self, name):
        return self.class_to_index.get(name, 1)
    
    def _load_image_paths(self):
        image_paths = []
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                class_idx = self.get_class_index(class_name)
                if class_idx != -1:
                    for filename in os.listdir(class_dir):
                        image_paths.append((os.path.join(class_dir, filename), class_idx))
        return image_paths
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path, label = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transforms:
            image = self.transforms(image)
            
        if self.happy_transforms and label == 0:
            image = T.ToPILImage()(image)
            image = self.happy_transforms(image)
            
        label_tensor = torch.zeros(len(self.class_to_index))
        label_tensor[label] = 1
        
        return image, label_tensor
    
def mean_and_std(dataset:Dataset):
    ds = DataLoader(dataset, shuffle = True)
    train_images = torch.stack([img_t for img_t, _ in iter(ds)], dim=3)
    mean, std = train_images.view(3, -1).mean(dim=1), train_images.view(3, -1).std(dim=1)
    return mean, std

def acc_plot(history):
    plt.plot(history['epochs'], history['train_acc'], c= 'b', label = 'training acc')
    plt.plot(history['epochs'], history['test_acc'], c = 'r', label = 'testing acc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc = 0)
    plt.show()

def loss_plot(history):
    plt.plot(history['epochs'], history['train_loss'], c= 'b', label = 'training loss')
    plt.plot(history['epochs'], history['test_loss'], c = 'r', label = 'testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc = 0)
    plt.show()