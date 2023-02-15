import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class Pix2PixDataset(Dataset):
    def __init__(self, data_dir, transform):
        super().__init__()
        self.to_tensor = transforms.PILToTensor()
        self.apply_transformation = transform
        self.transform_both = A.Compose(
            [A.Resize(width=256, height=256),], additional_targets={'image0':'image'},
        )
        self.transform_input = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.2),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
            ToTensorV2(),
        ])
        self.transform_target = A.Compose([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
            ToTensorV2(),
        ])
        self.data_dir = data_dir
        self.img_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                self.img_files.append(os.path.join(root, file))
    
    def __getitem__(self, index):
        img_path = self.img_files[index]
        img = np.array(Image.open(img_path))
        x = img[:, :600, :]
        y = img[:, 600:, :]
        
        augmentations = self.transform_both(image=x, image0=y)
        x = augmentations['image']
        y = augmentations['image0']
        
        if self.apply_transformation:
            x = self.transform_input(image=x)['image']
            y = self.transform_target(image=y)['image']
        else:
            x = self.to_tensor(x)
            y = self.to_tensor(y)
            
        return x, y
    
    def __len__(self):
        return len(self.img_files)