import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Pix2PixDataset(Dataset):
    def __init__(self, data_dir, transform):
        super().__init__()
        self.to_tensor = transforms.PILToTensor()
        self.apply_transformation = transform
        self.transform = transforms.Compose([
            transforms.Resize((286, 286)),
            transforms.RandomCrop((256, 256)),
            transforms.PILToTensor()
        ])
        self.data_dir = data_dir
        self.img_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                self.img_files.append(os.path.join(root, file))
    
    def __getitem__(self, index):
        img_path = self.img_files[index]
        img = Image.open(img_path)
        w, h = img.size
        x = img.crop((0, 0, w//2, h))
        y = img.crop((w//2, 0, w, h))
        
        if self.apply_transformation:
            x = self.transform(x)
            y = self.transform(y)
        else:
            x = self.to_tensor(x)
            y = self.to_tensor(y)
        return x, y
    
    def __len__(self):
        return len(self.img_files)