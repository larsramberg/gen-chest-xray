from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision.io import read_image, ImageReadMode

class ChestXRayDataset(Dataset):
    '''
    Class intended for loading data and annotations for ChestXRay14
    '''
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, read_lib="torch"):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.target_labels = ['Edema', 'Hernia', 'Atelectasis', 'Consolidation', 'Pneumonia', 'Infiltration', 'Nodule', 'Pleural_Thickening', 'Effusion',  'Cardiomegaly', 'Emphysema', 'Pneumothorax', 'Mass', 'Fibrosis']
        self.image_read_lib = read_lib

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        if self.image_read_lib == "torch":
            image = read_image(path=img_path, mode=ImageReadMode.RGB)
        elif self.image_read_lib == "pil":
            image = Image.open(fp=img_path).convert("RGB")
        else:
            raise Exception("Invalid image read library. Ensure read_lib is either 'torch' or 'pil'")
            
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label, self.target_labels)
        return image, label
        