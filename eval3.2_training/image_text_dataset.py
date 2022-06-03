import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from torchvision.transforms.transforms import ToTensor
from torchvision import transforms
from PIL import Image

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

tqdm.pandas()

class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_file_path, image_size=224, transformer=transforms.Compose([ToTensor()])):
        super(ImageTextDataset, self).__init__()
        
        self.image_size = image_size
        self.transformer = transformer
        
        # load dataframe
        self.df = pd.read_csv(tsv_file_path, sep="\t", encoding="utf-8")
        self.df["tid"] = self.df["tid"].astype(str)
        self.df["text"] = self.df["text"].astype(str)
        self.df["image_file"] = self.df["image_file"].astype(str)
        self.df["label"] = self.df["label"].astype(int)
    
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        image = self.load_image(idx)
        if self.transformer:
            image = self.transformer(image)
            
        sample = {
            "tid": self.df["tid"][idx],
            "text": self.df["text"][idx],
            "image_file": self.df["image_file"][idx],
            "image": image,
            "label": torch.tensor(self.df["label"][idx])
        }
        
        return sample
        
    def load_image(self, idx):
        image_file = self.df["image_file"][idx]
        # open image
        image = Image.open(image_file).resize((self.image_size, self.image_size), Image.ANTIALIAS).convert("RGB")
        return image