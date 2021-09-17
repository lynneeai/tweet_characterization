import csv
import logging
import math
import os
import pickle
import random

import torch
import pandas as pd
from torchvision.transforms.transforms import ToTensor
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

from config import TRAIN_CONFIG
from utils import init_logger, program_sleep

"""Init logger"""
if not os.path.exists(TRAIN_CONFIG.LOGS_ROOT):
    os.makedirs(TRAIN_CONFIG.LOGS_ROOT)
log_filename = os.path.basename(__file__).split(".")[0]
init_logger(TRAIN_CONFIG.LOGS_ROOT, log_filename)
LOGGER = logging.getLogger(log_filename)
"""------------------"""

tqdm.pandas()


class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_file_path, image_dir, image_size=224, transformer=None):
        super().__init__()
        
        self.image_dir = image_dir
        self.image_size = image_size
        self.transformer = transformer
        
        # load dataframe
        self.df = pd.read_csv(tsv_file_path, sep="\t", encoding="utf-8")
        self.df["id"] = self.df["id"].astype(str)
        self.df["text"] = self.df["text"].astype(str)
        self.df["label"] = self.df["label"].astype(int)
        self.df["label_name"] = self.df["label"].progress_apply(lambda x: TRAIN_CONFIG.LABEL_NAMES[x])
    
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        image = self.load_image(idx)
        if self.transformer:
            image = self.transformer(image)
            
        sample = {
            "image_path": f"{self.image_dir}/{self.df['id'][idx]}.jpg",
            "image": image,
            "text": self.df["text"][idx],
            "label": self.df["label"][idx],
            "label_name": self.df["label_name"][idx]
        }
        
        return sample
        
    def load_image(self, idx):
        id = self.df["id"][idx]
        image_path = f"{self.image_dir}/{id}.jpg"
        # open image
        image = Image.open(image_path).resize((self.image_size, self.image_size), Image.ANTIALIAS).convert("RGB")
        return image
    
    
def load_dataloaders():
    train_dataset = ImageTextDataset(tsv_file_path=f"{TRAIN_CONFIG.TSV_ROOT}/train.tsv", image_dir=f"{TRAIN_CONFIG.IMAGE_ROOT}/train")
    dev_dataset = ImageTextDataset(tsv_file_path=f"{TRAIN_CONFIG.TSV_ROOT}/validate.tsv", image_dir=f"{TRAIN_CONFIG.IMAGE_ROOT}/validate")
    test_dataset = ImageTextDataset(tsv_file_path=f"{TRAIN_CONFIG.TSV_ROOT}/test.tsv", image_dir=f"{TRAIN_CONFIG.IMAGE_ROOT}/test")
    
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG.BATCH_SIZE, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=TRAIN_CONFIG.BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG.BATCH_SIZE, shuffle=True)
    
    return train_dataloader, dev_dataloader, test_dataloader


test_dataset = ImageTextDataset(
    tsv_file_path=f"{TRAIN_CONFIG.TSV_ROOT}/test.tsv", 
    image_dir=f"{TRAIN_CONFIG.IMAGE_ROOT}/test", 
    transformer=transforms.Compose([
        transforms.ToTensor()
    ])
)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

from transformers import CLIPTokenizer, CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

for items in test_dataloader:
    image_inputs = items["image"]
    image_features = model.get_image_features(image_inputs)
    print(image_features)
    
    image_paths = items["image_path"]
    images = [Image.open(ip).convert("RGB") for ip in image_paths]
    image_inputs = processor(images=images, return_tensors="pt")
    image_features = model.get_image_features(**image_inputs)
    print(image_features)
    
    break