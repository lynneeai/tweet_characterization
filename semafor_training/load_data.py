import csv
import os
import sys
import logging
import torch
import random
import math
import pandas as pd
from torchvision.transforms.transforms import ToTensor
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from config import TRAIN_CONFIG
from util_scripts.utils import init_logger

"""Init logger"""
if not os.path.exists(TRAIN_CONFIG.LOGS_ROOT):
    os.makedirs(TRAIN_CONFIG.LOGS_ROOT)
log_filename = os.path.basename(__file__).split(".")[0]
init_logger(TRAIN_CONFIG.LOGS_ROOT, log_filename)
LOGGER = logging.getLogger(log_filename)
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
        self.df["label_name"] = self.df["label_name"].astype(str)
        self.df["POLAR"] = self.df["POLAR"].astype(int)
        self.df["CALL_TO_ACTION"] = self.df["CALL_TO_ACTION"].astype(int)
        self.df["VIRAL"] = self.df["VIRAL"].astype(int)
        self.df["SARCASM"] = self.df["SARCASM"].astype(int)
        self.df["HUMOR"] = self.df["HUMOR"].astype(int)
    
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
            "label": torch.tensor(self.df["label"][idx]),
            "label_name": self.df["label_name"][idx],
            "POLAR": torch.tensor(self.df["POLAR"][idx]),
            "CALL_TO_ACTION": torch.tensor(self.df["CALL_TO_ACTION"][idx]),
            "VIRAL": torch.tensor(self.df["VIRAL"][idx]),
            "SARCASM": torch.tensor(self.df["SARCASM"][idx]),
            "HUMOR": torch.tensor(self.df["HUMOR"][idx]),
        }
        
        return sample
        
    def load_image(self, idx):
        image_file = self.df["image_file"][idx]
        # open image
        image = Image.open(image_file).resize((self.image_size, self.image_size), Image.ANTIALIAS).convert("RGB")
        return image
    
    
def create_partitions(train_split=TRAIN_CONFIG.TRAIN_SPLIT, dev_split=TRAIN_CONFIG.DEV_SPLIT):
    LOGGER.info("Creating partitions...")
    
    tweet_obj_list = []
    with open(f"{TRAIN_CONFIG.TSV_ROOT}/tweets.tsv", "r") as infile:
        tsv_reader = csv.DictReader(infile, delimiter="\t")
        for row in tsv_reader:
            tweet_obj_list.append(dict(row))
    
    # randomly generate train dev test sets
    total_tweets = len(tweet_obj_list)
    train_len = math.floor(total_tweets * train_split)
    dev_len = math.floor(total_tweets * dev_split)
    test_len = total_tweets - train_len - dev_len
    
    all_idx = [i for i in range(total_tweets)]
    random.shuffle(all_idx)
    
    train_idx = all_idx[:train_len]
    dev_idx = all_idx[train_len:train_len+dev_len]
    test_idx = all_idx[train_len+dev_len:]
    assert len(train_idx) == train_len
    assert len(dev_idx) == dev_len
    assert len(test_idx) == test_len
    
    # write to tsv files
    for batch_name, batch_idx in [("train", train_idx), ("dev", dev_idx), ("test", test_idx)]:
        with open(f"{TRAIN_CONFIG.TSV_ROOT}/{batch_name}.tsv", "w") as outfile:
            tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text", "image_file", "label", "label_name", "POLAR", "CALL_TO_ACTION", "VIRAL", "SARCASM", "HUMOR"])
            tsv_writer.writeheader()
            for idx in batch_idx:
                tweet_obj = tweet_obj_list[idx]
                tweet_obj["image_file"] = f"{project_root_dir}/{tweet_obj['image_file']}"
                tsv_writer.writerow(tweet_obj)
                outfile.flush()
                

def load_dataloaders(batch_size=TRAIN_CONFIG.BATCH_SIZE, tsv_root=TRAIN_CONFIG.TSV_ROOT):
    train_dataset = ImageTextDataset(f"{tsv_root}/train.tsv")
    dev_dataset = ImageTextDataset(f"{tsv_root}/dev.tsv")
    test_dataset = ImageTextDataset(f"{tsv_root}/test.tsv")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, dev_dataloader, test_dataloader
