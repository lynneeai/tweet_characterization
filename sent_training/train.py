import logging
import os
import sys
<<<<<<< Updated upstream
import torch
import pandas as pd
from torchvision.transforms.transforms import ToTensor
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

=======
>>>>>>> Stashed changes

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
init_logger(TRAIN_CONFIG.LOGS_ROOT, TRAIN_CONFIG.OUTPUT_FILES_NAME)
LOGGER = logging.getLogger(log_filename)
"""------------------"""

from trainers import create_partitions, load_dataloaders, MULTIMODAL_TRAINER
from models.clip_sent import CLIP_SENT


<<<<<<< Updated upstream
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
    

if TRAIN_CONFIG.RECREATE_PARTITIONS:
    LOGGER.info("Recreating train/test/validate partitions...")
    create_partitions(TRAIN_CONFIG, tsv_fieldnames=["tid", "text", "image_file", "label", "label_name", "POLAR", "CALL_TO_ACTION", "VIRAL", "SARCASM", "HUMOR"])
train_dataloader, validate_dataloader, test_dataloader = load_dataloaders(TRAIN_CONFIG, ImageTextDataset)
=======
if TRAIN_CONFIG.RECREATE_PARTITIONS:
    LOGGER.info("Recreating train/test/validate partitions...")
    create_partitions(TRAIN_CONFIG)
train_dataloader, validate_dataloader, test_dataloader = load_dataloaders(TRAIN_CONFIG)
>>>>>>> Stashed changes

model_states_file = f"{TRAIN_CONFIG.TRAINED_MODELS_ROOT}/{TRAIN_CONFIG.OUTPUT_FILES_NAME}.weights.best"
model_file = f"{TRAIN_CONFIG.TRAINED_MODELS_ROOT}/{TRAIN_CONFIG.OUTPUT_FILES_NAME}.pth"
results_file = f"{TRAIN_CONFIG.RESULTS_ROOT}/{TRAIN_CONFIG.OUTPUT_FILES_NAME}_class_report.txt"

model = CLIP_SENT(device=TRAIN_CONFIG.DEVICE).to(TRAIN_CONFIG.DEVICE)

trainer = MULTIMODAL_TRAINER(
    model=model,
    config=TRAIN_CONFIG,
    dataloaders=[train_dataloader, validate_dataloader, test_dataloader],
    model_states_file=model_states_file,
    model_file=model_file,
    results_file=results_file,
    include_intents=False
)

if TRAIN_CONFIG.TRAIN:
    trainer.train()
if TRAIN_CONFIG.TEST:
    trainer.test()