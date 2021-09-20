import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPConfig, CLIPTokenizer, CLIPProcessor, CLIPModel

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from config import TRAIN_CONFIG


class CLIP_MODEL(nn.Module):
    def __init__(self):
        super(CLIP_MODEL, self).__init__()
        
        self.clip_config = CLIPConfig()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        self.image_enc = nn.Linear(self.clip_config.projection_dim, 128)
        self.text_enc = nn.Linear(self.clip_config.projection_dim, 128)
        
        self.image_text_layer = nn.Linear(256, 64)
        self.out = nn.Linear(64, len(TRAIN_CONFIG.LABEL_NAMES))
        
    def forward(self, image_files, texts):
        images = [Image.open(im).convert("RGB") for im in image_files]
        image_inputs = self.clip_processor(images=images, return_tensors="pt")
        image_features = self.clip.get_image_features(**image_inputs)
        
        text_inputs = self.clip_tokenizer(texts, padding=True, return_tensors="pt")
        text_features = self.clip.get_text_features(**text_inputs)
        
        image_enc = F.relu(self.image_enc(image_features))
        text_enc = F.relu(self.text_enc(text_features))
        image_text_feats = F.relu(self.image_text_layer(torch.cat((image_enc, text_enc), dim=1)))
        
        output = self.out(image_text_feats)
        output = F.log_softmax(output, dim=1)
        
        return output
