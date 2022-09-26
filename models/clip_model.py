import os
import sys
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPTokenizer, CLIPProcessor, CLIPModel


class CLIP_MODEL(nn.Module):
    def __init__(self, device, output_size=2):
        super(CLIP_MODEL, self).__init__()
        
        self.device = device
        
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        self.text_enc = nn.Linear(self.clip.config.projection_dim, 128)
        self.image_enc = nn.Linear(self.clip.config.projection_dim, 128)
        
        self.text_image_att = nn.Parameter(torch.Tensor(128, 1))
        
        self.output_layer = nn.Linear(256, 64)
        self.out = nn.Linear(64, output_size)
        
    def forward(self, image_files, texts):
        text_inputs = self.clip_tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        # input sequence length <= 77
        if text_inputs["input_ids"].size(dim=1) > 77:
            text_inputs["input_ids"] = torch.narrow(text_inputs["input_ids"], 1, 0, 77) 
            text_inputs["attention_mask"] = torch.narrow(text_inputs["attention_mask"], 1, 0, 77) 
        text_features = self.clip.get_text_features(**text_inputs)
        
        images = [Image.open(im).convert("RGB") for im in image_files]
        image_inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        image_features = self.clip.get_image_features(**image_inputs)
        
        text_enc = F.relu(self.text_enc(text_features))
        image_enc = F.relu(self.image_enc(image_features))
        
        # attention
        att_text = torch.mean(torch.matmul(text_enc, self.text_image_att), dim=0)
        att_image = torch.mean(torch.matmul(image_enc, self.text_image_att), dim=0)
        att_score = F.softmax(torch.cat((att_text, att_image), dim=0), dim=0)
        feature_enc = torch.cat([att_score[0] * text_enc, att_score[1] * image_enc], dim=1)
        
        enc = F.relu(self.output_layer(feature_enc))
        logits = self.out(enc)
        outputs = F.log_softmax(logits, dim=1)
        
        return outputs, logits
    
    def loss(self, outputs, labels):
        loss = F.nll_loss(outputs, labels)
        return loss