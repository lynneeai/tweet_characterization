from cgitb import text
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPTokenizer, CLIPProcessor, CLIPModel, CLIPConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

class CLIP_SENT(nn.Module):
    def __init__(self, device, output_size=2):
        super(CLIP_SENT, self).__init__()
        
        self.device = device
        
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")  
        
        self.sent_tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False, normalization=True)
        self.sent_model = AutoModelForSequenceClassification.from_pretrained(f"{current_file_dir}/sentiment_model", output_hidden_states=True)
        for _, param in self.sent_model.named_parameters():
            param.requires_grad = False
            
        self.image_enc = nn.Linear(self.clip.config.projection_dim, 128)
        self.text_enc = nn.Linear(self.clip.config.projection_dim, 128)
        self.sent_enc = nn.Linear(self.sent_model.config.hidden_size, 128)
        
        self.att = nn.Parameter(torch.FloatTensor(128, 1))
        
        # maliciousness output
        self.output_layer = nn.Linear(128 * 3, 64)
        self.out = nn.Linear(64, output_size)
        
    def forward(self, image_files, texts):
        # image enc
        images = [Image.open(im).convert("RGB") for im in image_files]
        image_inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        image_features = self.clip.get_image_features(**image_inputs)
        image_enc = F.relu(self.image_enc(image_features))
        
        # text enc
        text_inputs = self.clip_tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        # input sequence length <= 77
        if text_inputs["input_ids"].size(dim=1) > 77:
            text_inputs["input_ids"] = torch.narrow(text_inputs["input_ids"], 1, 0, 77) 
            text_inputs["attention_mask"] = torch.narrow(text_inputs["attention_mask"], 1, 0, 77) 
        text_features = self.clip.get_text_features(**text_inputs)
        text_enc = F.relu(self.text_enc(text_features))
        
        # sentiment enc
        sent_inputs = self.sent_tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(self.device)
        sent_hidden_states = self.sent_model(sent_inputs["input_ids"], attention_mask=sent_inputs["attention_mask"]).hidden_states
        sent_features = sent_hidden_states[-1][:, 0, :]
        sent_enc = F.relu(self.sent_enc(sent_features))
        
        # attention
        att_weights = []
        for enc in [image_enc, text_enc, sent_enc]:
            weights = torch.mean(torch.matmul(enc, self.att), dim=0)
            att_weights.append(weights)
        att_scores = F.softmax(torch.cat(att_weights, dim=0), dim=0)
        features_enc = []
        for i, enc in enumerate([image_enc, text_enc, sent_enc]):
            features_enc.append(att_scores[i] * enc)
        features_enc = torch.cat(features_enc, dim=1)
        
        # maliciousness output
        enc = F.relu(self.output_layer(features_enc))
        logits = self.out(enc)
        output = F.log_softmax(logits, dim=1)
        
        return output
    
    def loss(self, outputs, labels):
        loss = F.nll_loss(outputs, labels)
        return loss
        