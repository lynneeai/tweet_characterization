import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPTokenizer, CLIPProcessor, CLIPModel, CLIPConfig

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""


class CLIP_MODEL(nn.Module):
    def __init__(self, device, output_size):
        super(CLIP_MODEL, self).__init__()
        
        self.device = device

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        self.image_enc = nn.Linear(self.clip.config.projection_dim, 128)
        self.text_enc = nn.Linear(self.clip.config.projection_dim, 128)
        
        # maliciousness output
        self.output_layer = nn.Linear(256, 64)
        self.out = nn.Linear(64, output_size)
        
        # polar output
        self.polar_output_layer = nn.Linear(256, 64)
        self.polar_out = nn.Linear(64, output_size)
        
        # call_to_action output
        self.cta_output_layer = nn.Linear(256, 64)
        self.cta_out = nn.Linear(64, output_size)
        
        # viral output
        self.viral_output_layer = nn.Linear(256, 64)
        self.viral_out = nn.Linear(64, output_size)
        
        # sarcasm output
        self.sarcasm_output_layer = nn.Linear(256, 64)
        self.sarcasm_out = nn.Linear(64, output_size)
        
    def forward(self, image_files, texts):
        images = [Image.open(im).convert("RGB") for im in image_files]
        image_inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        image_features = self.clip.get_image_features(**image_inputs)
        
        text_inputs = self.clip_tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        text_features = self.clip.get_text_features(**text_inputs)
        
        image_enc = F.relu(self.image_enc(image_features))
        text_enc = F.relu(self.text_enc(text_features))
        
        # maliciousness output
        enc = F.relu(self.output_layer(torch.cat((image_enc, text_enc), dim=1)))
        logits = self.out(enc)
        outputs = F.log_softmax(logits, dim=1)
        
        # polar output
        polar_enc = F.relu(self.polar_output_layer(torch.cat((image_enc, text_enc), dim=1)))
        polar_logits = self.polar_out(polar_enc)
        polar_outputs = F.log_softmax(polar_logits, dim=1)
        
        # call_to_action output
        cta_enc = F.relu(self.cta_output_layer(torch.cat((image_enc, text_enc), dim=1)))
        cta_logits = self.cta_out(cta_enc)
        cta_outputs = F.log_softmax(cta_logits, dim=1)
        
        # viral output
        viral_enc = F.relu(self.viral_output_layer(torch.cat((image_enc, text_enc), dim=1)))
        viral_logits = self.viral_out(viral_enc)
        viral_outputs = F.log_softmax(viral_logits, dim=1)
        
        # sarcasm output
        sarcasm_enc = F.relu(self.sarcasm_output_layer(torch.cat((image_enc, text_enc), dim=1)))
        sarcasm_logits = self.sarcasm_out(sarcasm_enc)
        sarcasm_outputs = F.log_softmax(sarcasm_logits, dim=1)
        
        # intent outputs dict
        intent_outputs_dict = {
            "POLAR": polar_outputs,
            "CALL_TO_ACTION": cta_outputs,
            "VIRAL": viral_outputs,
            "SARCASM": sarcasm_outputs
        }
        
        return outputs, logits, intent_outputs_dict

    def loss(self, outputs, labels, intent_outputs_dict, intents_dict, intent_names):
        # maliciousness loss
        loss = F.nll_loss(outputs, labels)
        # intents loss
        for intent in intent_names:
            loss += F.nll_loss(intent_outputs_dict[intent], intents_dict[intent])
        
        return loss