import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""


class SENT_MODEL(nn.Module):
    def __init__(self, device, output_size=2):
        super(SENT_MODEL, self).__init__()
        
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False, normalization=True)
        self.sent_model = AutoModelForSequenceClassification.from_pretrained(f"{current_file_dir}/sentiment_model", output_hidden_states=True)

        for _, param in self.sent_model.named_parameters():
            param.requires_grad = False
        
        self.dropout = nn.Dropout(p=0.2)
        self.output_enc1 = nn.Linear(768, 256)
        self.output_enc2 = nn.Linear(256, 32)
        self.out = nn.Linear(32, output_size)
        
    def forward(self, texts):
        encodings = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(self.device)
        input_ids, att_mask = encodings["input_ids"], encodings["attention_mask"]
        hidden_states = self.sent_model(input_ids, attention_mask=att_mask).hidden_states
        
        sent_outputs = self.dropout(hidden_states[-1][:, 0, :])
        sent_outputs = F.relu(self.output_enc1(sent_outputs))
        
        sent_outputs = self.dropout(sent_outputs)
        sent_outputs = F.relu(self.output_enc2(sent_outputs))
        
        outputs = self.out(sent_outputs)
        outputs = F.log_softmax(outputs, dim=1)
        
        return outputs
    
    def loss(self, outputs, labels):
        return F.nll_loss(outputs, labels)