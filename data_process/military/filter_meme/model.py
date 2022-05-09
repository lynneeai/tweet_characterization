import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pprint import pprint


class MemeDetector(nn.Module):
    def __init__(self, device, output_size=2):
        super(MemeDetector, self).__init__()
        
        self.device = device
        
        self.resnet = timm.create_model("hf_hub:sgugger/resnet50d", pretrained=True)
        # for _, param in self.resnet.named_parameters():
        #     param.requires_grad = False
        
        resnet_config = self.resnet.default_cfg
        self.input_transform = timm.data.transforms_factory.transforms_imagenet_eval(
            img_size=resnet_config["input_size"][-2:],
            interpolation=resnet_config["interpolation"],
            mean=resnet_config["mean"],
            std=resnet_config["std"],
        )
        
        self.dropout = nn.Dropout(p=0.2)
        self.output_enc1 = nn.Linear(1000, 512)
        self.output_enc2 = nn.Linear(512, 256)
        self.output_enc3 = nn.Linear(256, 32)
        self.out = nn.Linear(32, output_size)
        
    def forward(self, image_files):
        images = [Image.open(im).convert("RGB") for im in image_files]
        input_tensors = [self.input_transform(im) for im in images]
        inputs = torch.stack(input_tensors).to(self.device)
        
        # resnet embed
        # with torch.no_grad():
        resnet_embed = self.resnet(inputs)
        
        outputs = self.dropout(resnet_embed)
        outputs = F.relu(self.output_enc1(outputs))
        
        outputs = self.dropout(outputs)
        outputs = F.relu(self.output_enc2(outputs))
        
        outputs = self.dropout(outputs)
        outputs = F.relu(self.output_enc3(outputs))
        
        outputs = self.out(outputs)
        outputs = F.log_softmax(outputs, dim=1)
        return outputs
    
    def loss(self, outputs, labels):
        return F.nll_loss(outputs, labels)