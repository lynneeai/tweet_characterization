from PIL import Image
import requests

from transformers import CLIPTokenizer, CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

image1 = Image.open("1aansr.jpg").convert("RGB")
image2 = Image.open("1ae5iy.jpg").convert("RGB")
images = [image1, image2]
image_inputs = processor(images=images, return_tensors="pt")

text1 = "onelegged man takes homeless womans car for joyride"
text2 = "goats battle head to head"
texts = [text1, text2]
text_inputs = tokenizer(texts,  padding=True, return_tensors="pt")

image_features = model.get_image_features(**image_inputs)
text_features = model.get_text_features(**text_inputs)

print(image_features)
print(text_features)

# import torch
# import clip
# import numpy as np

# model, preprocess = clip.load("ViT-B/32")
# model.eval()
# input_resolution = model.visual.input_resolution
# context_length = model.context_length
# vocab_size = model.vocab_size

# print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
# print("Input resolution:", input_resolution)
# print("Context length:", context_length)
# print("Vocab size:", vocab_size)