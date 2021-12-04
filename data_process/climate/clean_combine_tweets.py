import csv
import os
import re
import json
import sys
import PIL
import pycld2 as cld2
from collections import defaultdict
from pprint import pprint
from PIL import Image

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/../.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

DATASET = "climate"
DATASETS_FOLDER = f"{project_root_dir}/datasets/{DATASET}"
PROCESSED_DATA_FOLDER = f"{project_root_dir}/processed_data/{DATASET}"
if not os.path.exists(PROCESSED_DATA_FOLDER):
    os.makedirs(PROCESSED_DATA_FOLDER)

# output files
TWEETS_TSV = f"{PROCESSED_DATA_FOLDER}/tweets.tsv"


# constants
LABEL_NAME_DICT = {"BENIGN": 0, "MALICIOUS": 1}

label_count = defaultdict(int)
tid2obj = {}
for intent in ["polar", "viral", "sarcasm", "humor"]:
    tweets_with_image = set()
    for image in os.scandir(f"{DATASETS_FOLDER}/images/{intent}_images/"):
        if image.name.endswith(".jpg"):
            try:
                opened_image = Image.open(image.path)
                tid = image.name.split(".")[0]
                tweets_with_image.add(tid)
            except PIL.UnidentifiedImageError:
                pass
    
    with open(f"{DATASETS_FOLDER}/tweets/{intent}.tsv", "r") as infile:
        tsv_reader = csv.DictReader(infile, delimiter = "\t")
        for row in tsv_reader:
            tid = row["tid"]
            if tid in tweets_with_image:
                label_name = "BENIGN" if intent in ["sarcasm", "humor"] else "MALICIOUS"
                if tid not in tid2obj:
                    tid2obj[tid] = {
                        "tid": tid,
                        "text": row["text"],
                        "image_file": f"datasets/{DATASET}/images/{intent}_images/{tid}.jpg",
                        "label": LABEL_NAME_DICT[label_name],
                        "label_name": label_name,
                        "POLAR": 0,
                        "CALL_TO_ACTION": 0,
                        "VIRAL": 0,
                        "SARCASM": 0,
                        "HUMOR": 0
                    }
                    label_count[label_name] += 1
                tid2obj[tid][intent.upper()] = 1

for batch in ["support_selected", "denial_selected"]:       
    tweets_with_image = set()
    for image in os.scandir(f"{DATASETS_FOLDER}/images/{batch}_images/"):
        if image.name.endswith(".jpg"):
            try:
                opened_image = Image.open(image.path)
                tid = image.name.split(".")[0]
                tweets_with_image.add(tid)
            except PIL.UnidentifiedImageError:
                pass
    with open(f"{DATASETS_FOLDER}/tweets/{batch}.tsv", "r")  as infile:
        tsv_reader = csv.DictReader(infile, delimiter = "\t")
        for row in tsv_reader:
            tid = row["tid"]
            if tid in tweets_with_image:
                if tid  not in tid2obj:
                    tid2obj[tid] = {
                        "tid": tid,
                        "text": row["text"],
                        "image_file": f"datasets/{DATASET}/images/{batch}_images/{tid}.jpg",
                        "label": 0,
                        "label_name": "BENIGN",
                        "POLAR": 0,
                        "CALL_TO_ACTION": 0,
                        "VIRAL": 0,
                        "SARCASM": 0,
                        "HUMOR": 0
                    }
                    if batch == "support_selected":
                        tid2obj[tid]["label"] = 0
                        tid2obj[tid]["label_name"] = "BENIGN"
                        label_count["BENIGN"] += 1
                    elif batch == "denial_selected":
                        tid2obj[tid]["label"] = 1
                        tid2obj[tid]["label_name"] = "MALICIOUS"
                        label_count["MALICIOUS"] += 1
                
with open(TWEETS_TSV, "w") as outfile:
    tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text", "image_file", "label", "label_name", "POLAR", "CALL_TO_ACTION", "VIRAL", "SARCASM", "HUMOR"])
    tsv_writer.writeheader()
    for _, tweet_obj in tid2obj.items():
        tsv_writer.writerow(tweet_obj)
        outfile.flush()
        
print(label_count)