import csv
import os
import re
import json
import sys
import pycld2 as cld2
from collections import defaultdict
from pprint import pprint

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/../.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

DATASETS_FOLDER = f"{project_root_dir}/datasets/covid19"
PROCESSED_DATA_FOLDER = f"{project_root_dir}/processed_data/covid19"

# output files
TWEETS_TSV = f"{PROCESSED_DATA_FOLDER}/tweets.tsv"


# constants
LABEL_NAME_DICT = {"BENIGN": 0, "MALICIOUS": 1}


tid2obj = {}
for intent in ["polar", "call_to_action", "viral", "sarcasm"]:
    tweets_with_image = set()
    for image in os.scandir(f"{DATASETS_FOLDER}/{intent}_images/"):
        if image.name.endswith(".jpg"):
            tid = image.name.split(".")[0]
            tweets_with_image.add(tid)
    
    with open(f"{DATASETS_FOLDER}/{intent}.tsv", "r") as infile:
        tsv_reader = csv.DictReader(infile, delimiter = "\t")
        for row in tsv_reader:
            tid = row["tid"]
            if tid in tweets_with_image:
                label_name = "BENIGN" if intent == "sarcasm" else "MALICIOUS"
                if tid  not in tid2obj:
                    tid2obj[tid] = {
                        "tid": tid,
                        "text": row["text"],
                        "image_file": f"{DATASETS_FOLDER}/{intent}_images/{tid}.jpg",
                        "label": LABEL_NAME_DICT[label_name],
                        "label_name": label_name,
                        "POLAR": 0,
                        "CALL_TO_ACTION": 0,
                        "VIRAL": 0,
                        "SARCASM": 0
                    }
                tid2obj[tid][intent.upper()] = 1
                
with open(TWEETS_TSV, "w") as outfile:
    tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text", "image_file", "label", "label_name", "POLAR", "CALL_TO_ACTION", "VIRAL", "SARCASM"])
    tsv_writer.writeheader()
    for _, tweet_obj in tid2obj.items():
        tsv_writer.writerow(tweet_obj)
        outfile.flush()