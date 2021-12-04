import csv
import os
import json
import sys
import random

from pprint import pprint
from collections import defaultdict

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/../.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

# input files
DENIAL_JSONL = f"{project_root_dir}/datasets/climate/tweets/denial.jsonl"
SUPPORT_ACT_JSONL = f"{project_root_dir}/datasets/climate/climate_belief_action.jsonl"
SUPPORT_VIRAL_JSONL = f"{project_root_dir}/datasets/climate/climate_belief_viral.jsonl"

# output files
DENIAL_HASHTAGS = "denial_hashtags.txt"
SUPPORT_HASHTAGS = "support_hashtags.txt"


# denial hashtags
denial_hashtags = defaultdict(int)
with open(DENIAL_JSONL, "r") as infile:
    for line in infile:
        tweet_obj = json.loads(line)
        if "entities" in tweet_obj and "hashtags" in tweet_obj["entities"]:
            for tag_obj in tweet_obj["entities"]["hashtags"]:
                denial_hashtags[tag_obj["tag"]] += 1
denial_hashtags = sorted([(tag, count) for tag, count in denial_hashtags.items()], key=lambda x: x[1], reverse=True)
with open(DENIAL_HASHTAGS, "w") as outfile:
    for tag, count in denial_hashtags:
        outfile.write(f"{tag}\t{count}\n")
        outfile.flush()
        
# support hashtags
support_hashtags = defaultdict(int)
for support_file in [SUPPORT_ACT_JSONL, SUPPORT_VIRAL_JSONL]:
    with open(support_file, "r") as infile:
        for line in infile:
            tweet_obj = json.loads(line)["data"]
            if "entities" in tweet_obj and "hashtags" in tweet_obj["entities"]:
                for tag_obj in tweet_obj["entities"]["hashtags"]:
                    support_hashtags[tag_obj["tag"]] += 1
support_hashtags = sorted([(tag, count) for tag, count in support_hashtags.items()], key=lambda x: x[1], reverse=True)
with open(SUPPORT_HASHTAGS, "w") as outfile:
    for tag, count in support_hashtags:
        outfile.write(f"{tag}\t{count}\n")
        outfile.flush()