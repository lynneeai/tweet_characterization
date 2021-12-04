import argparse
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


from util_scripts.utils import remove_url, is_english

# input files
DENIAL_JSONL = f"{project_root_dir}/datasets/climate/tweets/denial.jsonl"
GENERAL_JSONL = f"{project_root_dir}/datasets/climate/tweets/general.jsonl"
POLAR_JSONL = f"{project_root_dir}/datasets/climate/tweets/polar.jsonl"
SARCASTIC_TWEETS = f"{project_root_dir}/datasets/climate/sarcastic_tweets"
SUPPORT_JSONL = f"{project_root_dir}/datasets/climate/tweets/support.jsonl"

# output files
POLAR_TSV = f"{project_root_dir}/datasets/climate/tweets/polar.tsv"
VIRAL_TSV = f"{project_root_dir}/datasets/climate/tweets/viral.tsv"
SARCASM_TSV = f"{project_root_dir}/datasets/climate/tweets/sarcasm.tsv"
SUPPORT_SELECTED_TSV = f"{project_root_dir}/datasets/climate/tweets/support_selected.tsv"
DENIAL_SELECTED_TSV = f"{project_root_dir}/datasets/climate/tweets/denial_selected.tsv"


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch", required=True, choices=["polar", "viral", "sarcasm", "support", "denial"])
args = parser.parse_args()

# polar tweets
if args.batch == "polar":
    tid_text_tuples = []
    seen_text = set()
    with open(DENIAL_JSONL, "r") as infile:
        for line in infile:
            tweet_obj = json.loads(line.strip())
            tid = tweet_obj["id"]
            text = " ".join(remove_url(tweet_obj["text"]).split())
            if "entities" in tweet_obj and "hashtags" in tweet_obj["entities"]:
                tags = [tag_obj["tag"] for tag_obj in tweet_obj["entities"]["hashtags"]]
                if text not in seen_text and is_english(text) and ("ClimateFraud" in tags or "ClimateHoax" in tags):
                    tid_text_tuples.append((tid, text))
                    seen_text.add(text)
    with open(POLAR_TSV, "w") as outfile:
        tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text"])
        tsv_writer.writeheader()
        for tid, text in tid_text_tuples:
            tsv_writer.writerow({"tid": tid, "text": text})


# viral tweets
if args.batch == "viral":
    tid_text_rtcount_tuples = []
    seen_text = set()
    with open(DENIAL_JSONL, "r") as infile:
        for line in infile:
            tweet_obj = json.loads(line.strip())
            tid = tweet_obj["id"]
            text = " ".join(remove_url(tweet_obj["text"]).split())
            rtcount = tweet_obj["public_metrics"]["retweet_count"]
            if text not in seen_text and is_english(text):
                tid_text_rtcount_tuples.append((tid, text, rtcount))
                seen_text.add(text)
    tid_text_rtcount_tuples.sort(key=lambda x: x[2], reverse=True)
    viral_counts = 0
    for _, _, rtcount in tid_text_rtcount_tuples:
        if rtcount < 50:
            break
        viral_counts += 1
    with open(VIRAL_TSV, "w") as outfile:
        tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text"])
        tsv_writer.writeheader()
        for tid, text, _ in tid_text_rtcount_tuples[:viral_counts+1]:
            tsv_writer.writerow({"tid": tid, "text": text})
        
        
# sarcasm tweets
if args.batch == "sarcasm":
    tid_text_tuples = []
    seen_text = set()
    for tsv_file in os.scandir(SARCASTIC_TWEETS):
        if tsv_file.name.endswith(".tsv"):
            with open(tsv_file.path, "r") as infile:
                tsv_reader = csv.DictReader(infile, delimiter="\t")
                for row in tsv_reader:
                    text = " ".join(remove_url(row["text"]).split())
                    if text not in seen_text and is_english(text):
                        tid_text_tuples.append((row["tid"], text))
                        seen_text.add(text)
    with open(SARCASM_TSV, "w") as outfile:
        tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text"])
        tsv_writer.writeheader()
        for tid, text in tid_text_tuples:
            tsv_writer.writerow({"tid": tid, "text": text})


# support tweets
if args.batch == "support":
    tid_text_rtcount_tuples = []
    seen_text = set()
    with open(SUPPORT_JSONL, "r") as infile:
        for line in infile:
            tweet_obj = json.loads(line.strip())
            tid = tweet_obj["id"]
            text = " ".join(remove_url(tweet_obj["text"]).split())
            rtcount = tweet_obj["public_metrics"]["retweet_count"]
            if text not in seen_text and is_english(text):
                tid_text_rtcount_tuples.append((tid, text, rtcount))
                seen_text.add(text)
    tid_text_rtcount_tuples.sort(key=lambda x: x[2], reverse=True)
    selected_tuples = tid_text_rtcount_tuples[:500]
    print(selected_tuples[-1])
    with open(SUPPORT_SELECTED_TSV, "w") as outfile:
        tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text"])
        tsv_writer.writeheader()
        for tid, text, _ in selected_tuples:
            tsv_writer.writerow({"tid": tid, "text": text})
            

# denial tweets
if args.batch == "denial":
    polar_viral_tids = set()
    for tsv_file in [POLAR_TSV, VIRAL_TSV]:
        with open(tsv_file, "r") as infile:
            tsv_reader = csv.DictReader(infile, delimiter="\t")
            for row in tsv_reader:
                polar_viral_tids.add(row["tid"])
    tid_text_rtcount_tuples = []
    seen_text = set()
    with open(DENIAL_JSONL, "r") as infile:
        for line in infile:
            tweet_obj = json.loads(line.strip())
            tid = tweet_obj["id"]
            text = " ".join(remove_url(tweet_obj["text"]).split())
            rtcount = tweet_obj["public_metrics"]["retweet_count"]
            if tid not in polar_viral_tids and text not in seen_text and is_english(text):
                tid_text_rtcount_tuples.append((tid, text, rtcount))
                seen_text.add(text)
    tid_text_rtcount_tuples.sort(key=lambda x: x[2], reverse=True)
    selected_tuples = tid_text_rtcount_tuples[:10000]
    with open(DENIAL_SELECTED_TSV, "w") as outfile:
        tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text"])
        tsv_writer.writeheader()
        for tid, text, _ in selected_tuples:
            tsv_writer.writerow({"tid": tid, "text": text})