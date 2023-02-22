import argparse
import csv
import os
import math
import sys

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/../.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from data_process.covid19.select_tweets import remove_url, is_english

DATASETS_FOLDER = f"{project_root_dir}/datasets/covid19"
PROCESSED_DATA_FOLDER = f"{project_root_dir}/processed_data/covid19"

# input files
FB_CSV = f"{DATASETS_FOLDER}/fb_posts.csv"

# output files
HUMOR_TSV = f"{DATASETS_FOLDER}/tweets/humor.tsv"


fb_posts = []
seen_text = set()
with open(FB_CSV, "r") as infile:
    csv_reader = csv.DictReader(infile)
    for row in csv_reader:
        uid, post_url, type = row["Facebook Id"], row["URL"], row["Type"]
        if type == "Photo":
            post_id = post_url.split("/")[-1]
            text = " ".join(remove_url(row["Message"]).split())
            if text not in seen_text:
                image_url = row["Link"]
                haha = int(row["Haha"])
                total_reaction = int(row["Love"]) + int(row["Wow"]) + int(row["Haha"]) + int(row["Sad"]) + int(row["Angry"]) + int(row["Care"])
                humor_score = (haha / total_reaction) * math.tanh(total_reaction / 50)
                fb_posts.append((post_id, text, image_url, humor_score))
                seen_text.add(text)

fb_posts.sort(key=lambda x: x[-1], reverse=True)

with open(HUMOR_TSV, "w") as outfile:
    tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text", "image_url", "humor_score"])
    tsv_writer.writeheader()
    for post in fb_posts[:1500]:
        tsv_writer.writerow({
            "tid": post[0],
            "text": post[1],
            "image_url": post[2],
            "humor_score": post[3]
        })
        outfile.flush()