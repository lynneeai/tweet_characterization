import csv
import os
import sys
import random
from collections import defaultdict

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/../.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from util_scripts.utils import count_file_lines, remove_url

# input files
NEG_TSV = f"{project_root_dir}/datasets/military/negative/negative_processed_with_images.tsv"
NEG_IMAGE_FOLDER = "datasets/military/negative/images"
CTA_TSV = f"{project_root_dir}/datasets/military/cta/cta_processed_with_images.tsv"
CTA_IMAGE_FOLDER = "datasets/military/cta/images"
DE_TSV = f"{project_root_dir}/datasets/military/de/de_processed_with_images.tsv"
DE_IMAGE_FOLDER = "datasets/military/de/images"

# output folders
CTA_FOLDER = f"{project_root_dir}/processed_data/military_cta"
DE_FOLDER = f"{project_root_dir}/processed_data/military_de"
if not os.path.exists(CTA_FOLDER):
    os.makedirs(CTA_FOLDER)
if not os.path.exists(DE_FOLDER):
    os.makedirs(DE_FOLDER)
    

"""
READ NEGATIVE TWEETS
"""
with open(NEG_TSV, "r") as infile:
    tsv_reader = csv.DictReader(infile, delimiter="\t")
    neg_rows = [dict(row) for row in tsv_reader]


"""
PROCESS CTA TWEETS
"""
random.shuffle(neg_rows)
with open(CTA_TSV, "r") as infile:
    tsv_reader = csv.DictReader(infile, delimiter="\t")
    cta_rows = [dict(row) for row in tsv_reader]
with open(f"{CTA_FOLDER}/tweets.tsv", "w") as outfile:
    tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text", "image_file", "label"])
    tsv_writer.writeheader()
    for row in cta_rows:
        for image_file in row["image_files"].split(","):
            tsv_writer.writerow({
                "tid": row["tid"],
                "text": remove_url(row["text"]),
                "image_file": f"{CTA_IMAGE_FOLDER}/{image_file}",
                "label": 1
            })
            outfile.flush()
    for row in neg_rows[:len(cta_rows)]:
        for image_file in row["image_files"].split(","):
            tsv_writer.writerow({
                "tid": row["tid"],
                "text": remove_url(row["text"]),
                "image_file": f"{NEG_IMAGE_FOLDER}/{image_file}",
                "label": 0
            })


"""
PROCESS DE TWEETS
"""
random.shuffle(neg_rows)
with open(DE_TSV, "r") as infile:
    tsv_reader = csv.DictReader(infile, delimiter="\t")
    de_rows = [dict(row) for row in tsv_reader]
with open(f"{DE_FOLDER}/tweets.tsv", "w") as outfile:
    tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text", "image_file", "label"])
    tsv_writer.writeheader()
    for row in de_rows:
        for image_file in row["image_files"].split(","):
            tsv_writer.writerow({
                "tid": row["tid"],
                "text": remove_url(row["text"]),
                "image_file": f"{DE_IMAGE_FOLDER}/{image_file}",
                "label": 1
            })
            outfile.flush()
    for row in neg_rows[:len(de_rows)]:
        for image_file in row["image_files"].split(","):
            tsv_writer.writerow({
                "tid": row["tid"],
                "text": remove_url(row["text"]),
                "image_file": f"{NEG_IMAGE_FOLDER}/{image_file}",
                "label": 0
            })