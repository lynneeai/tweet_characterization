import csv
import os
import sys

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/../.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

print("Enter datasets separated by +, e.g. covid19+climate:")
DATASETS = input()

OUTPUT_FOLDER = f"{project_root_dir}/processed_data/{DATASETS}"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# input files
INPUT_TSVS = [f"{project_root_dir}/processed_data/{dataset}/tweets.tsv" for dataset in DATASETS.split("+")]

# output files
OUTPUT_TSV = f"{OUTPUT_FOLDER}/tweets.tsv"


with open(OUTPUT_TSV, "w") as outfile:
    tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text", "image_file", "label", "label_name", "POLAR", "CALL_TO_ACTION", "VIRAL", "SARCASM", "HUMOR"])
    tsv_writer.writeheader()
    for tsv_file in INPUT_TSVS:
        with open(tsv_file, "r") as infile:
            tsv_reader = csv.DictReader(infile, delimiter="\t")
            for row in tsv_reader:
                tsv_writer.writerow(dict(row))
                outfile.flush()