import csv
import json
import os
import sys

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/../.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

DATASETS_FOLDER = f"{project_root_dir}/datasets/eval2"
PROCESSED_DATA_FOLDER = f"{project_root_dir}/processed_data/eval2"
if not os.path.exists(PROCESSED_DATA_FOLDER):
    os.makedirs(PROCESSED_DATA_FOLDER)

# output files
TWEETS_TSV = f"{PROCESSED_DATA_FOLDER}/tweets.tsv"

def read_aom(aom_file):
    text = ""
    image_uri = ""
    with open(aom_file, "r") as infile:
        aom_dict = json.load(infile)
        for item in aom_dict["content"]:
            if item["Type"] == "Paragraph":
                for content_obj in item["Content"]:
                    if content_obj["Type"] == "Text":
                        text += " ".join([t.strip() for t in content_obj["Content"]])
            elif item["Type"] == "Figure":
                image_uri = item["Media"][0]["Uri"].replace("\\", "/")
    return text, image_uri


with open(TWEETS_TSV, "w") as outfile:
    tsv_writer = csv.DictWriter(outfile, fieldnames=["tid", "text", "image_file", "label", "label_name"], delimiter="\t")
    tsv_writer.writeheader()
    for probe_folder in os.scandir(f"{DATASETS_FOLDER}/malicious"):
        text, image_uri = read_aom(f"{probe_folder.path}/aom.json")
        tsv_writer.writerow({
            "tid": probe_folder.name,
            "text": text,
            "image_file": f"datasets/eval2/malicious/{probe_folder.name}/{image_uri}",
            "label": 1,
            "label_name": "MALICIOUS"
        })
    for probe_folder in os.scandir(f"{DATASETS_FOLDER}/benign"):
        text, image_uri = read_aom(f"{probe_folder.path}/aom.json")
        tsv_writer.writerow({
            "tid": probe_folder.name,
            "text": text,
            "image_file": f"datasets/eval2/benign/{probe_folder.name}/{image_uri}",
            "label": 0,
            "label_name": "BENIGN"
        })

