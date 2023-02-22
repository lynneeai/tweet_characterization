import argparse
import csv
import os
import string
import sys
import random

from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/../.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

# input files
DATASET_FOLDER = f"{project_root_dir}/datasets/fakeddit"
KEYWORDS_FILE = "keywords.txt"

# output files
SELECTED_IMAGES_FOLDER = f"{DATASET_FOLDER}/selected_images"
PROCESSED_DATA_FOLDER = f"{project_root_dir}/processed_data"
for dir in [SELECTED_IMAGES_FOLDER, PROCESSED_DATA_FOLDER]:
    if not os.path.exists(dir):
        os.makedirs(dir)

# constants
WORDNET_LEMMATIZER = WordNetLemmatizer()
PORTER_STEMMER  = PorterStemmer()


def read_keywords():
    keyword_dict = dict()
    with open(KEYWORDS_FILE, "r") as infile:
        for line in infile:
            line = line.strip()
            if line.startswith("#"):
                dataset = line.split("# ")[1]
                keyword_dict[dataset] = set()
            if not line.startswith("#") and line != "":
                keyword = line.lower()
                keyword_lemma = WORDNET_LEMMATIZER.lemmatize(keyword)
                keyword_stem = PORTER_STEMMER.stem(keyword)
                
                keyword_dict[dataset].add(keyword)
                keyword_dict[dataset].add(keyword_lemma)
                keyword_dict[dataset].add(keyword_stem)
                
    return keyword_dict


def remove_punctuation(str):
    translation = {string.punctuation: None}
    str.translate(translation)
    return str


def preselect_batch(batch_name):
    input_tsv = f"{DATASET_FOLDER}/{batch_name}.tsv"
    image_folder = f"{DATASET_FOLDER}/images/{batch_name}"
    keyword_dict = read_keywords()
    
    selected_samples = defaultdict(lambda: defaultdict(list))
    with open(input_tsv, "r") as infile:
        tsv_reader = csv.DictReader(infile, delimiter="\t")
        for row in tsv_reader:
            id, text, domain, label = row["id"], row["clean_title"], row["domain"], int(row["6_way_label"])
            if os.path.exists(f"{image_folder}/{id}.jpg"):
                processed_text = remove_punctuation(text.lower())
                word_list = processed_text.split()
                word_lemma_list, word_stem_list = [], []
                for word in word_list:
                    word_lemma_list.append(WORDNET_LEMMATIZER.lemmatize(word))
                    word_stem_list.append(PORTER_STEMMER.stem(word))
                word_set = set(word_list + word_lemma_list + word_stem_list)
                for dataset in ["covid19", "climate_change", "military_vehicles"]:
                    for word in word_set:
                        if word in keyword_dict[dataset]:
                            selected_samples[dataset][label].append({
                                "id": id,
                                "text": text,
                                "domain": domain,
                                "label": label
                            })
                            break      
    # print stats
    for dataset in ["covid19", "climate_change", "military_vehicles"]:
        print(f"=======Stats for {dataset}=======")
        for label in range(6):
            print(f"Label {label}: {len(selected_samples[dataset][label])}")
    
    for dataset in ["covid19", "climate_change", "military_vehicles"]:
        output_folder = f"{PROCESSED_DATA_FOLDER}/fakeddit_{dataset}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_tsv = f"{output_folder}/{batch_name}.tsv"
        with open(output_tsv, "w") as outfile:
            tsv_writer = csv.DictWriter(outfile, fieldnames=["id", "text", "domain", "label"], delimiter="\t")
            tsv_writer.writeheader()
            for label in range(6):
                for sample in selected_samples[dataset][label]:
                    tsv_writer.writerow(sample)
                    
    return selected_samples


def exclude_true(tsv_file):
    output_tsv_file = f"{tsv_file[:-4]}_notrue.tsv"
    with open(output_tsv_file, "w") as outfile:
        tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["id", "text", "domain", "label"])
        tsv_writer.writeheader()
        with open(tsv_file, "r") as infile:
            tsv_reader = csv.DictReader(infile, delimiter="\t")
            for row in tsv_reader:
                if int(row["label"]) != 0:
                    tsv_writer.writerow({
                        "id": row["id"],
                        "text": row["text"],
                        "domain": row["domain"],
                        "label": int(row["label"]) - 1
                    })
                    
                    
def select_binary(tsv_file):
    output_tsv_file = f"{tsv_file[:-4]}_binary.tsv"
    
    label_rows_dict = defaultdict(list)
    with open(tsv_file, "r") as infile:
        tsv_reader = csv.DictReader(infile, delimiter="\t")
        for row in tsv_reader:
            label_rows_dict[int(row["label"])].append(row)

    malicious_list = label_rows_dict[2] + label_rows_dict[4] + label_rows_dict[5]
    benign_list = label_rows_dict[1] + label_rows_dict[3]
    
    true_list = label_rows_dict[0]
    random.shuffle(true_list)
    benign_list += true_list[:(len(malicious_list) - len(benign_list))]
    
    with open(output_tsv_file, "w") as outfile:
        tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["id", "text", "domain", "label"])
        tsv_writer.writeheader()
        for row in benign_list:
            tsv_writer.writerow({
                "id": row["id"],
                "text": row["text"],
                "domain": row["domain"],
                "label": 0
            })
        for row in malicious_list:
            tsv_writer.writerow({
                "id": row["id"],
                "text": row["text"],
                "domain": row["domain"],
                "label": 1
            })


def merge_samples_for_calibration():
    if not os.path.exists(f"{PROCESSED_DATA_FOLDER}/fakeddit_calibrated"):
        os.makedirs(f"{PROCESSED_DATA_FOLDER}/fakeddit_calibrated")
        
    for batch_name in ["train", "validate", "test"]:
        batch_tsv_file = f"{PROCESSED_DATA_FOLDER}/fakeddit_calibrated/{batch_name}_binary.tsv"
        batch_outfile = open(batch_tsv_file, "w")
        tsv_writer = csv.DictWriter(batch_outfile, delimiter="\t", fieldnames=["id", "text", "domain", "label"])
        tsv_writer.writeheader()
        id_set = set()
        for dataset in ["covid19", "climate_change", "military_vehicles"]:
            tsv_file = f"{PROCESSED_DATA_FOLDER}/fakeddit_{dataset}/{batch_name}_binary.tsv"
            with open(tsv_file, "r") as infile:
                tsv_reader = csv.DictReader(infile, delimiter="\t")
                for row in tsv_reader:
                    if row["id"] not in id_set:
                        tsv_writer.writerow(dict(row))
                        id_set.add(row["id"])
        batch_outfile.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", default="default", choices=["default", "notrue", "binary", "calibrate"])
    args = parser.parse_args()
    
    if args.task == "notrue":
        for dataset in ["covid19", "climate_change", "military_vehicles"]:
            for batch_name in ["train", "validate", "test"]:
                tsv_file = f"{PROCESSED_DATA_FOLDER}/fakeddit_{dataset}/{batch_name}.tsv"
                exclude_true(tsv_file)
                
    elif args.task == "binary":
        for dataset in ["covid19", "climate_change", "military_vehicles"]:
            for batch_name in ["train", "validate", "test"]:
                tsv_file = f"{PROCESSED_DATA_FOLDER}/fakeddit_{dataset}/{batch_name}.tsv"
                select_binary(tsv_file)
    
    elif args.task == "calibrate":
        merge_samples_for_calibration()
    
    else:
        for batch_name in ["train", "validate", "test"]:
            print(f"preselect batch {batch_name}...")
            selected_samples = preselect_batch(batch_name)
            os.makedirs(f"{SELECTED_IMAGES_FOLDER}/{batch_name}")
            for dataset in ["covid19", "climate_change", "military_vehicles"]:
                for label in range(6):
                    for sample in selected_samples[dataset][label]:
                        os.system(f"cp {DATASET_FOLDER}/images/{batch_name}/{sample['id']}.jpg {SELECTED_IMAGES_FOLDER}/{batch_name}")