import csv
import os
import string
import sys

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
        

if __name__ == "__main__":
    for batch_name in ["train", "validate", "test"]:
        print(f"preselect batch {batch_name}...")
        selected_samples = preselect_batch(batch_name)
        os.makedirs(f"{SELECTED_IMAGES_FOLDER}/{batch_name}")
        for dataset in ["covid19", "climate_change", "military_vehicles"]:
            for label in range(6):
                for sample in selected_samples[dataset][label]:
                    os.system(f"cp {DATASET_FOLDER}/images/{batch_name}/{sample['id']}.jpg {SELECTED_IMAGES_FOLDER}/{batch_name}")