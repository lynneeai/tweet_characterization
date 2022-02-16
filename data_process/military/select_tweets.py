import argparse
import csv
import os
import re
import json
import sys
import pycld2 as cld2
from collections import defaultdict

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/../.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

TWEETS_FOLDER = f"{project_root_dir}/datasets/military/tweets"

hashtag_count = defaultdict(int)
with open(f"{TWEETS_FOLDER}/afghanistan.tsv", "r") as infile:
    tsv_reader = csv.DictReader(infile, delimiter="\t")
    