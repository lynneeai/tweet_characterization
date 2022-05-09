import csv
from itertools import count
import os
from collections import defaultdict


def get_popular_hashtags(input_tsv, first_k, output_txt):
    hashtag_count = defaultdict(int)
    with open(input_tsv, "r") as infile:
        tsv_reader = csv.DictReader(infile, delimiter="\t")
        for row in tsv_reader:
            tokens = row["text"].split()
            for token in tokens:
                if token.startswith("#"):
                    hashtag_count[token.lower()] += 1
    hashtags = sorted([(tag, count) for tag, count in hashtag_count.items()], key=lambda x: x[1], reverse=True)
    
    with open(output_txt, "w") as outfile:
        for tag, count in hashtags[:first_k]:
            outfile.write(f"{tag}\t{count}\n")

get_popular_hashtags("../../../datasets/military/zelenskywarcriminal/zelenskywarcriminal.tsv", 100, "zelenskywarcriminal_popular_hashtags.txt")