#
# This script will walk through all the tweet id files and
# hydrate them with twarc. The line oriented JSON files will
# be placed right next to each tweet id file. Each tweet id file
# should contain 1 tweet id per line
#
# Note: you will need to install twarc, tqdm, and run twarc configure
# from the command line to tell it your Twitter API keys.
#

import argparse
import gzip
import json
import os

from tqdm import tqdm
from twarc import Twarc
from pathlib import Path


twarc = Twarc()


def main(tweet_id_files):
    for data_f in tweet_id_files:
        hydrate(Path(data_f))
        

def _reader_generator(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def raw_newline_count(fname):
    """
    Counts number of lines in file
    """
    f = open(fname, "rb")
    f_gen = _reader_generator(f.raw.read)
    return sum(buf.count(b"\n") for buf in f_gen)


def hydrate(id_file):
    print("hydrating {}".format(id_file))

    gzip_path = id_file.with_suffix(".jsonl.gz")
    if gzip_path.is_file():
        print("skipping json file already exists: {}".format(gzip_path))
        return

    num_ids = raw_newline_count(id_file)

    with gzip.open(gzip_path, "w") as output:
        with tqdm(total=num_ids) as pbar:
            for tweet in twarc.hydrate(id_file.open()):
                output.write(json.dumps(tweet).encode("utf8") + b"\n")
                pbar.update(1)
                

def hydrate_wrapper(tid_list, file_path):
    temp_txt = file_path.with_suffix(".txt")
    with open(temp_txt, "w") as outfile:
        for tid in tid_list:
            outfile.write(f"{tid}\n")
    hydrate(Path(temp_txt))
    os.system(f"rm -f {temp_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=False, help="Folder containing input txt files of Tweet IDs")
    parser.add_argument("-i", "--input_file", required=False, help="Input txt file of Tweet IDs")
    args = parser.parse_args()
    assert((args.directory != None and args.input_file == None) or (args.directory == None and args.input_file != None))
    
    if args.directory:
        tweet_id_files = []
        for entry in os.scandir(args.data_directory):
            if entry.name.endswith(".txt"):
                tweet_id_files.append(entry.name)
        main(tweet_id_files)
    
    if args.input_file:
        hydrate(Path(args.input_file))
