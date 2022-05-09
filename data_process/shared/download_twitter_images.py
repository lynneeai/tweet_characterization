import argparse
import csv
import json
import os
import socket
import urllib.request
import http.client
from tqdm import tqdm
from urllib.error import HTTPError, URLError
from collections import defaultdict
from pathlib import Path

from hydrate import hydrate, hydrate_wrapper


def download_images(input_jsonl_file, output_image_folder):
    print("Downloading images...")
    socket.setdefaulttimeout(30)
    tid_text_url_imageurl_pairs = []
    with open(input_jsonl_file, "r") as infile:
        for line in infile:
            tweet_obj = json.loads(line.strip())
            tid = tweet_obj["id"]
            text = " ".join(tweet_obj["full_text"].split())
            img_counter = 0
            if "extended_entities" in tweet_obj:
                if "media" in tweet_obj["extended_entities"]:
                    for media_obj in tweet_obj["extended_entities"]["media"]:
                        if media_obj["type"] == "photo":
                            size = "large" if "large" in media_obj["sizes"] else "medium"
                            image_url = f"{media_obj['media_url_https']}:{size}"
                            tweet_url = media_obj['expanded_url'].split("/photo")[0]       
                            if not os.path.exists(f"{output_image_folder}/{tid}_{img_counter}.jpg"):
                                tid_text_url_imageurl_pairs.append((tid, text, tweet_url, image_url, img_counter))
                            img_counter += 1
    
    tid2text_url = {}
    tid2images = defaultdict(list) 
    for tid, text, tweet_url, image_url, img_counter in tqdm(tid_text_url_imageurl_pairs):
        try:
            urllib.request.urlretrieve(image_url, f"{output_image_folder}/{tid}_{img_counter}.jpg")
            tid2text_url[str(tid)] = (text, tweet_url)
            tid2images[str(tid)].append(f"{tid}_{img_counter}.jpg")
        except HTTPError as err:
            print(err)
            pass
        except URLError as err:
            print(err)
            pass
        except socket.timeout as err:
            print(err)
            pass
        except http.client.RemoteDisconnected as err:
            print(err)
            pass
    print("Done!")
    return tid2text_url, tid2images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True, help="Input txt file of Tweet IDs, tsv file with tid as a column, or a hydtrated jsonl file.")
    parser.add_argument("-o", "--output_folder", required=True, help="Output image folder")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    hydrated_jsonlgz_file = Path(args.input_file).with_suffix(".jsonl.gz")
    hydrated_jsonl_file = Path(args.input_file).with_suffix(".jsonl")
    input_filename = Path(args.input_file).name.split(".")[0]
    
    if args.input_file.endswith(".txt"):
        input_txt_file = Path(args.input_file)
        os.system(f"rm -f {hydrated_jsonlgz_file}")
        hydrate(input_txt_file)
        os.system(f"gunzip {hydrated_jsonlgz_file}")
        
    elif args.input_file.endswith(".tsv"):
        tids = []
        with open(args.input_file, "r") as infile:
            tsv_reader = csv.DictReader(infile, delimiter="\t")
            for row in tsv_reader:
                tids.append(row["tid"])
        os.system(f"rm -f {hydrated_jsonlgz_file}")
        hydrate_wrapper(tids, Path(args.input_file))
        os.system(f"gunzip {hydrated_jsonlgz_file}")
    
    elif args.input_file.endswith(".jsonl"):
        hydrated_jsonl_file = args.input_file
    
    tid2text_url, tid2images = download_images(hydrated_jsonl_file, args.output_folder)
    
    if not os.path.exists(f"{args.output_folder}/../{input_filename}_with_images.tsv"):
        with open(f"{args.output_folder}/../{input_filename}_with_images.tsv", "w") as outfile:
            tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text", "image_files", "tweet_url"])
            tsv_writer.writeheader()
            for tid, images in tid2images.items():
                tsv_writer.writerow({"tid": tid, "text": tid2text_url[tid][0], "image_files": ",".join(images), "tweet_url": tid2text_url[tid][1]})
    else:
        with open(f"{args.output_folder}/../{input_filename}_with_images.tsv", "a") as outfile:
            tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text", "image_files", "tweet_url"])
            for tid, images in tid2images.items():
                tsv_writer.writerow({"tid": tid, "text": tid2text_url[tid][0], "image_files": ",".join(images), "tweet_url": tid2text_url[tid][1]})