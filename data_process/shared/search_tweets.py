import argparse
import csv
import os
import json
import requests
import sys
import math
import time
from tqdm import tqdm
from pathlib import Path

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/../.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from util_scripts.api_keys import TWITTER_API_KEYS
from util_scripts.twitter_auth import BearerTokenAuth
from util_scripts.utils import boolean_string, remove_url
from hydrate import hydrate
from download_twitter_images import download_images

        
def search(query, start_time, end_time, bearer_token, next_token):
    
    time.sleep(1) # twitter API rate limits 1 request per 1s
    
    endpoint_url = "https://api.twitter.com/2/tweets/search/all"
    
    params = {"query": query, "start_time": start_time, "end_time": end_time, "max_results": 500}
    if next_token:
        params["next_token"] = next_token
    
    headers = {"Accept-Encoding": "gzip"}
    response = requests.get(endpoint_url, auth=bearer_token, headers=headers, params=params)
    
    if response.status_code > 201:
        print(f"{response.status_code}: {response.text}")
        raise Exception(f"{response.status_code}: {response.text}")
    
    try:
        # handels twitter API rate limit 300 pulls per 900s
        print(f"Remaining pulls: {response.headers['x-rate-limit-remaining']}", " ", end="\r")
        if int(response.headers['x-rate-limit-remaining']) <= 0:
            print(f"Reached Twitter API rate limit!")
            sleep_time = float(response.headers["x-rate-limit-reset"]) - time.time()
            
            print(f"Sleeping {math.ceil(sleep_time)}s...")
            for i in tqdm(range(math.ceil(sleep_time))):
                time.sleep(1)
        
        r_json = response.json()
        tweet_dicts = r_json["data"]
        next_token = r_json["meta"]["next_token"] if "next_token" in r_json["meta"] else "DOES NOT EXISTS!"
    except Exception as e:
        print(e)
        raise Exception(e)
    
    return tweet_dicts, next_token

def search_wrapper(query, start_time, end_time, bearer_token, output_tsv):
    total_tweets = 0
    seen_text = set()
    with open(output_tsv, "w") as outfile:
        tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text"])
        tsv_writer.writeheader()
        next_token = None
        while next_token != "DOES NOT EXISTS!":
            try:
                tweet_dicts, next_token = search(query, start_time, end_time, bearer_token, next_token)
                for tweet_obj in tweet_dicts:
                    text = " ".join(remove_url(tweet_obj["text"]).split())
                    if text not in seen_text:
                        tsv_writer.writerow({
                            "tid": tweet_obj["id"],
                            "text": text
                        })
                        outfile.flush()
                        total_tweets += 1
                        seen_text.add(text)
                print(f"{total_tweets} scraped!", end="\r")
            except:
                break
    print(f"{total_tweets} scraped!")
    print("Done!")


if __name__ == "__main__":
    BEARER_TOKEN = BearerTokenAuth(TWITTER_API_KEYS.consumer_key, TWITTER_API_KEYS.consumer_secret)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config .json file")
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, "r") as infile:
            config_dict = json.load(infile)
            QUERY = config_dict["query"]
            START_TIME = config_dict["start_time"]
            END_TIME = config_dict["end_time"]
            OUTPUT_TSV = config_dict["output_tsv"]
            DOWNLOAD_IMAGE = boolean_string(config_dict["download_image"])
            if DOWNLOAD_IMAGE:
                IMAGE_FOLDER = config_dict["image_folder"]
                if not os.path.exists(IMAGE_FOLDER):
                    os.makedirs(IMAGE_FOLDER)
    else: 
        print("Enter query:")
        QUERY = input()

        print("Enter start time or leave blank for default (2020-10-01T00:00:00Z):")
        START_TIME = input()
        START_TIME = START_TIME if START_TIME != "" else "2020-10-01T00:00:00Z"
        print("Enter end time or leave blank for default (2021-10-31T23:59:59Z):")
        END_TIME = input()
        END_TIME = END_TIME if END_TIME != "" else "2021-10-31T23:59:59Z"
        
        print("Enter output tsv file:")
        OUTPUT_TSV = input()

        print("Do you want to download images?")
        DOWNLOAD_IMAGE = boolean_string(input())
        if DOWNLOAD_IMAGE:
            print("Enter output image folder:")
            IMAGE_FOLDER = input()
            if not os.path.exists(IMAGE_FOLDER):
                os.makedirs(IMAGE_FOLDER)

    output_folder = "/".join(OUTPUT_TSV.split("/")[:-1])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    QUERY += " -is:retweet -is:reply -is:quote has:images lang:en"

    print(f"Query: {QUERY}")
    print(f"Output tsv file: {OUTPUT_TSV}")
    print(f"Start time: {START_TIME} ; End time: {END_TIME}")
    if DOWNLOAD_IMAGE:
        print(f"Output image folder: {IMAGE_FOLDER}")
    
    search_wrapper(QUERY, START_TIME, END_TIME, BEARER_TOKEN, OUTPUT_TSV)
    
    if DOWNLOAD_IMAGE:
        with open("temp_tids.txt", "w") as outfile:
            with open(OUTPUT_TSV, "r") as infile:
                tsv_reader = csv.DictReader(infile, delimiter="\t")
                for row in tsv_reader:
                    outfile.write(f"{row['tid']}\n")
        hydrate(Path("temp_tids.txt"))
        os.system("gunzip temp_tids.jsonl.gz")
        tid2images = download_images("temp_tids.jsonl", IMAGE_FOLDER)
        os.system("rm temp_tids.txt temp_tids.jsonl")
        # write image file names to tsv
        with open(OUTPUT_TSV, "r") as infile:
            tsv_reader = csv.DictReader(infile, delimiter="\t")
            with open("temp.tsv", "w") as outfile:
                tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text", "image_files"])
                tsv_writer.writeheader()
                for row in tsv_reader:
                    tsv_writer.writerow({
                        "tid": row["tid"],
                        "text": row["text"],
                        "image_files": ",".join(tid2images[row["tid"]])
                    })
        os.system(f"mv temp.tsv {OUTPUT_TSV}")