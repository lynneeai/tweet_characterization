import argparse
import csv
import os
import sys
import requests
import json
import pytz
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from ratelimit import limits, sleep_and_retry

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/../.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from util_scripts.api_keys import TWITTER_API_KEYS
from util_scripts.twitter_auth import BearerTokenAuth
from util_scripts.utils import remove_url

# Twitter authentication
twitter_api_keys = TWITTER_API_KEYS()
consumer_key = twitter_api_keys.consumer_key
consumer_secret = twitter_api_keys.consumer_secret
BEARER_TOKEN = BearerTokenAuth(consumer_key, consumer_secret)


@sleep_and_retry
@limits(calls=900, period=900)
def get_account_timeline(uid, start_time=None, end_time=None, next_token=None):
    endpoint_url = f"https://api.twitter.com/2/users/{uid}/tweets"
    headers = {"Accept-Encoding": "gzip"}
    if next_token:
        params = {"exclude": "retweets", "tweet.fields": "id,author_id,created_at,entities,referenced_tweets,lang,public_metrics,text", "user.fields": "created_at,description,entities,id,location,name,pinned_tweet_id,public_metrics,username,protected,verified", "max_results": 100, "pagination_token": next_token}
    else:
        params = {"exclude": "retweets", "tweet.fields": "id,author_id,created_at,entities,referenced_tweets,lang,public_metrics,text", "user.fields": "created_at,description,entities,id,location,name,pinned_tweet_id,public_metrics,username,protected,verified", "max_results": 100}
    
    if start_time is not None and end_time is not None:
        params["start_time"] = start_time
        params["end_time"] = end_time
    elif start_time is not None:
        params["start_time"] = start_time
    elif end_time is not None:
        params["end_time"] = end_time
    
    response = requests.get(endpoint_url, auth=BEARER_TOKEN, headers=headers, params=params)
    if response.status_code > 201:
        print(f"{response.status_code}: {response.text}")
        raise Exception(response.status_code)
    
    r_json = response.json()
    next_token = r_json["meta"]["next_token"] if "next_token" in r_json["meta"] else "Does not exist!"
    
    if r_json["meta"]["result_count"] == 0:
        return {"data": [], "next_token": "Does not exist!"}
    
    return {"data": r_json["data"], "next_token": next_token}


@sleep_and_retry
@limits(calls=300, period=900)
def get_user_id(username):
    endpoint_url = f"https://api.twitter.com/2/users/by/username/{username}"
    headers = {"Accept-Encoding": "gzip"}
    params = {"user.fields": "created_at"}
    response = requests.get(endpoint_url, auth=BEARER_TOKEN, headers=headers, params=params)
    if response.status_code > 201:
        print(f"{response.status_code}: {response.text}")
        raise Exception(response.status_code)
    r_obj = response.json()
    return r_obj["data"]["id"]


def get_user_tweets(uid, output_file, start_time=None, end_time=None):
    tid_outfile = open(Path(output_file).with_suffix(".txt"), "w")
    with open(output_file, "w") as outfile:
        csv_writer = csv.DictWriter(outfile, fieldnames=["tid", "uid", "created_at", "text", "urls", "mentions", "retweet_count"], delimiter="\t")
        csv_writer.writeheader()
        next_token = None
        total_tweets = 0
        while next_token != "Does not exist!":
            obj = get_account_timeline(uid, start_time, end_time, next_token=next_token)
            tweet_obj_list, next_token = obj["data"], obj["next_token"]
            for tweet_obj in tweet_obj_list:
                if "referenced_tweets" not in tweet_obj and tweet_obj["lang"] == "en":
                    tid = tweet_obj["id"]
                    created_at = tweet_obj["created_at"]
                    text = " ".join(tweet_obj["text"].split())
                    
                    urls = []
                    if "entities" in tweet_obj and "urls" in tweet_obj["entities"]:
                        for url_obj in tweet_obj["entities"]["urls"]:
                            urls.append(url_obj["expanded_url"])
                    urls = ",".join(urls)
                    
                    mentions = []
                    if "entities" in tweet_obj and "mentions" in tweet_obj["entities"]:
                        for mention_obj in tweet_obj["entities"]["mentions"]:
                            mentions.append(f"@{mention_obj['username']}")
                    mentions = ",".join(mentions)
                        
                    rt_count = tweet_obj["public_metrics"]["retweet_count"]
                    csv_writer.writerow({"tid": tid, "uid": uid, "created_at": created_at, "text": text, "urls": urls, "mentions": mentions, "retweet_count": rt_count})
                    outfile.flush()
                    
                    tid_outfile.write(f"{tid}\n")
                    tid_outfile.flush()
            total_tweets += len(tweet_obj_list)
            print(f"\r{total_tweets} tweets processed! Next token: {next_token}...", end="", flush=True)
        tid_outfile.close()
        print()
                    
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--username", type=str, required=False)
    parser.add_argument("-i", "--input_file", type=str, required=False, help="A txt file of all usernames")
    parser.add_argument("-o", "--output_folder", type=str, required=True)
    args = parser.parse_args()
    
    assert(args.username is not None or args.input_file is not None)
    
    OUTPUT_FOLDER = args.output_folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    if args.username is not None:
        try:
            uid = get_user_id(args.username)
            # start_time = "2010-11-06T00:00:00.000Z"
            # end_time = generate(datetime.utcnow().replace(tzinfo=pytz.utc))
            output_file = f"{OUTPUT_FOLDER}/{args.username}.tsv"
            get_user_tweets(uid, output_file)
        except Exception as e:
            if e == "404":
                pass
            else:
                raise(e)
    
    if args.input_file is not None:
        username_list = []
        with open(args.input_file, "r") as infile:
            for line in infile:
                username = line.strip().split("@")[1]
                username_list.append(username)
        for username in tqdm(username_list):
            try:
                uid = get_user_id(username)
                output_file = f"{OUTPUT_FOLDER}/{username}.tsv"
                get_user_tweets(uid, output_file)
                print(f"{username} done!")
            except Exception as e:
                if e == "404":
                    continue
                else:
                    raise(e)
            