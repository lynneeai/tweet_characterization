import csv
import os
import requests
import sys
from ratelimit import limits, sleep_and_retry
from pprint import pprint

from requests.models import Response

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/../.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from select_tweets import remove_url
from util_scripts.api_keys import TWITTER_API_KEYS
from util_scripts.twitter_auth import BearerTokenAuth

BEARER_TOKEN = BearerTokenAuth(TWITTER_API_KEYS.consumer_key, TWITTER_API_KEYS.consumer_secret)

print("Enter query:")
QUERY = input()
print("Enter output file:")
OUTPUT_FILE = input()
print("Enter start time (or leave blank for default):")
START_TIME = input()
START_TIME = START_TIME if START_TIME != "" else "2020-10-01T00:00:00Z"
print("Enter end time (or leave blank for default):")
END_TIME = input()
END_TIME = END_TIME if END_TIME != "" else "2021-10-31T23:59:59Z"

QUERY += " -is:retweet -is:reply -is:quote has:images lang:en"

print(f"Query: {QUERY}")
print(f"Output file: {OUTPUT_FILE}")
print(f"Start time: {START_TIME} ; End time: {END_TIME}")
        
@sleep_and_retry
@limits(calls=300, period=900)
@sleep_and_retry
@limits(calls=1, period=1)
def search(next_token):
    endpoint_url = "https://api.twitter.com/2/tweets/search/all"
    
    params = {"query": QUERY, "start_time": START_TIME, "end_time": END_TIME, "max_results": 500}
    if next_token:
        params["next_token"] = next_token
    
    headers = {"Accept-Encoding": "gzip"}
    response = requests.get(endpoint_url, auth=BEARER_TOKEN, headers=headers, params=params)
    
    if response.status_code > 201:
        print(f"{response.status_code}: {response.text}")
        raise Exception(f"{response.status_code}: {response.text}")
    
    try:
        r_json = response.json()
        tweet_dicts = r_json["data"]
        next_token = r_json["meta"]["next_token"] if "next_token" in r_json["meta"] else "DOES NOT EXISTS!"
    except Exception as e:
        print(e)
        raise Exception(e)
    
    return tweet_dicts, next_token

def search_wrapper():
    output_file = f"{project_root_dir}/datasets/covid19/tweets/{OUTPUT_FILE}"
    
    total_tweets = 0
    seen_text = set()
    with open(output_file, "w") as outfile:
        tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text"])
        tsv_writer.writeheader()
        next_token = None
        while next_token != "DOES NOT EXISTS!":
            try:
                tweet_dicts, next_token = search(next_token)
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
    search_wrapper()