import argparse
import csv
import os
import re
import json
import sys
import pycld2 as cld2

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/../.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

# input files
JSONL_FOLDER = f"{project_root_dir}/datasets/covid19/avax-tweets-hydrated"
MONTHS = ["2020-10", "2020-11", "2020-12", "2021-01", "2021-02", "2021-03", "2021-04"]
POLAR_HASHTAGS_FILE = "./polar_hashtags.txt"
CTA_HASHTAGS_FILE = "./cta_hashtags.txt"
SARCASM_FOLDER = f"{project_root_dir}/datasets/covid19/sarcasm_tweets"

# output files
VIRAL_FILE = f"{project_root_dir}/datasets/covid19/viral.tsv"
POLAR_FILE = f"{project_root_dir}/datasets/covid19/polar.tsv"
CTA_FILE = f"{project_root_dir}/datasets/covid19/call_to_action.tsv"
SARCASM_FILE = f"{project_root_dir}/datasets/covid19/sarcasm.tsv"


def remove_url(txt):
    return re.sub(r'https?://\S+', '', txt, flags=re.MULTILINE)


def is_english(text):
    try:
        _, _, _, detected_language = cld2.detect(text, returnVectors=True)
        all_english = True
        for lang in detected_language:
            lang = lang[-1]
            if lang != "en":
                all_english = False
                break
    except:
        print(text)
        return False
    return all_english


def is_retweet(tweet_obj):
    if "referenced_tweets" in tweet_obj:
        for ref_tweet_obj in tweet_obj["referenced_tweets"]:
            if ref_tweet_obj["type"] == "retweeted":
                return True
    return False


def get_image_urls(tweet_obj):
    image_urls = []
    if "entities" in tweet_obj and "urls" in tweet_obj["entities"]:
        for url_obj in tweet_obj["entities"]["urls"]:
            if url_obj["display_url"].startswith("pic.twitter.com"):
                image_urls.append(url_obj["expanded_url"])
    image_urls = ",".join(image_urls)
    return image_urls


def select_tweets_by_hashtags(tags):
    selected_tweets =  []
    for month in MONTHS:
        with open(f"{JSONL_FOLDER}/{month}.jsonl", "r") as infile:
            for line in infile:
                tweet_obj = json.loads(line.strip())
                if "entities" in tweet_obj and "hashtags" in tweet_obj["entities"]:
                    text = " ".join(remove_url(tweet_obj["text"]).split())
                    hashtags = []
                    for tag_obj in tweet_obj["entities"]["hashtags"]:
                        hashtags.append(tag_obj["tag"])
                    image_urls = get_image_urls(tweet_obj)
                    if not is_retweet(tweet_obj) and image_urls and is_english(text):
                        for tag in tags:
                            if tag in hashtags:
                                selected_tweets.append(
                                    {
                                        "tid": tweet_obj["id"],
                                        "text": text,
                                        "image_urls": image_urls
                                    }
                                )
                                break
    return selected_tweets
    

def select_tweets_by_keywords(keywords):
    selected_tweets = []
    seen_text = set()
    for month in MONTHS:
        with open(f"{JSONL_FOLDER}/{month}.jsonl", "r") as infile:
            for line in infile:
                tweet_obj = json.loads(line.strip())
                text = " ".join(remove_url(tweet_obj["text"]).split())
                image_urls = get_image_urls(tweet_obj)
                if not is_retweet(tweet_obj) and image_urls and is_english(text):
                    words = text.lower().split()
                    words = [w for w in words if not w.startswith("@") and not w.startswith("#")]
                    cleaned_text = " ".join(words)
                    if cleaned_text not in seen_text:
                        for keyword in keywords:
                            if keyword in words:
                                selected_tweets.append(
                                    {
                                        "tid": tweet_obj["id"],
                                        "text": text,
                                        "image_urls": image_urls
                                    }
                                )
                                break
                        seen_text.add(cleaned_text)
    return selected_tweets


def select_tweets_by_rtqt_count():
    rtqt_tweet_pairs = []
    for month in MONTHS:
        with open(f"{JSONL_FOLDER}/{month}.jsonl", "r") as infile:
            for line in infile:
                tweet_obj = json.loads(line.strip())
                text = " ".join(remove_url(tweet_obj["text"]).split())
                image_urls = get_image_urls(tweet_obj)
                if not is_retweet(tweet_obj) and image_urls and is_english(text):
                    rtqt = tweet_obj["public_metrics"]["retweet_count"] + tweet_obj["public_metrics"]["quote_count"]
                    rtqt_tweet_pairs.append((
                        rtqt, 
                        {
                            "tid": tweet_obj["id"],
                            "text": text,
                            "image_urls": image_urls
                        }
                    ))
    
    rtqt_tweet_pairs.sort(key=lambda x: x[0], reverse=True)
    selected_tweets = [x[1] for x in rtqt_tweet_pairs[:1500]]
    return selected_tweets


def combine_sarcasm_tweets():
    selected_tweets = []
    seen_text = set()
    for jsonl_file in os.scandir(SARCASM_FOLDER):
        if jsonl_file.name.endswith(".jsonl"):
            with open(jsonl_file.path, "r") as infile:
                for line in infile:
                    tweet_obj = json.loads(line.strip())
                    tid = tweet_obj["id"]
                    text = " ".join(remove_url(tweet_obj["full_text"]).split())
                    if "media" in tweet_obj["entities"]:
                        for media_obj in tweet_obj["entities"]["media"]:
                            if media_obj["type"] == "photo":
                                words = text.lower().split()
                                words = [w for w in words if not w.startswith("@") and not w.startswith("#")]
                                cleaned_text = " ".join(words)
                                if cleaned_text not in seen_text:
                                    size = "large" if "large" in media_obj["sizes"] else "medium"
                                    image_url = f"{media_obj['media_url_https']}:{size}"
                                    selected_tweets.append((
                                        {
                                            "tid": tid,
                                            "text": text,
                                            "image_url": image_url
                                        }
                                    ))
                                    seen_text.add(cleaned_text)
                                    break
    return selected_tweets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--intents", default="p,c,v,s")
    args = parser.parse_args()
    
    intents = set(args.intents.split(","))
    
    # polar
    if "p" in intents:
        print("Select polar tweets...")
        polar_hashtags = []
        with open(POLAR_HASHTAGS_FILE, "r") as infile:
            for line in infile:
                polar_hashtags.append(line.strip())
        polar_tweets = select_tweets_by_hashtags(polar_hashtags)
        with open(POLAR_FILE, "w") as outfile:
            tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text", "image_urls"])
            tsv_writer.writeheader()
            for tweet in polar_tweets:
                tsv_writer.writerow(tweet)
                outfile.flush()
            
    # cta
    if "c" in intents:
        print("Select call_to_action tweets...")
        cta_hashtags = []
        with open(CTA_HASHTAGS_FILE, "r") as infile:
            for line in infile:
                cta_hashtags.append(line.strip())
        cta_tweets = select_tweets_by_hashtags(cta_hashtags)
        cta_keywords = ["retweet", "share", "must", "fight", "stand up"]
        cta_tweets.extend(select_tweets_by_keywords(cta_keywords))
        with open(CTA_FILE, "w") as outfile:
            tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text", "image_urls"])
            tsv_writer.writeheader()
            for tweet in cta_tweets:
                tsv_writer.writerow(tweet)
                outfile.flush()
            
    # viral
    if "v" in intents:
        print("Select viral tweets...")
        viral_tweets = select_tweets_by_rtqt_count()
        with open(VIRAL_FILE, "w") as outfile:
            tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text", "image_urls"])
            tsv_writer.writeheader()
            for tweet in viral_tweets:
                tsv_writer.writerow(tweet)
                outfile.flush()
            
    # sarcasm
    if "s" in intents:
        print("Select sarcasm tweets...")
        sarcasm_tweets = combine_sarcasm_tweets()
        with open(SARCASM_FILE, "w") as outfile:
            tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text", "image_url"])
            tsv_writer.writeheader()
            for tweet in sarcasm_tweets:
                tsv_writer.writerow(tweet)
                outfile.flush()