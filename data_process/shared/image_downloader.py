import argparse
import csv
import json
import os
import re
import sys
import logging
from tqdm import tqdm as tqdm
import urllib.request
import socket
import http.client
from urllib.error import HTTPError, URLError
from twarc.client import Twarc
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/../.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from util_scripts.utils import boolean_string, remove_url

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_filename", type=str, help="input file")
parser.add_argument("-da", "--dataset", type=str, help="dataset")
parser.add_argument("-hd", "--hydrate", type=boolean_string, default="True")
parser.add_argument("-u", "--update", type=boolean_string, default="True")
parser.add_argument("-d", "--download", type=boolean_string, default="True")
parser.add_argument("-p", "--platform", type=str, default="twitter")
args = parser.parse_args()

filepath = f"{project_root_dir}/datasets/{args.dataset}/tweets/{args.input_filename}"
filename = args.input_filename.split(".")[0]
filename_full = f"{project_root_dir}/datasets/{args.dataset}/tweets/{filename}"
output_folder = f"{project_root_dir}/datasets/{args.dataset}/images/{filename}_images/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

twarc = Twarc()


def tsv_file_linenum(tsv_file):
    linenum = 0
    with open(tsv_file, "r") as infile:
        tsv_reader = csv.DictReader(infile, delimiter="\t")
        for row in tsv_reader:
            linenum += 1
    return linenum

def get_tids():
    tids = []
    with open(filepath, "r") as infile:
        if filepath.endswith(".tsv"):
            tsv_reader = csv.DictReader(infile, delimiter="\t")
            for row in tsv_reader:
                tids.append(row["tid"])
        else:
            for line in infile:
                tweet_obj = json.loads(line.strip())
                tids.append(tweet_obj["id"])
    return tids
    

def tid_generator(tids):
    for tid in tids:
        yield tid
        

def get_tweets_details():
    print("Getting tweets details and image urls...")
    tid_list = get_tids()
    tid_gen = tid_generator(tid_list)
    pbar = tqdm(total=len(tid_list))
    with open(f"{filename_full}.jsonl", "w") as outfile:
        for tweet in twarc.hydrate(tid_gen):
            json.dump(tweet, outfile)
            outfile.write(f"\n")
            pbar.update(1)
    pbar.close()


def update_tsv_file():
    print("Create tsv file...")
    tid_text_imageurl_tuples = []
    with open(f"{filename_full}.jsonl", "r") as infile:
        for line in infile:
            tweet_obj = json.loads(line.strip())
            tid = tweet_obj["id"]
            text = " ".join(remove_url(tweet_obj["full_text"]).split())
            if "media" in tweet_obj["entities"]:
                for media_obj in tweet_obj["entities"]["media"]:
                    if media_obj["type"] == "photo":
                        size = "large" if "large" in media_obj["sizes"] else "medium"
                        image_url = f"{media_obj['media_url_https']}:{size}"
                        tid_text_imageurl_tuples.append((tid, text, image_url))
                        break
    with open(f"{filename_full}.tsv", "w") as outfile:
        tsv_writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["tid", "text", "image_url"])
        tsv_writer.writeheader()
        for tid, text, image_url in tid_text_imageurl_tuples:
            tsv_writer.writerow({"tid": tid, "text": text, "image_url":image_url})
            

def download_images():
    print("Downloading images...")
    socket.setdefaulttimeout(30)
    tid_imageurl_pairs = []
    with open(f"{filename_full}.tsv", "r") as infile:
        tsv_reader = csv.DictReader(infile, delimiter="\t")
        for row in tsv_reader:
            tid = row["tid"]
            if not os.path.exists(f"{output_folder}/{tid}.jpg"):
                tid_imageurl_pairs.append((tid, row["image_url"]))
            
    for tid, image_url in tqdm(tid_imageurl_pairs):
        try:
            urllib.request.urlretrieve(image_url, f"{output_folder}/{tid}.jpg")
        except HTTPError as err:
            pass
        except URLError as err:
            pass
        except socket.timeout as err:
            pass
        except http.client.RemoteDisconnected as err:
            pass
    print("Done!")
    

def download_fb_images():
    print("Downloading facebook images...")
    
    socket.setdefaulttimeout(30)
    os.environ["WDM_LOG_LEVEL"] = str(logging.WARNING)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
    
    # get actual image urls
    print("Getting actual image urls...")
    
    temp_tsv_outfile = open(f"{filename}_temp.tsv", "w")
    temp_tsv_writer = csv.DictWriter(temp_tsv_outfile, delimiter="\t", fieldnames=["tid", "text", "image_url", "humor_score"])
    temp_tsv_writer.writeheader()
    
    tid_imageurl_pairs = []
    pbar = tqdm(total=tsv_file_linenum(f"{filename_full}.tsv"))
    with open(f"{filename_full}.tsv", "r") as infile:
        tsv_reader = csv.DictReader(infile, delimiter="\t")
        for row in tsv_reader:
            try:
                chrome_driver.get(row["image_url"])
                WebDriverWait(chrome_driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "img")))
                image_url = chrome_driver.find_element_by_xpath("//img").get_attribute("src")
                
                row_dict = dict(row)
                row_dict["image_url"] = image_url
                temp_tsv_writer.writerow(row_dict)
                temp_tsv_outfile.flush()
                
                if not os.path.exists(f"{output_folder}/{row['tid']}.jpg"):
                    tid_imageurl_pairs.append((row["tid"], image_url))
            except:
                pass
            
            pbar.update(1)
    temp_tsv_outfile.close()
    chrome_driver.quit()
    pbar.close()
    
    # update tsv
    os.system(f"mv {filename}_temp.tsv {filename_full}.tsv")
    
    # download images
    print("Downloading images...")
    for tid, image_url in tqdm(tid_imageurl_pairs):
        try:
            urllib.request.urlretrieve(image_url, f"{output_folder}/{tid}.jpg")
        except HTTPError as err:
            pass
        except URLError as err:
            pass
        except socket.timeout as err:
            pass
        except http.client.RemoteDisconnected as err:
            pass
    print("Done!")
    

if __name__ == "__main__":
    if args.hydrate:
        get_tweets_details()
    if args.update:
        update_tsv_file()
    if args.download:
        if args.platform == "twitter":
            download_images()
        elif args.platform == "facebook":
            download_fb_images()