import json
import requests
import time
from pprint import pprint
from ratelimit import limits, sleep_and_retry
from api_keys import TWITTER_API_KEYS
from twitter_auth import BearerTokenAuth

twitter_api_keys = TWITTER_API_KEYS()
consumer_key = twitter_api_keys.consumer_key
consumer_secret = twitter_api_keys.consumer_secret
BEARER_TOKEN = BearerTokenAuth(consumer_key, consumer_secret)
print(BEARER_TOKEN.bearer_token)

"""
filtered stream
"""
class FILTERED_STREAM:
    def __init__(self, filter_rules, bearer_token):
        self.stream_url = "https://api.twitter.com/2/tweets/search/stream"
        self.rules_url = "https://api.twitter.com/2/tweets/search/stream/rules"
        self.filter_rules = filter_rules
        self.auth = bearer_token


    def get_all_rules(self):
        response = requests.get(self.rules_url, auth=self.auth)
        if response.status_code != 200:
            raise Exception(f"Cannot get rules (HTTP %d): %s" % (response.status_code, response.text))
        pprint(response.json())
        return response.json()

    def delete_all_rules(self, rules):
        if rules is None or "data" not in rules:
            return None
        ids = list(map(lambda rule: rule["id"], rules["data"]))
        payload = {"delete": {"ids": ids}}
        response = requests.post(self.rules_url, auth=self.auth, json=payload)
        pprint(response.json())
        if response.status_code != 200:
            raise Exception(f"Cannot delete rules (HTTP %d): %s" % (response.status_code, response.text))

    def set_rules(self):
        if self.filter_rules is None:
            return
        payload = {"add": self.filter_rules}
        response = requests.post(self.rules_url, auth=self.auth, json=payload)
        pprint(response.json())
        if response.status_code != 201:
            raise Exception(f"Cannot create rules (HTTP %d): %s" % (response.status_code, response.text))

    def rules_setup(self):
        print("Get original rules...")
        current_rules = self.get_all_rules()
        print("Delete original rules...")
        self.delete_all_rules(current_rules)
        print("Set new rules...")
        self.set_rules()
        print("Current rules:")
        current_rules = self.get_all_rules()
    
    def stream_connect(self):
        response = requests.get(self.stream_url, auth=self.auth, stream=True, params={"tweet.fields": "author_id,created_at,entities,text,public_metrics", "user.fields": "id", "expansions": "referenced_tweets.id,referenced_tweets.id.author_id"})

        if response.status_code > 201:
            raise Exception(f"{response.status_code}: {response.text}")

        for response_line in response.iter_lines():
            if response_line:
                tweet_dict = json.loads(response_line)
                # TODO: process or write to file
                

def filtered_stream_wrapper(filter_rules, bearer_token):
    fs = FILTERED_STREAM(filter_rules, bearer_token)
    fs.rules_setup()

    timeout = 0
    while True:
        try:
            fs.stream_connect()
            timeout = 0
        except Exception as e:
            if str(e).startswith("429"):
                time.sleep(2 ** timeout)
                timeout += 1
            else:
                print(e)


"""
recent search
"""
@sleep_and_retry
@limits(calls=450, period=900)
def recent_search(query, start_time, end_time, next_token, bearer_token):
    endpoint_url = f"https://api.twitter.com/2/tweets/search/recent"
    headers = {"Accept-Encoding": "gzip"}
    if next_token:
        params = {"query": query, "tweet.fields": "created_at,entities,public_metrics,author_id,referenced_tweets", "start_time": start_time, "end_time": end_time, "max_results": 100, "next_token": next_token}
    else:
        params = {"query": query, "tweet.fields": "created_at,entities,public_metrics,author_id,referenced_tweets", "start_time": start_time, "end_time": end_time, "max_results": 100}

    response = requests.get(endpoint_url, auth=bearer_token, headers=headers, params=params)
    if response.status_code > 201:
        print(f"{response.status_code}: {response.text}")
        raise Exception(f"{response.status_code}: {response.text}")

    r_json = response.json()
    tweet_dicts = r_json["data"]
    if "next_token" in r_json["meta"]:
        next_token = r_json["meta"]["next_token"]
    else:
        next_token = "Does not exist!"

    r_obj_list = []
    for tweet_dict in tweet_dicts:
        try:
            tweet_dict = {"data": tweet_dict}
            r_obj_list.append(tweet_dict)
        except Exception as e:
            print(e)

    return {"r_obj_list": r_obj_list, "next_token": next_token}


def recent_search_wrapper(query, start_time, end_time, bearer_token):
    next_token = None
    while next_token != "Does not exist!":
        r_obj_list, next_token = recent_search(query, start_time, end_time, next_token, bearer_token)
        # TODO: process or write to file