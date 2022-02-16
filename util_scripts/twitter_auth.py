import os
import sys

import requests
from requests.auth import AuthBase
from requests.auth import HTTPBasicAuth

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""


class BearerTokenAuth(AuthBase):
    def __init__(self, consumer_key, consumer_secret):
        self.bearer_token_url = "https://api.twitter.com/oauth2/token"
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.bearer_token = self.get_bearer_token()

    def get_bearer_token(self):
        response = requests.post(
            self.bearer_token_url,
            auth=(self.consumer_key, self.consumer_secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": "TwitterDevFilteredStreamQuickStartPython"},
        )

        if response.status_code != 200:
            raise Exception(f"Cannot get a Bearer token (HTTP {response.status_code}): {response.text}")

        body = response.json()
        return body["access_token"]

    def __call__(self, r):
        r.headers["Authorization"] = f"Bearer {self.bearer_token}"
        r.headers["User-Agent"] = "TwitterDevFilteredStreamQuickStartPython"
        return r
