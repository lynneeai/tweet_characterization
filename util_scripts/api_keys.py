# all Twitter API keys we have
ALL_API_KEYS = {
    "Lin_Columbia": {
        "user": "Lin",
        "account_email": "la2734@columbia.edu",
        "consumer_key": "dxAq8WyDTOFgtPJ96ydr5ocbT",
        "consumer_secret": "85eYNazPS5xAzxOdTyCpmnsnTEKXcEnE7LK5bpnHyuHcR2hNKe",
        "access_token_key": "1258832589445464064-YuUNvMcQgcHr1fJTiimsHITw0g0HV0",
        "access_token_secret": "tM63a8MHRV8zqdBV6HYogoxD8FnIL5193mBj3rovbYGhn"
    },
    "Lin_Hotmail": {
        "user": "Lin",
        "account_email": "lynneeai@hotmail.com",
        "consumer_key": "G55LIvWCQnKSmouAfauOvMaYk",
        "consumer_secret": "XIlEUemxMSXzHeWlvsqRjOex3FuWIjaggZcoFOt7p0JD40bwsN",
        "access_token_key": "774225590689955840-eRZ34tKArCARsXL0UbD0qBuAxP85cgv",
        "access_token_secret": "eZi5RtwcDUuIu9mLN21GjA2tZlFfmKMZCabKaadPUjZPo"
    }
}

# choose which account to use
ACCOUNT = "Lin_Columbia"

# define keys to be called
class TWITTER_API_KEYS:
    # keys
    consumer_key = ALL_API_KEYS[ACCOUNT]["consumer_key"]
    consumer_secret = ALL_API_KEYS[ACCOUNT]["consumer_secret"]
    access_token_key = ALL_API_KEYS[ACCOUNT]["access_token_key"]
    access_token_secret = ALL_API_KEYS[ACCOUNT]["access_token_secret"]
    