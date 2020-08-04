# DS4CG2020-aucode


# Getting Targeted Data from Twitter API:
In order to retrieve tweets related to specific topic, you should use `get_hydrated_tweets.py` script under `data` repository. 
## Instructions:
- Clone  https://github.com/echen102/COVID-19-TweetIDs
- Open `get_hydrated_tweets.py` in your code editor. 
- Replace `dataPath` and `mainpath` to the paths that you cloned "COVID-19-TweetIDs" and this repo.
- In the function `hydrate()` modify `keywords` to your intended keywords. (Currently it is set to relevant tweets about "flu")
- Open `keys.json` and add your own Twitter API credentials. 
- Run `get_hydrated_tweets.py` to start extracting relevant tweets.  

# Getting meta data of each user in labeled data:
To get metadata of each user, you can simply run `get_metadata.py` . 

