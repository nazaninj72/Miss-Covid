# import the module 
import tweepy 
import json
import os
import os.path
from os import path

import re
from tqdm import tqdm
import numpy as np
import pandas as pd

# assign the values accordingly 
consumer_key='9HJFwgSvjjwdciAwQHDAw6WTD'
consumer_secret='1OwOHIkHiJrEx6kqFB1mUlRXOxKRxWVdF1eYrdwTlYvBS5zYtZ'
access_token='1268553900459806727-0Kv0iLIptT2RR9OUOKQHCoG6IHueoV'
access_token_secret='ynsIooJA70BsU27C161LNcDXagiOaRUv03w8mhNW9D1Sa'
  
# authorization of consumer key and consumer secret 
auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
  
# set access to user's access key and access secret  
auth.set_access_token(access_token, access_token_secret) 
# calling the api  
api = tweepy.API(auth,wait_on_rate_limit=True) 

    
def get_friendscount(user):
    return user.friends_count

def get_followercount(user):
    return user.followers_count

def get_uid(user):
    return user.id_str

def get_desc(user):
    if not user.description:
        return ""
    else:
        return user.description

def get_loc(user):
    if not user.location:
        return ""
    else:
        return user.location

def get_url(user):
    if not user.url:
        return ""
    else:
        return user.url

# def get_tweetcounts(user):

path_to_data='/home/nazaninjafar/ds4cg2020/UMassDS/DS4CG2020-aucode/data/topics/health.tsv'
path_to_metadata='/home/nazaninjafar/ds4cg2020/UMassDS/DS4CG2020-aucode/data/topics/health-metadata.tsv'

data=pd.read_csv(path_to_data)
ids=data['id'].values.tolist()
records=[]
for id in ids:
    
    
    try:
        tweet = api.get_status(id)
        screen_name=tweet.user.screen_name
        user=api.get_user(screen_name)
        friend_count=get_friendscount(user)
        follower_count=get_followercount(user)
        uid=str(get_uid(user))
        description=get_desc(user)
        created=str(user.created_at)
        location=get_loc(user)
        url=get_url(user)
        # tweet_count=
        verified=user.verified
        print(verified)
        fav_count=user.favourites_count
        tweet_count=user.statuses_count
        records.append([str(id),screen_name,description,url,friend_count,follower_count,created,uid,location,verified,fav_count,tweet_count])
    except tweepy.error.TweepError as e:
        if "u'code': 63" in e.reason:
          print('Twitter screen name  is suspended.')
        




df = pd.DataFrame(records, columns=['id','screen_name','description','url','friend_count','follower_count','created','uid','location','verified','fav_count','tweet_count']) 
df.to_csv(path_to_metadata)
