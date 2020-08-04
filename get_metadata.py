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


def write_friends(foldername,friendslimit):
    directory = "/home/nazaninjafar/ds4cg2020/data/ppnetwork"+"/"+foldername
    access_rights = 0o755
    try:
        os.mkdir(directory,access_rights)
        # c = tweepy.Cursor(api.friends, screen_name).items(friendslimit) 
        with open(directory+"/"+'friends.json', 'w') as f:
            for friend in tweepy.Cursor(api.friends, screen_name).items(friendslimit): 
                json.dump(friend._json, f)
                
        f.close()
        print("sucessfully generated friend list for user: "+foldername)
    except OSError:
        print ("Creation of the directory %s failed" % directory)
#     else:
#         print ("Successfully created the directory %s " % path)
    # getting all the friends 
    
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



data=pd.read_csv('/home/nazaninjafar/ds4cg2020/bert-covid/data/alldata.tsv')
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
df.to_csv('/home/nazaninjafar/ds4cg2020/bert-covid/data/user_metadata.tsv')
