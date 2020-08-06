
import re
from sklearn.preprocessing import normalize
import numpy as np
import datetime
from datetime import timedelta
import pandas as pd
# import the module 
import tweepy 
import json
import os
import os.path
from os import path



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

# path_to_data='../alldata.tsv'
# path_to_metadata='../metadata.tsv'
def get_metadata(data):
    # data=pd.read_csv(path_to_data)
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
            # print(verified)
            fav_count=user.favourites_count
            tweet_count=user.statuses_count
            records.append([str(id),screen_name,description,url,friend_count,follower_count,created,uid,location,verified,fav_count,tweet_count])
        except tweepy.error.TweepError as e:
            if "u'code': 63" in e.reason:
                print('Twitter screen name  is suspended.')
            




    df = pd.DataFrame(records, columns=['id','screen_name','description','url','friend_count','follower_count','created','uid','location','verified','fav_count','tweet_count']) 
    return df






def norm(X):
    X=normalize(X, axis=0, norm='max')
    return X
def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r"http\S+", "", text)
    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text=text.replace("#","")

    return text







def get_metadata_features(meta_data):
    # meta_data=pd.read_csv('/home/nazaninjafar/ds4cg2020/bert-covid/data/user_metadata.tsv')
    meta_data=meta_data.replace(np.nan, '', regex=True)
    fc = np.expand_dims((meta_data.friend_count.values), axis=1)
    fwc = np.expand_dims(meta_data.follower_count.values, axis=1)
    favc = np.expand_dims(meta_data.fav_count.values, axis=1)
    tc = np.expand_dims(meta_data.tweet_count.values, axis=1)
    TFF=fc+1/fwc+1
    creation_times=meta_data.created.values
    today = datetime.datetime.today()
# print(today)
    time_difference=[]
    for a in creation_times:
        account_date=datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S')
        time_difference.append((today - account_date).days)
    age=np.array(time_difference).reshape(-1,1)
    md_X=np.concatenate((favc,tc,TFF,age),axis=1)
    md_X=norm(md_X)
    return md_X

