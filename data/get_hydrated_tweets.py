import json,sys,csv
from twarc import Twarc
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
from os import walk
import sys
#path to the cloned repository : https://github.com/echen102/COVID-19-TweetIDs
dataPath="/home/nazaninjafar/ds4cg2020/COVID-19-TweetIDs"
def hydrate(files,api_key):
    consumer_key=api_key['consumer_key']
    consumer_secret=api_key['consumer_secret']
    access_token=api_key['access_token']
    access_token_secret=api_key['access_token_secret']
    t = Twarc(consumer_key, consumer_secret, access_token, access_token_secret)
    file1 = open('/home/nazaninjafar/ds4cg2020/data/flutids.txt', 'w')
    keywords = ('flu', 'common flu', 'covid19 flu', 'coronavirus common flu')
    records=[]
    hashtags=[""]
    for tweetIDs in files:
        
        for tweet in t.hydrate(open(dataPath+"/"+tweetIDs)):
            txt=tweet['full_text']
            if (tweet["lang"]=="en") and (not tweet['retweeted'] and 'RT @' not in tweet['full_text']):
                if any(keyword in tweet["full_text"].lower() for keyword in keywords):


                    tid=str(tweet['id_str'])
                    file1.write(tid+'\n')
                    screen_name=tweet['user']['screen_name']
                    if not tweet["entities"]["hashtags"]: 
                        hashtags=[""]
                    else:)
                        for h in tweet["entities"]["hashtags"]:
                            hashtags.append(h["text"])
                            continue
                    if not tweet["entities"]["urls"]:
                            url=""
                    else:
                        for urls in tweet["entities"]["urls"]:
                            url=str(urls["expanded_url"])
                            continue

                    retweets=str(tweet['retweet_count'])
                    favorites=str(tweet['favorite_count'])
                    records.append([screen_name,txt,hashtags,url,retweets,favorites])
    df = pd.DataFrame(records, columns=['screen_name' ,'tweet','hashtag','url','#retweets','#favorites']) 
    df.to_csv('/home/nazaninjafar/ds4cg2020/data/tweets.csv')  
    file1.close()

allfiles=[]
def getdirList():
    mainpath=dataPath+"/"
    # mypath=[paths for paths in ]
    paths=['2020-01', '2020-02', '2020-03', '2020-04', '2020-05','2020-06']
    for path in paths:
        onlyfiles = [path+"/"+f for f in listdir(mainpath+path) if isfile(join(mainpath+path, f))]
        allfiles.append(onlyfiles)
    return allfiles
    
fileLists=getdirList()
with open('keys.json') as f:
  keys = json.load(f)

hydrate(fileLists[0],keys)






