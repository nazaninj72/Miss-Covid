import nltk
import pandas as pd
import numpy as np
from newspaper import Article
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
col_list = ['user_id', 'Tweet','hashtag','url']
# df = pd.read_csv("samplenews.csv", usecols=col_list)
# df = pd.read_csv("rawtweets.csv", usecols=col_list)
# urls=[url for url in df['url'] if not pd.isnull(url)]
# import json,sys,csv
# mycsv = csv.writer(open('fullnews.csv', 'w'))
# mycsv.writerow(['doc', 'Content','url','keywords','summary'])

def preprocess(text):
    text = text.lower()# lowercase text
    text = REPLACE_BY_SPACE_RE.sub(" ",text)# replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub("",text)# delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join([word for word in text.split() if word not in STOPWORDS])# delete stopwords from text
    
    return text



# 
# for url in urls:
def scrapeurl(url):
    topic=""
    title=""
    content=""
    article = Article(url)
    try:
        article.download()
        article.html
        article.parse()
        title=preprocess(article.title)
        if len(article.text)>=200:
            content=preprocess(article.text)
            article.nlp()
            keywords=article.keywords
            summary=preprocess(article.summary)
            
    #         article2 = article.text.split()
            # mycsv.writerow([title, content,url,keywords,summary])
    except:
        print('***FAILED TO DOWNLOAD***', article.url)
    return content,keywords,summary
    
    
 
