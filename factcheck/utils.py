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
def preprocess(text):
    text = text.lower()# lowercase text
    text = REPLACE_BY_SPACE_RE.sub(" ",text)# replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub("",text)# delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join([word for word in text.split() if word not in STOPWORDS])# delete stopwords from text
    
    return text

def scrapecontent(url):
    title,content,keywords,summary="","","",""
    article = Article(url)
    try:
        article.download()
        article.html
        article.parse()
        title=preprocess(article.title)
        if len(article.text)>200:
            content=preprocess(article.text)
            article.nlp()
            keywords=article.keywords
            summary=preprocess(article.summary)
    except:
        print('***FAILED TO DOWNLOAD***', article.url)
    return title,content,keywords,summary

def scrape_statement(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    statement=soup.find("div",attrs={'class':'m-statement__quote'}).text
    labeling_img = soup.find('div', attrs={'class':'m-statement__meter'}).find('div', attrs={'class':'c-image'}).find('picture')
        #     print(labeling_img)
    label = labeling_img.find('img')['alt']
#     print(statement,label)
    return statement,label