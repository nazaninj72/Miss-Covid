
import re
from sklearn.preprocessing import normalize
import numpy as np
import datetime
from datetime import timedelta
import pandas as pd

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


def get_metadata_features():
    meta_data=pd.read_csv('/home/nazaninjafar/ds4cg2020/bert-covid/data/user_metadata.tsv')
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

