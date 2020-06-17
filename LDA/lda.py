from functions import *
from gensim import corpora
import pickle
import random
import gensim
import pandas as pd

def getdata(file):
    col_list = ['doc', 'Content','url','keywords','summary']
    df = pd.read_csv(file, usecols=col_list)
    return df




def train_lda():
    NUM_TOPICS = 10
    num_words=5
    filename="/home/nazaninjafar/ds4cg2020/data_preprocess/news.csv"

    df = getdata(filename)
    corpuses=[corpus for corpus in df['summary'] if not pd.isnull(corpus)]
    text_data = []
    for corpus in corpuses:
        tokens = prepare_text_for_lda(corpus)
        if random.random() > .99:
    #         print(tokens)
            text_data.append(tokens)

    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]

    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')


    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=5)
    topiclist=[]
    for topic in topics:
        templist=topic[1].split("+")
        topiclist.append([s.split("*")[1].replace('"', '').replace(" ", "") for s in templist])


    return topiclist


if __name__ == "__main__":
    # print(train_lda())
    topics=train_lda()
    # print((topics))
    i=1
    print("Generated Topics by LDA:")
    for t in topics:
        print(str(i)+"-"+str(t).replace("[","").replace("]",""))
        i+=1

