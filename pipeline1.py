import csv
import re
import nltk
import scipy
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

def read_csv(csv_file):
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader
        csv_reader = csv.DictReader(file)
        tweets=[]
        data=[]
        for row in csv_reader:
            tweets.append([row['Tweet id'], row['Tweet text']])
            data.append(row['Tweet text'])
    return tweets,data

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    with open(gloveFile, encoding="utf8" ) as f:
        content = f.readlines()
    model = {}
    for line in content:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

def preprocess(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split 
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    stop_words=['corona', 'covid', 'coronavirus', 'covid19', 'covid-19']
    cleaned_words = list(set([w for w in words if w not in stopword_set and w not in stop_words]))
    #print(cleaned_words)
    return cleaned_words

def cosine_distance_between_two_words(word1, word2):
    import scipy
    return (1- scipy.spatial.distance.cosine(model[word1], model[word2]))

def calculate_heat_matrix_for_two_sentences(s1,s2):
    s1 = preprocess(s1)
    s2 = preprocess(s2)
    result_list = [[cosine_distance_between_two_words(word1, word2) for word2 in s2] for word1 in s1]
    result_df = pd.DataFrame(result_list)
    result_df.columns = s2
    result_df.index = s1
    return result_df

def cosine_distance_wordembedding(s1, s2):
    #import scipy
    #try:
    vector_1 = np.mean([model[word] if word in model else 0 for word in preprocess(s1) ],axis=0)
    #except:
    #    vector_1 = np.zeros(100)
    #try:
    vector_2 = np.mean([model[word] if word in model else 0 for word in preprocess(s2) ],axis=0)
    #except:
    #    vector_2 = np.zeros(100)
    #print(vector_1, vector_2)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    #print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')
    return cosine

#print(cosine_distance_wordembedding("bleach cures corona" , "bleach cure corona"))
'''def heat_map_matrix_between_two_sentences(s1,s2):
    df = calculate_heat_matrix_for_two_sentences(s1,s2)
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5,5)) 
    ax_blue = sns.heatmap(df, cmap="YlGnBu")
    # ax_red = sns.heatmap(df)
    print(cosine_distance_wordembedding_method(s1, s2))
    return ax_blue'''

def get_neighbors(training_set,  
                  test_instance, 
                  k, 
                  distance=cosine_distance_wordembedding, max_d=0.1):
    """
    get_neighors calculates a list of the k nearest neighbors
    of an instance 'test_instance'.
    The list neighbors contains 3-tuples with  
    (index, dist, label)
    where 
    index    is the index from the training_set, 
    dist     is the distance between the test_instance and the 
             instance training_set[index]
    distance is a reference to a function used to calculate the 
             distances
    """
    distances = []
    for index in range(len(training_set)):
        dist = cosine_distance_wordembedding(test_instance, training_set[index])
        #print(dist)
        if dist<=max_d:
            distances.append((training_set[index], dist))
    distances.sort(key=lambda x: x[1])
    if(len(distances)<k):
        return []
    neighbors = distances[:k]
    return(neighbors)
            
def main(k=10):
    tweets,data=read_csv('incoming_tweets.csv')

    with open('claims.csv', mode='r', encoding='utf-8') as file:
        reader = csv.reader
        csv_reader = csv.DictReader(file)
        claims_data=[]
        for row in csv_reader:
            claims_data.append(row['Claim Text'])
    
    nltk.download('stopwords')
    
    gloveFile = "glove.6B.100d.txt"
    
    model = loadGloveModel(gloveFile)
 
    potential_rumors=[]
    
    for tweet in tweets:
        #print('Entered')
        neighbors = get_neighbors(claims_data, tweet[1], 10)
        #print(neighbors)
        if len(neighbors) is not 0:
            #print(neighbors)
            #print("*****"*10)
            #v = vote(neighbors)
            #if v is not -1:
            potential_rumors.append([tweet[0],tweet[1]])

    with open('potential_misinformation.csv', mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Tweet id', 'Tweet text'])
            for tweet in potential_rumors:
                writer.writerow([str(tweet[0]),tweet[1]])
                
if __name__ == "__main__":
    main()  