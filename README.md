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

# Misinformation classification steps of twitter data:
After testing several baseline models (rf,svm,regression) and viewing their results, we decided on testing pretrained bert models on top of a simple fully connected classifier. For this purpose we followed different approaches:
First approach is testing the twitter data on the bert base model. 
First we read data from `alldata.tsv` file in ‘data’ directory
Data format is as follows:
`Id,label,tweet`

Where id refers to tweet id , label is 0,1 values such that 0 refers to true labels and 1 refers to false labels. And the tweet is the raw content of the tweet. 
To run a pre trained bert model on given data, we only need to run the main.py code on the models directory. 
To run only bert base model, we can set ‘classifier_type’ in main.py to ‘bert’ and to run the metadata incorporated features, you only require to set `classifier_type’` to `berta_metadata`
Metadata incorporated model adds 4 features to pretrained bert model on the content of tweets. These features are as follows: 
- Account age in terms of number of days, 
- Ratio of number of friends/number of followers
- Number of tweets
- Number of liked posts
All the features are normalized. 
After extracting these features, our previously defined model for pretrained bert gets concatenated with these features after dimension of each of them have been reduced with fully connected network to number of classes (2). 
Output of these separate fully connected layers are then concatenated for final classification. 
To run each of these models you can follow below instructions:
Assuming you have access to GPU you can train misinformation classification on our dataset. 
In this project,we used BERT base and ct-BERT(https://github.com/digitalepidemiologylab/covid-twitter-bert) models to fine tune our dataset. Additionally as explained above we experimented these models with and without metadata features. Therefore to reproduce our results with these options you can use following command line arguments: 
`MODEL=bert                                  # Choices :[bert,ct-bert]


BATCH_SIZE=32
LR=5e-5
NUM_EPOCHS=1
DATASET=flu-claims  #choices: [flu-claims,topics]
TOPIC=politics #choices: [politics,other,transmission,health,immunity]

python main.py \
  --model $MODEL \
  --batch_size $BATCH_SIZE \
  --epoch $NUM_EPOCHS \
  --learningrate $LR \
  --with_metadata #Whether to apply metadata features in experiments \
  --dataset $DATASET if topics are chosen then topic should be specified 
  --topic $TOPIC`
Please note that if you want to use ct-bert model, you should reduce batch-size to 8. CT-bert model is pretrained on top of bert large uncased with 1024 embedding dimensions. Therefore, running the model on larger batch sizes will return cuda out of memory error.
