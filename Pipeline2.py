import csv
from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import random
import csv
from simpletransformers.classification import ClassificationModel
from simpletransformers.classification import ClassificationArgs
import pandas as pd
import logging

def read_csv1(csv_file):
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader
        csv_reader = csv.DictReader(file)
        tweets=[]
        data=[]
        for row in csv_reader:
            tweets.append([row['Tweet id'], row['Tweet text']])
    return tweets

def transformer1(model_name, train_df, eval_df, epochs, labels):
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    model_args = ClassificationArgs()
    model_args.num_train_epochs=epochs
    model_args.labels_list = [0.0, 0.2, 0.5, 0.8, 1.0]
    model_args.reprocess_input_data=True
    model_args.overwrite_output_dir=True
    model = ClassificationModel(model_name, 'bert-base-cased', num_labels=labels, args=model_args)
    # You can set class weights by using the optional weight argument

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    # You can set class weights by using the optional weight argument

    return model, result, model_outputs, wrong_predictions

def prediction(model, data):
    predictions, raw_outputs = model.predict(data)
    return predictions, raw_outputs

def predict(lst):
    predictions, raw_outputs = model.predict(lst)
    return predictions, raw_outputs

def get_mismatched(labels, preds):
    mismatched = labels != preds
    #print(mismatched)
    examples = processor.get_dev_examples(args['data_dir'])
    wrong = [i for (i, v) in zip(examples, mismatched) if v]

    return wrong

def get_eval_report(labels, preds):
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return get_eval_report(labels, preds)

def read_csv_category(csv_file,category):
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader
        csv_reader = csv.DictReader(file)
        stored_claims=[]
        for row in csv_reader:
            if row['Class'].lower() == category.lower() and row['Value'] is not '-1' and row['Value'] is not '-2' :
                stored_claims.append((row['Claim Text'], row['Value']))
    random.shuffle(stored_claims)
    return stored_claims

def read_csv2(csv_file):
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader
        csv_reader = csv.DictReader(file)
        stored_claims=[]
        for row in csv_reader:
            stored_claims.append((row['Claim Text'], row['Value']))
    return stored_claims

def generate_data(data):
    X = [data[i][0] for i in range(len(data)) if data[i][1] != '-1' and data[i][1] != '-2']
    Y = [float(data[i][1]) for i in range(len(data)) if data[i][1] != '-1' and data[i][1] != '-2']
    return X, Y

def transformer2(model_name, train_df, eval_df, epochs, labels):
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    model_args = ClassificationArgs()
    model_args.num_train_epochs=epochs
    model_args.labels_list = [0, 1, 2, 3, 4]
    model_args.reprocess_input_data=True
    model_args.overwrite_output_dir=True
    model = ClassificationModel(model_name, 'bert-base-cased', num_labels=labels, args=model_args)
    # You can set class weights by using the optional weight argument

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    # You can set class weights by using the optional weight argument

    return model, result, model_outputs, wrong_predictions

# In[17]:


def read_csv3(csv_file):
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader
        csv_reader = csv.DictReader(file)
        tweets=[]
        data=[]
        for row in csv_reader:
            tweets.append([row['Tweet id'], row['Tweet text'], row['Tweet label']])
    return tweets


# In[18]:


# In[ ]:
def main(epoch1=7, epoch2=8):
    tweets=read_csv1('data/potential_misinformation.csv')
    data=read_csv2('data/claims.csv')

    X,Y=generate_data(data[:2000])
    random.shuffle(X)
    random.shuffle(Y)
    #print(len(X),len(Y))

    X_train, X_test, y_train, y_test=train_test_split(X, Y, test_size=0.2, random_state=42)

    train_data = [[X_train[i], float(y_train[i])] for i in range(len(X_train))]
    train_df = pd.DataFrame(train_data)

    eval_data = [[X_test[i], float(y_test[i])] for i in range (len(X_test))]
    eval_df = pd.DataFrame(eval_data)

    model, result, model_outputs, wrong_predictions = transformer1('bert', train_df, eval_df, epoch1, 5)


    # In[10]:


    trainers=[tweets[i][1] for i in range(len(tweets))]
    predictions, raw_outputs = prediction(model, trainers)


    # In[11]:


    predicts = [round(predictions[i]) for i in range(len(predictions))]
    #d={0:'False', 1:'True'}
    for i in range(len(tweets)):
        tweets[i].append(predicts[i])


    # In[12]:


    with open('data/potential_misinformation.csv', mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Tweet id', 'Tweet text', 'Tweet label'])
            for tweet in tweets:
                writer.writerow([str(tweet[0]),tweet[1],tweet[2]])

    with open('data/claims.csv', mode='r', encoding='utf-8') as file:
        d={'politics':0, 'health':1, 'transmission':2, 'immunity':3, 'other':4}
        reader = csv.reader
        csv_reader = csv.DictReader(file)
        data=[]
        for row in csv_reader:
            data.append((row['Claim Text'], d[row['Class'].strip().lower()]))


    # In[15]:


    X,Y=generate_data(data)
    #print(len(X), len(Y))
    X_train, X_test, y_train, y_test=train_test_split(X, Y, test_size=0.2, random_state=42)


    # In[16]:


    train_data = [[X_train[i], float(y_train[i])] for i in range(len(X_train))]
    train_df = pd.DataFrame(train_data)

    eval_data = [[X_test[i], float(y_test[i])] for i in range (len(X_test))]
    eval_df = pd.DataFrame(eval_data)

    model, result, model_outputs, wrong_predictions = transformer('bert', train_df, eval_df, epoch2, 5)

    tweets = read_csv3('data/potential_misinformation.csv')
    #print(len(tweets))
    trainers=[tweets[i][1] for i in range(len(tweets))]
    predictions, raw_outputs = prediction(model, trainers)


    # In[19]:


    politics=[]
    health=[]
    transmission=[]
    immunity=[]
    other=[]


    # In[20]:


    d={0:'politics', 1:'health', 2:'transmission',3:'immunity',4:'other'}
    for i in range(len(tweets)):
        if predictions[i] is 0:
            politics.append(tweets[i])

        elif predictions[i] is 1:
            health.append(tweets[i])

        elif predictions[i] is 2:
            transmission.append(tweets[i])

        elif predictions[i] is 3:
            immunity.append(tweets[i])

        else:
            other.append(tweets[i])



    # In[26]:


    with open('data/politics.tsv', mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Tweet id', 'Tweet text', 'True/False?'])
            for tweet in politics:
                writer.writerow([str(tweet[0]),tweet[1],tweet[2]])


    # In[27]:


    with open('data/other.tsv', mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Tweet id', 'Tweet text', 'True/False?'])
            for tweet in other:
                writer.writerow([str(tweet[0]),tweet[1],tweet[2]])


    # In[28]:


    with open('data/health.tsv', mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Tweet id', 'Tweet text', 'True/False?'])
            for tweet in health:
                writer.writerow([str(tweet[0]),tweet[1],tweet[2]])


    # In[29]:


    with open('data/transmission.tsv', mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Tweet id', 'Tweet text', 'True/False?'])
            for tweet in transmission:
                writer.writerow([str(tweet[0]),tweet[1],tweet[2]])


    # In[30]:


    with open('data/immunity.tsv', mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Tweet id', 'Tweet text', 'True/False?'])
            for tweet in immunity:
                writer.writerow([str(tweet[0]),tweet[1],tweet[2]])



if __name__ == "__main__":
    main()
