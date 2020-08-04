from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

import itertools
import matplotlib.pyplot as plt

import numpy as np

class LRegression:
    def __init__(self,random_state=0):
        self.random_state=random_state 

    def train(self,X,y):

    

        indices = np.arange(len(X))
        train_idx, test_idx, y_train, y_test= train_test_split(indices, y,stratify = y, test_size=0.25, random_state=0)
        train_X = X[train_idx]
        test_X = X[test_idx]
        clf = LogisticRegression(random_state=0)
        clf=clf.fit(train_X, y_train)
        return clf,test_X, y_test,y_train,train_X


    def evaluate(self,model,test_X, y_test,y_train,train_X):
        """Compare machine learning model to baseline performance.
        Computes statistics and shows ROC curve."""
        results = {}
        
        print(f'Model Accuracy: {model.score(test_X, y_test)}')
        train_probs = model.predict_proba(train_X)[:, 1]
        probs = model.predict_proba(test_X)[:, 1]

        train_predictions = model.predict(train_X)
        predictions = model.predict(test_X)


        print(f'Train ROC AUC Score: {roc_auc_score(y_train, train_probs)}')
        print(f'Test ROC AUC  Score: {roc_auc_score(y_test, probs)}')
        


        results['recall'] = recall_score(y_test, predictions, average="binary", pos_label='Fake')
        results['precision'] = precision_score(y_test, predictions, average="binary", pos_label='Fake')
        results['roc'] = roc_auc_score(y_test, probs)
        print(results)
        return y_test,predictions