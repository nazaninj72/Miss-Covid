from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

import itertools
import matplotlib.pyplot as plt

import numpy as np

class DecisionTree:
    def __init__(self,RSEED=50,MAX_DEPTH=20):
        self.rseed=RSEED 
        self.max_depth=MAX_DEPTH

    def train(self,X,y):
        #hyperparameters
        
    
        
        indices = np.arange(len(X))
        train_idx, test_idx, y_train, y_test= train_test_split(indices, y,stratify = y, test_size=0.25, random_state=42)
        train_X = X[train_idx]
        test_X = X[test_idx]
        clf = DecisionTreeClassifier(random_state=self.rseed,max_depth=self.max_depth)
        clf=clf.fit(train_X, y_train)

        return clf,test_X, y_test,y_train,train_X


    def plot_confusion_matrix(self,cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Oranges):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.figure(figsize = (10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, size = 24)
        plt.colorbar(aspect=4)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, size = 14)
        plt.yticks(tick_marks, classes, size = 14)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        
        # Labeling the plot
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
            
        plt.grid(None)
        plt.tight_layout()
        plt.ylabel('Real label', size = 18)
        plt.xlabel('Predicted label', size = 18)


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

