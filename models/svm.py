from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

class SVM:
    def __init__(self,cv=10):
        self.cv=cv
    def train(self,X,y):
       
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        svc=LinearSVC(dual=False)
        scores = cross_val_score(svc, X, y, cv=self.cv, scoring='accuracy')
        print(scores)
        print(scores.mean())
        C_range=list(range(1,26))
        acc_score=[]
        for c in C_range:
            svc = LinearSVC(dual=False, C=c)
            scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
            acc_score.append(scores.mean())
            
        C_values=list(range(1,26))
        fig = go.Figure(data=go.Scatter(x=C_values, y=acc_score))
        fig.update_layout(xaxis_title='Value of C for SVC',
                        yaxis_title='Cross Validated Accuracy', template='plotly_white',xaxis = dict(dtick = 1))
        fig.show()

    
