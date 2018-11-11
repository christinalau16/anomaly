import networkx as nx
import scipy as sp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

def classifyData(M):
    data = pd.DataFrame(M)
    idx = 0
    new_col = (data.index < 758).astype(int)
    data.insert(loc=idx, column = 'y', value=new_col)

    X = data.iloc[:,1:]
    y = data.iloc[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    print("accuracy score: " + str(accuracy_score))

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("confusion matrix: ")
    print(confusion_matrix)

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
from src.svd.computeTruncated import computeTrun
M = computeTrun(convert(politicalSet))
classifyData(M)
