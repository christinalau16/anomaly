import networkx as nx
import scipy as sp
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn import metrics
import random
from src.properties.nodeProperties.computeAll import compute_vertex_properties

def computeAllProperties(G):
    M = []
    for node in G.nodes():
        p = compute_vertex_properties(G, node)
        M.append(p)
    return M

def detectIsolation(M):
    data = pd.DataFrame(M)

    outlierIndexes = random.sample(range(758, 1489), 30)
    print("outlierIndexes: ", str(outlierIndexes))
    testIndexes = [i for i in range(0, 758)] + outlierIndexes

    X_train = data
    X_test = data.iloc[data.index.isin(testIndexes)]
    print("shape of X_test: " + str(X_test.shape))

    y_test=[]
    for i in range(len(X_test.index)):
        if i < 758:
            y_test.append(1)
        else:
            y_test.append(-1)

    # 0.038 = 30/(758 + 30)
    clf= IsolationForest(contamination=float(0.038))
    clf.fit(X_train)
    y_pred = clf.predict(X_test)

    count = 1
    for j in range(len(y_pred)):
        if y_pred[j] == -1:
            print("found anomaly " + str(count) + " at index " + str(testIndexes[j]))
            count = count + 1

    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    print("accuracy score: " + str(accuracy_score))

    return accuracy_score

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
M = computeAllProperties(convert(politicalSet))
detectIsolation(M)
