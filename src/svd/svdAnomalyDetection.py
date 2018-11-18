import networkx as nx
import scipy as sp
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn import metrics
import matplotlib.pyplot as plt
import random

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

def detectIsolationGeneric(G):
    #Calculate Frobenius norm for matrix A
    normA = sp.linalg.norm(nx.adjacency_matrix(G).toarray())

    kArr = []
    accuracyArr = []

    k = 20
    error = 1

    #Express graph as graph adjacency matrix / SciPy sparse matrix
    A = sp.sparse.csc_matrix(nx.to_scipy_sparse_matrix(G)).asfptype()

    while (error > 0.1):
        #Find greatest n singular values of matrix
        U, S, VT = sp.sparse.linalg.svds(A, k)
        B = U.dot(sp.sparse.diags(S).dot(VT))

        #Calculate Frobenius norm for matrix A-B
        normAB = sp.linalg.norm(A - B)
        error = normAB / normA

        print("k: " + str(k))
        accuracy = detectIsolation(U)

        kArr.append(k)
        accuracyArr.append(accuracy)

        k = k + 50

    plotkError(kArr, accuracyArr)

def plotkError(kArr, accuracyArr):
    plt.plot(kArr, accuracyArr, linestyle='None',
           marker='x', markeredgecolor='blue')
    plt.title("Linear-linear graph, classification accuracy vs. k, political set")
    plt.ylabel("Classification Accuracy")
    plt.xlabel("k")
    plt.show()

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
from src.svd.computeTruncated import computeTrun
M = computeTrun(convert(politicalSet))
#detectIsolation(M)
detectIsolationGeneric(convert(politicalSet))
