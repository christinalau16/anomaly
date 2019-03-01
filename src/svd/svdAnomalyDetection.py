import networkx as nx
import scipy as sp
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import random
import pickle

def sortFirst(val):
    return val[0]

def detectIsolation(M):
    N = 758
    n = 30

    lookup_classify = pickle.load(open('data/datasetPolitical/blogs_classify.pkl', 'rb'))
    print(lookup_classify)

    data = pd.DataFrame(M)

    accuracyArr = []

    for j in range(20):
        random.seed(j)
        first = [k for k, v in lookup_classify.items() if v == 0]
        second = [k for k, v in lookup_classify.items() if v == 1]
        outlierIndexes = random.sample(second, n)
        print("outlierIndexes: ", str(outlierIndexes))
        testIndexes = first + outlierIndexes
        print("testIndexes: ", str(testIndexes))

        X = data.iloc[data.index.isin(testIndexes)]
        print("shape of X: " + str(X.shape))

        """
        u1 = data.iloc[data.index.isin(testIndexes)][data.columns[0]]
        u2 = data.iloc[data.index.isin(testIndexes)][data.columns[1]]
        plt.plot(u1, u2, linestyle='None',
               marker='x', markeredgecolor='blue')

        O = data.iloc[data.index.isin(outlierIndexes)]
        o1 = data.iloc[data.index.isin(outlierIndexes)][data.columns[0]]
        o2 = data.iloc[data.index.isin(outlierIndexes)][data.columns[1]]
        plt.plot(o1, o2, linestyle='None',
               marker='x', markeredgecolor='red')
        plt.ylabel("u2")
        plt.xlabel("u1")
        plt.show()
        """

        contamination_value = float(n) / (N+n)
        print("contamination value: " + str(contamination_value))
        clf= IsolationForest(contamination=contamination_value)
        clf.fit(X)
        anomaly_score = clf.decision_function(X)

        indexed_anomaly_score = []
        for i in range(len(anomaly_score)):
            indexed_anomaly_score.append([anomaly_score[i], testIndexes[i]])
        #print("indexed anomaly score: ", indexed_anomaly_score)
        indexed_anomaly_score.sort(key = sortFirst)
        #print("indexed anomaly score sorted: ", indexed_anomaly_score)

        count = 0
        for i in range(n):
            if indexed_anomaly_score[i][1] in outlierIndexes:
                print("found correct anomaly index: ", str(indexed_anomaly_score[i][1]))
                count = count + 1
        print("count: " + str(count))
        accuracy_score = float(count) / n
        print("accuracy score: " + str(accuracy_score))
        accuracyArr.append(accuracy_score)

    averageAccuracy = sum(accuracyArr, 0.0) / len(accuracyArr)
    print("average accuracy score: " + str(averageAccuracy))
    return averageAccuracy

def detectIsolationGeneric(G):
    #Calculate Frobenius norm for matrix A
    normA = sp.linalg.norm(nx.adjacency_matrix(G).toarray())

    kArr = []
    accuracyArr = []

    k = 2
    error = 1

    lookup_classify = pickle.load(open('data/datasetPolitical/blogs_classify.pkl', 'rb'))
    first = [k for k, v in lookup_classify.items() if v == 0]
    second = [k for k, v in lookup_classify.items() if v == 1]

    #Express graph as graph adjacency matrix / SciPy sparse matrix
    A = sp.sparse.csc_matrix(nx.to_scipy_sparse_matrix(G)).asfptype()

    #while(k <= 2):
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

        k = k*2

    plotkError(kArr, accuracyArr)

def plotkError(kArr, accuracyArr):
    plt.plot(kArr, accuracyArr, linestyle='None',
           marker='x', markeredgecolor='blue')
    plt.title("k={1...50}, Linear-linear graph, anomaly accuracy vs. k, political set")
    plt.ylabel("Anomaly Accuracy")
    plt.xlabel("k")
    plt.show()

def printMatrix(infile):
    lookup_classify = pickle.load(open('data/datasetPolitical/blogs_classify.pkl', 'rb'))
    first = [k for k, v in lookup_classify.items() if v == 0]
    second = [k for k, v in lookup_classify.items() if v == 1]

    df = pd.read_csv(infile)
    df.columns=['source', 'target', 'weight']

    plt.plot(df[df.columns[0]], df[df.columns[1]], linestyle='None',
           marker='.', markeredgecolor='blue')
    plt.ylabel("col")
    plt.xlabel("row")
    plt.show()

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final1.csv'
detectIsolationGeneric(convert(politicalSet))
#printMatrix(politicalSet)
