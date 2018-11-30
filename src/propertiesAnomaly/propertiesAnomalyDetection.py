import networkx as nx
import scipy as sp
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn import metrics
import random
import pickle
from src.properties.nodeProperties.computeAll import compute_vertex_properties

def computeAllProperties(G):
    print(G.number_of_edges)
    print(G.number_of_nodes)
    print("shape of G's adjacency matrix: " + str(nx.adjacency_matrix(G).shape))
    M = []
    for node in G.nodes():
        p = compute_vertex_properties(G, node, properties = ['degree', 'ego_edge_count', 'ego_triangle_count', 'ego_singular_value', 'clustering_coefficient'])
        print(p)
        M.append(p)
    print("shape of M: ", len(M))
    return M

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
        #print("testIndexes: ", str(testIndexes))

        X = data.iloc[data.index.isin(testIndexes)]
        print("shape of X: " + str(X.shape))

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

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
M = computeAllProperties(convert(politicalSet))
detectIsolation(M)
