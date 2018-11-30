import networkx as nx
import scipy as sp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle

def classifyData(M):
    lookup_classify = pickle.load(open('data/datasetPolitical/blogs_classify.pkl', 'rb'))

    data = pd.DataFrame(M)
    new_col = []
    for i in range(data.shape[0]):
        new_col.append(lookup_classify[i])
    data.insert(loc=0, column = 'y', value=new_col)

    X = data.iloc[:,1:]
    y = data.iloc[:,0]

    accuracyArr = []
    for i in range(20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

        classifier = LogisticRegression(random_state=i)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy_score = metrics.accuracy_score(y_test, y_pred)
        print("accuracy score: " + str(accuracy_score))
        accuracyArr.append(accuracy_score)
    averageAccuracy = sum(accuracyArr, 0.0) / len(accuracyArr)
    print("average accuracy score: " + str(averageAccuracy))
    return averageAccuracy

def classifyDataGeneric(G):
    #Calculate Frobenius norm for matrix A
    normA = sp.linalg.norm(nx.adjacency_matrix(G).toarray())

    kArr = []
    accuracyArr = []

    k = 1
    error = 1

    #Express graph as graph adjacency matrix / SciPy sparse matrix
    A = sp.sparse.csc_matrix(nx.to_scipy_sparse_matrix(G)).asfptype()
    print("shape of A: " + str(A.shape))

    #while (error > 0.1):
    while(k <= 50):
        #Find greatest n singular values of matrix
        U, S, VT = sp.sparse.linalg.svds(A, k)
        B = U.dot(sp.sparse.diags(S).dot(VT))

        #Calculate Frobenius norm for matrix A-B
        normAB = sp.linalg.norm(A - B)
        error = normAB / normA

        print("k: " + str(k))
        accuracy = classifyData(U)

        kArr.append(k)
        accuracyArr.append(accuracy)

        k = k + 1

    plotkError(kArr, accuracyArr)

def plotkError(kArr, accuracyArr):
    plt.plot(kArr, accuracyArr, linestyle='None',
           marker='x', markeredgecolor='blue')
    plt.title("k={1...50}, Linear-linear graph, average classification accuracy vs. k, political set")
    plt.ylabel("Classification Accuracy")
    plt.xlabel("k")
    plt.show()

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
#from src.svd.computeTruncated import computeTrun
#M = computeTrun(convert(politicalSet))
#classifyData(M)
classifyDataGeneric(convert(politicalSet))
