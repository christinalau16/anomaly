import networkx as nx
import scipy as sp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

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

    return accuracy_score

def classifyDataGeneric(G):
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
        accuracy = classifyData(U)

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
    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
#from src.svd.computeTruncated import computeTrun
#M = computeTrun(convert(politicalSet))
#classifyData(M)
classifyDataGeneric(convert(politicalSet))
