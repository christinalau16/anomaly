import networkx as nx
import scipy as sp
import matplotlib.pyplot as plt

def computeTrun(G):
    #Calculate Frobenius norm for matrix A
    normA = sp.linalg.norm(nx.adjacency_matrix(G).toarray())

    kArr = []
    errorArr = []

    k = 10
    error = 1

    #Express graph as graph adjacency matrix / SciPy sparse matrix
    A = sp.sparse.csc_matrix(nx.to_scipy_sparse_matrix(G)).asfptype()

    #while(k <=100):
    while (error > 0.1):
        print("k: ", k)
        #Find greatest n singular values of matrix
        U, S, VT = sp.sparse.linalg.svds(A, k)
        B = U.dot(sp.sparse.diags(S).dot(VT))

        #Calculate Frobenius norm for matrix A-B
        normAB = sp.linalg.norm(A - B)

        error = normAB / normA
        if (error <= 0.1):
            finalU = U

        kArr.append(k)
        errorArr.append(error)

        k = k + 20

    plotkError(kArr, errorArr)
    return finalU

def plotkError(kArr, errorArr):
    #Draw loglog graph
    plt.plot(kArr, errorArr, linestyle='None',
           marker='x', markeredgecolor='blue')
    plt.title("Linear-linear graph, political set")
    plt.ylabel("Reconstruction Error")
    plt.xlabel("k")
    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
computeTrun(convert(politicalSet))
