#Compute the top 50 singular values of graph
#Log-log rank plot singular values

import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import src.generate.generatePPM as genPPM

def computeTopSVal():
    n = 50
    nArr = []
    for i in range(n):
        nArr.append(n-i)

    #Generate graph
    G = genPPM.generate()
    #Express graph as graph adjacency matrix / SciPy sparse matrix
    M = sp.sparse.csc_matrix(nx.to_scipy_sparse_matrix(G)).asfptype()
    #Find greatest n singular values of matrix
    s = sp.sparse.linalg.svds(M, n)[1]

    #Draw loglog graph
    plt.plot(np.log10(nArr), np.log10(s), linestyle='None',
           marker='x', markeredgecolor='blue')
    plt.show()

def computeAllSVal():
    #Generate graph
    G = genPPM.generate()
    print(G.number_of_nodes())
    print(G.number_of_edges())

    n = G.number_of_nodes()
    nArr = []
    for i in range(n):
        nArr.append(i+1)
    print(nArr)

    #Express graph as graph adjacency matrix / Numpy matrix
    M = nx.to_numpy_matrix(G)
    #Find all singular values of matrix
    s = np.linalg.svd(M)[1]
    print(s)

    #Draw loglog graph
    plt.plot(np.log10(nArr), np.log10(s), linestyle='None',
           marker='x', markeredgecolor='blue')
    plt.show()

computeTopSVal()
