import networkx as nx
import scipy as sp

def largestSingularValue(G, n):
    EG = nx.ego_graph(G, n)
    #Express graph as graph adjacency matrix / SciPy sparse matrix
    M = sp.sparse.csc_matrix(nx.to_scipy_sparse_matrix(EG)).asfptype()
    #Find greatest singular value of matrix
    s = sp.sparse.linalg.svds(M, 1)[1]
    print("largest singular value of adjacency of ego-network: " + str(s))

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
largestSingularValue(convert(emailSet), 212)
