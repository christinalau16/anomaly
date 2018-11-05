import networkx as nx
import scipy as sp

def compute_vertex_properties(graph, vertex, properties=['degree', 'ego_edge_count', 'ego_triangle_count', 'ego_singular_value', 'clustering_coefficient']):
    EG = nx.ego_graph(graph, vertex)
    if 'degree' in properties:
        print("degree: ", graph.degree(vertex, weight = 'weight'))
    if 'ego_edge_count' in properties:
        numEdges =  EG.number_of_edges()
        print("number of edges in ego-network: " + str(numEdges))
    if 'ego_triangle_count' in properties:
        #each triangle is counted three times, once at each node
        numTri = sum(nx.triangles(EG).values())/3
        print("number of triangles in ego-network: " + str(numTri))
    if 'ego_singular_value' in properties:
        #Express graph as graph adjacency matrix / SciPy sparse matrix
        M = sp.sparse.csc_matrix(nx.to_scipy_sparse_matrix(EG)).asfptype()
        #Find greatest singular value of matrix
        s = sp.sparse.linalg.svds(M, 1)[1]
        print("largest singular value of adjacency of ego-network: " + str(s[0]))
    if 'clustering_coefficient' in properties:
        c = nx.clustering(graph, vertex)
        print("local clustering coefficient: " + str(c))

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
compute_vertex_properties(convert(emailSet), 2)
