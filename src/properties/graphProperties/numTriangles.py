import networkx as nx

def numTriangles(G):
    #each triangle is counted three times, once at each node
    num = sum(nx.triangles(G).values())/3
    print("Number of triangles: " + str(num))


def numTriangles2():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])
    num = sum(nx.triangles(G).values())/3
    print(nx.triangles(G))
    print("Number of triangles: " + str(num))

from src.preprocess.csvToGraph import convert
emailSet = 'data/datasetEmail/datasetEmail_final.csv'
politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
graph = convert(emailSet)
#numTriangles(graph)
numTriangles2()
