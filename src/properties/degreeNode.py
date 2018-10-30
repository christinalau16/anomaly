import networkx as nx

def testNode():
    DG = nx.DiGraph()
    DG.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75)])
    #degree = in_degree+out_degree
    print("out_degree: ", DG.out_degree(1, weight = 'weight'))
    print("in_degree: ", DG.in_degree(1, weight = 'weight'))

testNode()
