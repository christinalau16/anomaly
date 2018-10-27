import networkx as nx
import matplotlib.pyplot as plt

#Create an empty graph with no nodes and no edges
G = nx.Graph()

#Nodes
G.add_node(1) #add one node at a time
G.add_nodes_from([2, 3]) #add a list of nodes

H = nx.path_graph(10)
G.add_nodes_from(H) #G now contains the nodes of H as nodes of G
G.add_node(H) #G now contains H as a node

#Edges
G.add_edge(1, 2) #G can be grown by adding one edge at a time
e = (2, 3)
G.add_edge(*e) #unpack edge tuple*

G.add_edges_from([(1, 2), (1, 3)]) #Adding a list of edges
G.add_edges_from(H.edges) #Adding an ebunch of edges

#There are no complaints when adding existing nodes or edges
G.clear()
G.add_edges_from([(1, 2), (1, 3)])
G.add_node(1)
G.add_edge(1, 2)
G.add_node("spam")        # adds node "spam"
G.add_nodes_from("spam")  # adds 4 nodes: 's', 'p', 'a', 'm'
G.add_edge(3, 'm')

print(G.number_of_nodes())
print(G.number_of_edges())

#Nodes, edges, adjacencies, degrees
print(list(G.nodes))
print(list(G.edges))
print(list(G.adj[1]))
print(list(G.adj[3]))
print(G.degree[1])

#Report the edges and degree from a subset of all nodes using an nbunch
print(G.edges([2, 3]))
print(G.degree([2, 3]))

#Remove nodes and edges
G.remove_node(2)
G.remove_nodes_from("spam")
print(list(G.nodes))
G.remove_edge(1, 3)
print(list(G.edges))

#When creating a graph structure by instantiating one of the graph classes
#can specify data in several formats.
G.add_edge(1, 2)
H = nx.DiGraph(G) #create a Digraph using the connections from G
print(list(H.edges()))

edgelist = [(0, 1), (1, 2), (2, 3)] #create graph using edgelist
H = nx.Graph(edgelist)
print(list(H.edges()))

#To obtain a more traditional graph with integer labels,
#use convert_node_labels_to_integers()

#Accessing edges and neighbors using subscript notation
print("Accessing edges and neighbors")
print("nodes:", list(G.nodes))
print("edges:", list(G.edges))
print("G.adj[1]: ", list(G.adj[1])) #G[1] same as G.adj[1]
print("G[1]:", list(G[1]))
#get/set the attributes of edge using subscript notation, if it already exists!
G.add_edge(1, 3)
G[1][3]['color'] = "blue"
G.edges[1, 2]['color'] = "red"
print(G[1][2]['color'])

#Fast examination of all (node, adjacency) pairs is acheived using
#G.adjacency() or G.adh.items()
#Note that for undirected graphs, adgancency iteration sees each edge twice
print("Fast examination with adjacency")
FG = nx.Graph()
FG.add_weighted_edges_from([(1, 2, 0.125),
  (1, 3, 0.75),
  (2, 4, 1.2),
  (3, 4, 0.375)])
for n, nbrs in FG.adj.items():
    for nbr, eattr in nbrs.items():
        wt = eattr['weight']
        if wt < 0.5:
            print('(%d, %d, %.3f)' % (n, nbr, wt))

print("Fast examination of all edges")
for (u, v, wt) in FG.edges.data('weight'):
    if wt < 0.5: print('(%d, %d, %.3f)' % (u, v, wt))

#Adding attributes to graphs, nodes, and edges
print("Adding attributes to graphs")
G = nx.Graph(day = "Friday")
print(G.graph)
G.graph['day'] = "Monday" #Modify attributes
print(G.graph)

print("Adding attibutes to nodes")
G.add_node(1, time='5pm')
G.add_nodes_from([3], time = '2pm')
print(G.nodes[1])
G.nodes[1]['room'] = 714
print(G.nodes.data())

print("Adding attributes to edges")
G.add_edge(1, 2, weight=4.7)
G.add_edges_from([(3, 4), (4, 5)], color = 'red')
G.add_edges_from([(1, 2, {'color:': 'blue'}), (2, 3, {'weight': 8})])
print(G.edges.data())
G[1][2]['weight'] = 2.3
G.edges[3, 4]['weight'] = 6.2
print(G.edges.data())

#Directed graphs
print("Directed graphs")
DG = nx.DiGraph()
DG.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75)])
#degree = in_degree+out_degree
print("out_degree: ", DG.out_degree(1, weight = 'weight'))
print("degree: ", DG.degree(1, weight = 'weight'))
#directed version of neighbors equivalent to successors
print("successors: ", list(DG.successors(1)))
print("neighbors: ", list(DG.successors(1)))
#convert G to undirected graph
H = nx.Graph(G)

#Multigraphs allow multiple edges between any pair of nodes
print("Multigraphs")
MG = nx.MultiGraph()
MG.add_weighted_edges_from([(1, 2, 0.5), (1, 2, 0.75), (2, 3, 0.5)])
print(dict(MG.degree(weight='weight')))

GG = nx.Graph()
for n, nbrs in MG.adjacency():
    print("n: ", n)
    print("nbrs: ", nbrs)
    for nbr, edict in nbrs.items():
        print("nbr: ", nbr)
        print("edict: ", edict)
        minvalue = min([d['weight'] for d in edict.values()])
        GG.add_edge(n, nbr, weight = minvalue)
print(nx.shortest_path(GG, 1, 3))

G = nx.petersen_graph()
plt.subplot(121)
nx.draw(G, with_labels=True, font_weight='bold')

plt.subplot(122)
nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight = 'bold')
plt.show()
plt.savefig("testMatplotlibImage")
