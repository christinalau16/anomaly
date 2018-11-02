import networkx as nx
import pandas as pd

def convert(infile):
    df = pd.read_csv(infile)
    df.columns=['source', 'target', 'weight']
    G = nx.from_pandas_edgelist(df, edge_attr='weight')
    #print(df.head())
    #print(G.number_of_nodes())
    #print(G.number_of_edges())
    return G

convert('data/datasetEmail/datasetEmail_final.csv')
#convert('data/datasetPolitical/datasetPolitical_final.csv')
