import numpy as np
import pandas as pd
import random
from collections import defaultdict
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import networkx as nx

#K = number of dimensions
#node_sets = picked using select_node_sets()
#node = node id to compute embedding for
#adj_list = a dictionary of dictionary such that adj_list[source][destination] = edge_weight. Using defaultdict may be helpful.
def compute_node_embedding(K, node_sets, node, adj_list):
    embedding = [0 for x in range(K)] #vector of K zeros

    for neighbor in adj_list[node]:
        for i in range(K):
            if neighbor in node_sets[i]:
                #print("found neighbor in node_sets[i]")
                embedding[i] += adj_list[node][neighbor]
    return embedding

#distributions = N*L dim array where N=#nodes, L=#distributions.
#dist_weights = L dim array, a probability distribution over the L distributions above. With probability dist_weight[d], distributions[:,d] is chosen and nodes are sampled according to it.
def select_node_sets(distributions, dist_weights, K):
    N = len(distributions) #number of nodes
    #L = len(distributions[0]) #number of distributions
    L = 1

    node_sets = []
    random.seed(0)
    for i in range(K):
        #d ~ sample an index from 1…L according to dist_weights
        population = range(L)
        d = random.choices(population, dist_weights)[0]
        #d = random.choices(population)[0]

        node_set = set()
        for node in range(N):
            #with probability distributions[node,d]:
            r = random.uniform(0, 1)
            if r <= distributions[node]:
            #if r <= distributions[node][d]:
                node_set.add(node)
        node_sets.append(node_set)

    print("node_sets: " + str(node_sets))
    return node_sets

def compute_adj_list(df):
    graph = defaultdict(dict)
    for index, row in df.iterrows():
        src = row['source']
        dest = row['target']
        weight = row['weight']
        graph[src][dest] = weight
        graph[dest][src] = weight
    print(graph)
    print(len(graph))
    return graph

def compute_all_nodes(infile):
    df = pd.read_csv(infile)
    df.columns=['source', 'target', 'weight']

    adj_list = compute_adj_list(df)
    N = len(adj_list)

    kArr = []
    pArr = []
    accuracyArr = []

    K = 50
    p_uniform = 0.1
    distributions_uniform = [p_uniform for x in range(N)]
    D_degree = max([len(adj_list[x]) for x in range(N)]) + 1
    distributions_degree = [(len(adj_list[x])/D_degree) for x in range(N)]
    distributions = distributions_uniform
    #distributions = np.column_stack((distributions_uniform, distributions_degree)).tolist()
    #distributions = np.column_stack((distributions_uniform)).tolist()
    #distributions = np.transpose(distributions_uniform).tolist()
    dist_weights = [1]

    #while(K <= 1200):
    while(p_uniform <= 1):
        #print("K: " + str(K))
        print("p_uniform: " + str(p_uniform))
        distributions_uniform = [p_uniform for x in range(N)]
        distributions = distributions_uniform

        node_sets = select_node_sets(distributions, dist_weights, K)

        M = []
        for node in range(N):
            embedding = compute_node_embedding(K, node_sets, node, adj_list)
            M.append(embedding)

        accuracy = classifyData(M)

        #kArr.append(K)
        pArr.append(p_uniform)
        accuracyArr.append(accuracy)

        #K = 2 * K
        p_uniform = p_uniform + 0.01

    #plotkError(kArr, accuracyArr)
    plotkError(pArr, accuracyArr)

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

def plotkError(kArr, accuracyArr):
    plt.plot(kArr, accuracyArr, linestyle='None',
           marker='x', markeredgecolor='blue')
    plt.title("p increased log, Linear-linear graph, average classification accuracy vs. k, political")
    plt.ylabel("Classification Accuracy")
    plt.xlabel("p")
    plt.xscale("log")
    plt.show()

politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
embedded_all = compute_all_nodes(politicalSet)
