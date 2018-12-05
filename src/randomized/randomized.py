import numpy as np
import pandas as pd
import random
from collections import defaultdict
from sklearn.ensemble import IsolationForest
import pickle

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
    L = len(distributions[0]) #number of distributions

    node_sets = []
    for i in range(K):
        random.seed(i)

        #d ~ sample an index from 1…L according to dist_weights
        population = range(L)
        d = random.choices(population, dist_weights)[0]
        print("d: " + str(d))

        node_set = set()
        for node in range(N):
            #with probability distributions[node,d]:
            r = random.uniform(0, 1)
            if r <= distributions[node][d]:
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
    return graph

def compute_all_nodes(infile):
    df = pd.read_csv(infile)
    df.columns=['source', 'target', 'weight']

    adj_list = compute_adj_list(df)
    N = len(adj_list)

    K = 4
    p_uniform = 0.1
    distributions_uniform = [p_uniform for x in range(N)]
    D_degree = max([len(adj_list[x]) for x in range(N)]) + 1
    distributions_degree = [(len(adj_list[x])/D_degree) for x in range(N)]
    distributions = np.column_stack((distributions_uniform, distributions_degree)).tolist()
    dist_weights = [1, 1]
    node_sets = select_node_sets(distributions, dist_weights, K)

    node = 0
    embedding = compute_node_embedding(K, node_sets, node, adj_list)
    print("embedding for node " + str(node) + ": "+ str(embedding))

    node = 16
    embedding = compute_node_embedding(K, node_sets, node, adj_list)
    print("embedding for node " + str(node) + ": "+ str(embedding))

    node = 1214
    embedding = compute_node_embedding(K, node_sets, node, adj_list)
    print("embedding for node " + str(node) + ": "+ str(embedding))

    M = []
    for node in range(N):
        embedding = compute_node_embedding(K, node_sets, node, adj_list)
        M.append(embedding)
    return M

def sortFirst(val):
    return val[0]
    
def detectIsolation(M):
    N = 758
    n = 30

    lookup_classify = pickle.load(open('data/datasetPolitical/blogs_classify.pkl', 'rb'))

    data = pd.DataFrame(M)

    accuracyArr = []

    for j in range(20):
        random.seed(j)
        first = [k for k, v in lookup_classify.items() if v == 0]
        second = [k for k, v in lookup_classify.items() if v == 1]
        outlierIndexes = random.sample(second, n)
        print("outlierIndexes: ", str(outlierIndexes))
        testIndexes = first + outlierIndexes
        print("testIndexes: ", str(testIndexes))

        X = data.iloc[data.index.isin(testIndexes)]
        print("shape of X: " + str(X.shape))

        contamination_value = float(n) / (N+n)
        print("contamination value: " + str(contamination_value))
        clf= IsolationForest(contamination=contamination_value)
        clf.fit(X)
        anomaly_score = clf.decision_function(X)

        indexed_anomaly_score = []
        for i in range(len(anomaly_score)):
            indexed_anomaly_score.append([anomaly_score[i], testIndexes[i]])
        #print("indexed anomaly score: ", indexed_anomaly_score)
        indexed_anomaly_score.sort(key = sortFirst)
        #print("indexed anomaly score sorted: ", indexed_anomaly_score)

        count = 0
        for i in range(n):
            if indexed_anomaly_score[i][1] in outlierIndexes:
                print("found correct anomaly index: ", str(indexed_anomaly_score[i][1]))
                count = count + 1
        print("count: " + str(count))
        accuracy_score = float(count) / n
        print("accuracy score: " + str(accuracy_score))
        accuracyArr.append(accuracy_score)

    averageAccuracy = sum(accuracyArr, 0.0) / len(accuracyArr)
    print("average accuracy score: " + str(averageAccuracy))
    return averageAccuracy

politicalSet = 'data/datasetPolitical/datasetPolitical_final.csv'
embedded_all = compute_all_nodes(politicalSet)
accuracy = detectIsolation(embedded_all)
print(accuracy)
