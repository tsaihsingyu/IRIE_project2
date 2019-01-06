import json, sys, re, os
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
from argparse import ArgumentParser

def construct_arg():
    parser = ArgumentParser()
    parser.add_argument("--node_file", help="path to node label file", dest="mname", default=None)
    
    args = parser.parse_args()  
    return args

args = construct_arg()

mname = args.mname

def convert_nodes(nodes_j):
    # convert the node's representation format
    nodes = []
    for node_j in nodes_j: # node in json format
        # node = [the starting token index of a node, the label of the node]
        node = np.array([node_j[0][0], list(node_j[1])[0]]) 
        nodes.append(node)
    nodes = np.array(nodes)

    return nodes

def convert_edges(edges_j):
    # convert the edge's representation format
    edges = []
    for edge_j in edges_j: # edge in json format
        # edge = [the starting token index of the first node, the starting token index of the second node, the label of the edge]
        edge = np.array([edge_j[0][0], edge_j[1][0], list(edge_j[2])[0]])
        edges.append(edge)

    return np.array(edges)

def get_edges(file="data/test.json", mypredict=None):
    X, y = [], []
    with open(file) as fin:
        sentence_x, sentence_y = [], []
        # for task 1, use the self prediction of node
        if(mypredict != None):
            node_predict = np.load(mypredict)

        for sid, line in enumerate(fin):
            data = json.loads(line) # a single sentence
            edges = convert_edges(np.array(data['edges'])) # convert edges to the desired format
            nodes = convert_nodes(np.array(data['nodes'])) # convert nodes to the desired format
            for edge in edges:
                node1 = np.where(nodes[:, ] == edge[0])[0][0]
                node2 = np.where(nodes[:, ] == edge[1])[0][0]
                
                y.append(edge[-1]) # edge true label

                if(mypredict == None):
                    X.append(np.array([nodes[node1][1], nodes[node2][1], node1, node2, sid]))
                else:
                    X.append(np.array([node_predict[sid][node1], node_predict[sid][node2], node1, node2, sid]))
                
    return np.array(X), np.array(y)


def find_v(nid, line_r):
    vset = set()
    line_r = line_r[np.logical_or(line_r[:, 3]==nid, line_r[:, 4]==nid)]
    for line in line_r:
        if(line[1] != "value" and line[2] != "value"):
            continue
        vset.add(line[3])
        vset.add(line[4])

    if(nid in vset):
        vset.remove(nid)

    return vset

def test_constraint(test_relations, y_predict):
    max_id = np.max(test_relations[:, -1].astype(int)) # maximum sentence id
    y_predict = np.array(y_predict).reshape(-1, 1)
    test_relations = np.concatenate([y_predict, test_relations], axis=-1)

    for rid, relations in enumerate(test_relations):
        # only nodes with the "analogy" label will need to test the constraint
        if(relations[0] != "analogy" or (relations[1] == "value" and relations[2] == "value")):
            continue
        line_r = test_relations[test_relations[:, -1]==relations[-1]] # the same sentence
        line_r = line_r[line_r[:, 0] == 'fact'] # nodes with "fact" label in the sentence
        
        set1 = find_v(relations[3], line_r)
        set2 = find_v(relations[4], line_r)
        if(len(set1.intersection(set2)) > 0):
            test_relations[rid][0] = "equivalence"

    return test_relations[:, 0]

def test_edges(X_test, y_true):
    y_predict = []
    # load the statistical result (task 2)
    r_statistics = np.load("model/r_statistics.npy").item()

    for x in X_test:
        key = x[0]+"="+x[1]
        if(key in r_statistics):
            ################################################
            # if(key == "quant=quant"):
                # y_predict.append("analogy")
            # else:
                # y_predict.append(r_statistics[key])
            ################################################################
            # The above and below method will result in the same performance            
            ################################################################
            if(key == "manner=manner"):
                y_predict.append("equivalence")
            elif(x[0] != x[1] and (x[0] == "value" or x[1] == "value")):
                y_predict.append("fact")
            else:
                y_predict.append("analogy")
            ################################################
        elif(x[0] != x[1] and (x[0] == "value" or x[1] == "value")):
            y_predict.append("fact")
        else:
            y_predict.append("analogy")

    y_predict = test_constraint(X_test, y_predict)


    # calculate evaluation matrix
    ave = "macro"
    acc = accuracy_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict, average=None)
    recall = recall_score(y_true, y_predict, average=None)
    f1 = f1_score(y_true, y_predict, average=None)
    
    precision = np.concatenate([precision, [precision_score(y_true, y_predict, average=ave)]])
    recall = np.concatenate([recall, [recall_score(y_true, y_predict, average=ave)]])
    f1 = np.concatenate([f1, [f1_score(y_true, y_predict, average=ave)]])

    print(np.unique(y_true))
    print("acc:", acc)
    print("precision:", precision)
    print("recall:", recall)
    print("f1:", f1)

if __name__ == '__main__':
    X_test, y_true = get_edges(mypredict=mname) # for task 1
    # X_test, y_true = get_edges() # for task 2
    
    test_edges(X_test, y_true)
    