import json, sys, re, os
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
import nltk 
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.optimizers import RMSprop
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import datasets, svm, metrics,cross_validation, ensemble


def convert_nodes(nodes_j):
    nodes = []
    for node_j in nodes_j:
        node = np.array([node_j[0][0],node_j[0][1],list(node_j[1])[0]])
        nodes.append(node)
    nodes = np.array(nodes)

    return nodes


def read_data(path="data/train.json"):
	Pos_data = []
	Nodes_data = []
	dic_pos = {}
	dic_ind = 0
	with open(path) as fin:
		for tid, line in enumerate(fin):
			temp_token = []
			data = json.loads(line)
			tokens = data['tokens']
			pos = nltk.pos_tag(tokens)
			for p in pos:
				if(p[1] not in dic_pos):
					dic_pos[p[1]]=dic_ind
					dic_ind+=1		
				temp_token.append(p[1])
			Pos_data.append(temp_token)
			nodes_j = data['nodes']
			nodes = convert_nodes(nodes_j)
			Nodes_data.append(nodes)
			
	Pos_data = np.array(Pos_data)
	Nodes_data = np.array(Nodes_data)
	
	return(Pos_data,Nodes_data,dic_pos)

def read_test(path="data/test.json"):
	Pos_data = []
	Nodes_data = []

	with open(path) as fin:
		for tid, line in enumerate(fin):
			temp_token = []
			data = json.loads(line)
			tokens = data['tokens']
			pos = nltk.pos_tag(tokens)
			for p in pos:	
				temp_token.append(p[1])
			Pos_data.append(temp_token)
			nodes_j = data['nodes']
			nodes = convert_nodes(nodes_j)
			Nodes_data.append(nodes)
			
	Pos_data = np.array(Pos_data)
	Nodes_data = np.array(Nodes_data)
	
	return(Pos_data,Nodes_data)

def preprocess(Pos_data,Nodes_data,dic_pos):
	All_Node_Emb=[] 
	All_Node_Ans=[]
	Sen_ind = []
	ind_ind = 0
	pos_dim = len(dic_pos)
	for i in range(len(Pos_data)):
		Sen_ind.append(ind_ind)
		for node in Nodes_data[i]:
			ind_ind+=1
			temp_embed = np.zeros(pos_dim)
			for rn in range(int(node[0]),int(node[1])):
				temp_embed[int(dic_pos[str(Pos_data[i][rn])])]=1
			All_Node_Emb.append(temp_embed)
			tar = np.zeros(3)
			if(node[2]=="manner"):
				tar[0]=1
			elif(node[2]=="value"):
				tar[1]=1
			else:
				tar[2]=1
			All_Node_Ans.append(tar)
			
	All_Node_Emb = np.array(All_Node_Emb)
	All_Node_Ans = np.array(All_Node_Ans)
	Sen_ind.append(ind_ind)

	return(All_Node_Emb,All_Node_Ans,Sen_ind)

if __name__ == '__main__':
	#讀train data檔中的token（並轉成pos_tag)以及nodes / 紀錄pos_tag種類的dictionary
	(Pos_data,Nodes_data,dic_pos) = read_data("data/train.json") 
	#讀test data檔中的token（並轉成pos_tag)以及nodes 
	(Pos_test,Nodes_test) = read_test("data/test.json") 
	#Preprocess 轉成one-hot encoding/紀錄sentence的起始index（Sen_ind)因node都拆掉來train
	(All_Node_Emb,All_Node_Ans,ALLSen_ind) = preprocess(Pos_data,Nodes_data,dic_pos) 
	(Test_Node_Emb,Test_Node_Ans,TestSen_ind) = preprocess(Pos_test,Nodes_test,dic_pos)	
	#放入Random Forest Classifier並預測結果
	forest = ensemble.RandomForestClassifier(n_estimators = 10,max_features=10)
	forest_fit = forest.fit(All_Node_Emb, All_Node_Ans)

	test_predicted = forest.predict(Test_Node_Emb)
	test_predicted = np.array(test_predicted)
	accuracy = metrics.accuracy_score(Test_Node_Ans, test_predicted)
	print(accuracy)
	Ans_res = np.argmax(test_predicted,axis=1)
	#將預測結果種類轉為node label / 並組回sentence
	res_sen = []
	for k in range(len(TestSen_ind)-1):
		temp_sen = []
		for w in range(TestSen_ind[k],TestSen_ind[k+1]):
			if(Ans_res[w]==0):
				temp_sen.append("manner")
			elif(Ans_res[w]==1):
				temp_sen.append("value")
			elif(Ans_res[w]==2):
				temp_sen.append("others")
		res_sen.append(temp_sen)
	#將test data 97句的node label預測結果存起來 交由task2 的Majority＋Unifacts做後續預測
	res_sen = np.array(res_sen)
	np.save("res_sen_rf.npy",res_sen)
	res_sen_1 = np.load("res_sen_rf.npy")
	