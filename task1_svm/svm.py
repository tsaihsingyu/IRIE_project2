import json
#import xml.etree.ElementTree as ET
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from nltk import pos_tag
import numpy as np

# Nodes (18 kinds of labels)
NODE = ['value', 'agent', 'condition', 'theme', 'theme_mod', 'quant_mod', 'co_quant', 'null', 'location', 'whole', 'source', 'reference_time', 'quant', 'manner', 'time', 'cause', '+', '-']
# Edges (3 kinds of labels)
EDGE = ['equivalence', 'fact', 'analogy']
# POS tagging (40 kinds of labels))
PosTag = []


class DataManager:
	# Data for a sentence
	def __init__(self):
		self.tokens = []
		self.nodes = []
		self.edges = []

	# Save data for each sentence
	def insertData(self,data):
		self.tokens = data['tokens']
		for node in data['nodes']:
			temp = []
			temp.append(node[0])
			k = list(node[1].items())[0][0]
			temp.append(NODE.index(k))
			self.nodes.append(temp)
		for edge in data['edges']:
			temp = []
			temp.append(edge[0])
			temp.append(edge[1])
			k = list(edge[2].items())[0][0]
			temp.append(EDGE.index(k))
			self.edges.append(temp)

# Find the POS of each tokens in a sentence
def POS(tokens):
	tags = []
	word_tags = pos_tag(tokens)
	for (a,b) in word_tags:
		if b not in PosTag:
			PosTag.append(b)
		tags.append(PosTag.index(b))
	return tags

# Count the POS of each tokens in a sentence
def Count(tags):
	n = [0]*40 #len(PosTag)
	for i in tags:
		n[i] += 1
	return n


if __name__ == '__main__':
	
	# Load training data
	train_data = []
	filename = 'data/train.json'
	infile = open(filename, 'r', encoding='utf-8')
	for line in infile:
		jsonData = json.loads(line)
		temp = DataManager() 
		temp.insertData(jsonData)
		train_data.append(temp)
	infile.close()

	# Load testing data
	test_data = []
	filename = 'data/test.json'
	infile = open(filename, 'r', encoding='utf-8')
	for line in infile:
		jsonData = json.loads(line)
		temp = DataManager() 
		temp.insertData(jsonData)
		test_data.append(temp)
	infile.close()

	# Get POS data for training SVM model
	X_train = []
	Y_train = []
	for data in train_data:
		tags = POS(data.tokens)
		for k in data.nodes:
			phrase = []
			for i in range(k[0][0], k[0][1]):			
				phrase.append(data.tokens[i])
			t = tags[k[0][0]:k[0][1]]
			merge = list(Count(t))
			X_train.append(merge)
			Y_train.append(k[1])

	# Train SVM model
	svm = SVC(kernel='linear', probability=True) #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
	svm.fit(X_train,Y_train)

	# Get POS data to predict the testing data by SVM model
	X_test = []
	Y_test = []
	Y_pred = []
	prediction = []
	for data in test_data:
		tags = POS(data.tokens)
		result = []
		for k in data.nodes:
			phrase = []
			for i in range(k[0][0], k[0][1]):
				phrase.append(data.tokens[i])
			t = tags[k[0][0]:k[0][1]]
			merge = list(Count(t))
			X_test.append(merge)
			pred = svm.predict([merge])[0]
			Y_pred.append(pred)
			result.append(NODE[pred])
			Y_test.append(k[1])
		prediction.append(result)

	# Save result of prediction which will be used to predict edges later
	np.save("y_predict.npy", np.array(prediction))

	# Output the result of node classification
	print(classification_report(Y_test, Y_pred, labels=list(range(18)), target_names=NODE))
	print("Accuracy =", accuracy_score(Y_test, Y_pred))


