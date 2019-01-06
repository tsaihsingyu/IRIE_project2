from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import _pickle as pk
from keras.utils.data_utils import Sequence
import json
import sys, argparse, os
import keras
import pickle
import readline
import numpy as np
import json
from keras.models import load_model
from keras.models import Sequential
from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout,Activation,CuDNNLSTM,CuDNNGRU, Bidirectional,Reshape,GaussianDropout,AlphaDropout
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
import keras.backend.tensorflow_backend as K
import tensorflow as tf
from keras.utils import plot_model
from sklearn.metrics import f1_score,precision_score,recall_score
class DataSequence(Sequence):
    def __init__(self,x_set,y_set,batch_size):
        self.batch_size = batch_size
        self.x,self.y=x_set,y_set
    def __len__(self):
        return len(self.y) // self.batch_size
    def __getitem__(self,idx):
        return self.x[idx*self.batch_size:(idx+1)*self.batch_size],self.y[idx*self.batch_size:(idx+1)*self.batch_size]
    def on_epoch_end(self):
        pass
    def __iter__(self):
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item

def ConvertData():
	subdir = ["subj01","subj02","subj03","subj04","subj05","subj06","subj07","subj08","subj09","subj10","subj11","subj12","subj13","subj14","subj15","subj16"]
	actionname = ["balance","climbladder","climbup","duck","hop","kick","leap","punch","run","stepback","stepfront","stepleft","stepright","twistleft","twistright","vault"]
	actiondict = {}
	FilList = ""
	for i in range(1,17):
		actiondict[actionname[i-1]] = i
	for subdirpath in subdir:
		wholepath = "./data/"+subdirpath
		subfileIn = os.listdir(wholepath)
		actionnum = np.zeros((18))
		for subsubfile in subfileIn:
			subff = subsubfile.split('.')
			actionclass = subff[0]
			try:
				actionclassID = actiondict[actionclass]
				actionnum[actionclassID]+=1
				print(subsubfile)
				FilePath = wholepath+"/"+subsubfile
				datain = ReadFileske(FilePath)
				NewPath = "./skeleton/%s_a%d_%d.pickle"%(subdirpath,actionclassID,actionnum[actionclassID])
				with open(NewPath,'wb') as wp:
					pickle.dump(datain,wp)
					print(NewPath,len(datain))
				FilList += NewPath+"\n"
			except:
				print(subsubfile)
	with open("./filelist.txt","w") as fp:
		fp.write(FilList)



def Model(X_data,Y_data,X_test,Y_test,Batchsize,epochh,tim,dim):
	print(X_data.shape)
	model = Sequential()
	model.add(Bidirectional(LSTM(units=16),batch_input_shape=(None,tim, dim)))
	#model.add(GaussianDropout(0.7))
	#model.add(Bidirectional(CuDNNLSTM(units=256,return_sequences=True)))
	#model.add(GaussianDropout(0.7))
	#model.add(Bidirectional(LSTM(units=16)))
	#model.add(GaussianDropout(0.7))
	#model.add(Dense(128))
	#model.add(Dense(64))
	#model.add(Dense(32))
	model.add(Dense(3))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])#adam or nadam
	model.summary()
	checkpointer = ModelCheckpoint(filepath="./M2_{epoch:003d}.hdf5", verbose=1, save_best_only=True,mode='max',monitor='val_acc')
	his = model.fit_generator(DataSequence(X_data,Y_data,Batchsize),steps_per_epoch=16959,epochs=epochh,validation_data=DataSequence(X_test,Y_test,Batchsize),validation_steps=1661,callbacks=[checkpointer])

def LoadData():
	Traindata_X = []
	Traindata_Y = []
	Testdata_X = []
	Testdata_Y = []
	with open('./data/train.json','r') as fp:
		for line in fp:
			jd = json.loads(line)
			keepdict = {}
			for nd in jd['nodes']:
				keey = nd[0][0]
				typ = list(nd[1].keys())[0]
				tlist = ['value', 'agent', 'condition', 'theme', 'theme_mod', 'quant_mod', 'co_quant', 'null', 'location', 'whole', 'source', 'reference_time', 'quant', 'manner', 'time', 'cause', '+', '-']
				for i in range(0,len(tlist)):
					if(typ==tlist[i]):
						keepdict[keey] = i
			for ed in jd['edges']:
				typ = list(ed[2].keys())[0]

				if(typ=='equivalence'):
					y = 0
				elif(typ=='fact'):
					y = 1
				elif(typ=='analogy'):
					y = 2
				ox1 = ed[0][0]
				ox2 = ed[1][0]
				Traindata_X.append([[keepdict[ox1],keepdict[ox2]]])
				Traindata_Y.append(y)
	with open('./data/test.json','r') as fp:
		for line in fp:
			jd = json.loads(line)
			keepdict = {}
			for nd in jd['nodes']:
				keey = nd[0][0]
				typ = list(nd[1].keys())[0]
				tlist = ['value', 'agent', 'condition', 'theme', 'theme_mod', 'quant_mod', 'co_quant', 'null', 'location', 'whole', 'source', 'reference_time', 'quant', 'manner', 'time', 'cause', '+', '-']
				for i in range(0,len(tlist)):
					if(typ==tlist[i]):
						keepdict[keey] = i
			for ed in jd['edges']:
				typ = list(ed[2].keys())[0]

				if(typ=='equivalence'):
					y = 0
				elif(typ=='fact'):
					y = 1
				elif(typ=='analogy'):
					y = 2
				ox1 = ed[0][0]
				ox2 = ed[1][0]
				Testdata_X.append([[keepdict[ox1],keepdict[ox2]]])
				Testdata_Y.append(y)
	return np.array(Traindata_X),np.array(Traindata_Y),np.array(Testdata_X),np.array(Testdata_Y)
		#coun = 0
		#for line in jdict:
		#	if(coun == 0 ):
		#		print(line["nodes"])
		#	coun += 1
def TestModel(modelpath):#測試model
	Traindata_X,Traindata_Y,Testdata_X,Testdata_Y = LoadData()
	model = load_model(modelpath)
	classes = model.predict_classes(Testdata_X)
	print("F1: ",f1_score(Testdata_Y,classes,average='macro'))
	print("Recall: ",recall_score(Testdata_Y,classes,average='macro'))
	print("Precision: ",precision_score(Testdata_Y,classes,average='macro'))

def TrainModel():#堆疊model架構
	Traindata_X,Traindata_Y,Testdata_X,Testdata_Y = LoadData()
	Traindata_Y = keras.utils.to_categorical(Traindata_Y,num_classes=3)
	Testdata_Y = keras.utils.to_categorical(Testdata_Y,num_classes=3)
	print(Traindata_X.shape,Traindata_Y.shape,Testdata_X.shape,Testdata_Y.shape)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	K.set_session(sess)
	Model(Traindata_X,Traindata_Y,Testdata_X,Testdata_Y,16,10,1,2)
def main():
	TestModel('./M2_004.hdf5')
if __name__ == "__main__":
	main()