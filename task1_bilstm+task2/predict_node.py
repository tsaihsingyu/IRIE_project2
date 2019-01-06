import json, sys, re, os, itertools
import argparse
from argparse import ArgumentParser
import pickle
import numpy as np
from gensim.models import word2vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential, load_model
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
from keras.layers import Input, Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation, Masking, Add
from keras.layers import Multiply, Dropout, Concatenate, SpatialDropout1D, Dot
from keras.optimizers import Adam
from keras import callbacks
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from keras import regularizers
from keras.utils import plot_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

np.random.seed(1234)


w2v_name = 'model/w2v'
w2v_pos_name = 'model/w2v_pos'
mname = "model/model.h5"
max_seq = 100

def construct_arg():
    parser = ArgumentParser()
    parser.add_argument("--do_train", help="training or not", dest="train", default=0, type=int)
    parser.add_argument("--use_pos", help="using pos tagging or not", dest="use_pos", default=0, type=int)
    
    args = parser.parse_args()  
    return args

args = construct_arg()

def W2V(X_train, istrain=False, mname=w2v_name):
    # train a word2vec model using sentences in training data
    if(istrain):
        model = word2vec.Word2Vec(X_train, size=300, window=5, min_count=0, workers=8) # train a w2v model
        model.save(mname)
    else:
        model = word2vec.Word2Vec.load(mname) # load a w2v model
    return model

def get_data(file="data/train.json"):
    X, y = [], []
    with open(file) as fin:
        for tid, line in enumerate(fin):
            data = json.loads(line) # a single sentence
            tokens = np.array(data['tokens'])
            # token preprocessing
            for i in range(len(tokens)):
                tokens[i] = re.sub(r"\d", "1", tokens[i]) # convert all number to 1
                try:
                    tokens[i] = str(int(tokens[i].replace(",", ""))) # delete ',' in number
                except:
                    tokens[i] = tokens[i].replace(",", "，") # in order to write to a csv file
                if("\/" in tokens[i]):
                    try:
                        temp = tokens[i].split("\/")
                        tokens[i] = str(float(temp[0])/float(temp[1])) # convert to floating point number
                    except:
                        # print(tokens[i])
                        pass
            
            nodes_j = np.array(data['nodes']) # nodes in json format
            nodes = list(np.array([" ".join(tokens[m[0][0]:m[0][1]]) for m in nodes_j]).flatten()) # concat all nodes
            nodes = [re.sub(r"\$ ", r"", n) for n in nodes] # delete "$" 
            labels = [list(m[1].keys())[0] for m in nodes_j] # concat all labels
 
            X.append(nodes)
            y.append(labels)

    return X, y


def to_categorical(sequences, categories):
    # convert features to one-hot encoding; for node's label 
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        for i in range(max_seq-len(s)): # padding part
            cats.append(np.zeros(categories))
            # index 0 states for "-PAD-" 
            cats[-1][0] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        # in order to get the accuracy without "-PAD-"
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy

def train_model(seq_index, pos_index, y_tags, word_vectors, pos_vectors, word2index, pos2index, tag2index, istrain=True):
    # let all sentence in the same length
    X_train = pad_sequences(seq_index, maxlen=max_seq, padding="post", value=-1.)
    
    # get the embedding dimension
    word_dim = word_vectors["he"].shape[0]

    # create embedding vectors
    embedding_words = np.zeros((len(word2index)+1, word_dim))
    
    for word, ind in word2index.items():
        if(word in word_vectors):
            embedding_words[ind] = word_vectors[word]
        else: # OOV
            embedding_words[ind] = np.random.uniform(1e-3, 1e-2, size=(word_dim,))


    inputs = Input(shape=(max_seq,))

    # mask the "-PAD-" part 
    mask = Masking(mask_value=-1.)(inputs)
    embedding = Embedding(len(word2index)+1,
                                 output_dim=word_dim,
                                 trainable=True,
                                 weights=[embedding_words])(mask)
    embedding = SpatialDropout1D(0.2)(embedding)

    x = Bidirectional(LSTM(64, activation="tanh", dropout=0.2, return_sequences=True))(embedding)

    # if using pos features
    if(args.use_pos == 1):
        pos_dim = pos_vectors["CD"].shape[0]
        embedding_pos_init = np.zeros((len(pos2index)+1, pos_dim))
        
        for pos, ind in pos2index.items():
            if(pos == "-OOV-"):
                embedding_pos_init[ind] = np.random.uniform(1e-3, 1e-2, size=(pos_dim,))
            elif(pos in pos2index):
                embedding_pos_init[ind] = pos_vectors[pos]

        # similar with the word features
        X_pos = pad_sequences(pos_index, maxlen=max_seq, padding="post", value=-1.)
        input_pos = Input(shape=(max_seq,))
        mask_pos = Masking(mask_value=-1.)(input_pos)
        embedding_pos = Embedding(len(pos2index)+1,
                                     output_dim=pos_dim,
                                     trainable=True,
                                     weights=[embedding_pos_init])(mask_pos)
        
        embedding_pos = SpatialDropout1D(0.2)(embedding_pos)

        # add two feature's embedding
        x = Add()([embedding, embedding_pos])
        x = Bidirectional(LSTM(64, activation="tanh", dropout=0.2, return_sequences=True))(x)
    
    x = Dropout(0.2)(x)

    # in order to get a sequence of output
    x = TimeDistributed(Dense(len(tag2index)+1))(x)
    x = Activation("softmax")(x)

    if(args.use_pos == 1):
        model = Model([inputs, input_pos], x)
    else:
        model = Model([inputs], x)

    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(0.001),
                  metrics=['accuracy', ignore_class_accuracy(0)])
     
    model.summary()
    # print(X_train)

    if(istrain):
        callbacks = []
        callbacks.append(EarlyStopping(monitor='val_loss', patience=5))
        callbacks.append(ModelCheckpoint(mname, monitor='val_loss', save_best_only=True))

        if(args.use_pos == 1):
            model.fit([X_train, X_pos], y_tags, batch_size=128, epochs=1000, validation_split=0.2, callbacks=callbacks,)
        else:
            model.fit([X_train], y_tags, batch_size=128, epochs=1000, validation_split=0.2, callbacks=callbacks,)
        
    return model


def get_pos(X):
    from nltk import pos_tag
    X_pos = []
    for x in X:
        # get the node's pos tag
        pos = [m[1] for m in pos_tag(x)]
        X_pos.append(pos)

    return X_pos

if __name__ == '__main__':
    # pre-calculate look up dict 
    word2index = np.load("model/word2index.npy").item()
    tag2index = np.load("model/tag2index.npy").item()
    index2tag = np.load("model/index2tag.npy").item()
    pos2index = np.load("model/pos2index.npy").item()


    if(args.train == 1):
        # get the nodes and their labels
        X_train, y_train = get_data()

        seq_index = []

        # convert word to index
        for s in X_train:
            X_int = []
            for w in s:
                try:
                    X_int.append(word2index[w])
                except KeyError:
                    X_int.append(word2index['-OOV-'])
     
            seq_index.append(X_int)

        if(args.use_pos == 1):
            pos_index = []
            X_pos = get_pos(X_train)

            # convert pos to index
            for s in X_pos:
                X_int = []
                for w in s:
                    try:
                        X_int.append(pos2index[w])
                    except KeyError:
                        X_int.append(pos2index['-OOV-'])
         
                pos_index.append(X_int)
            # get the pos embedding
            w2v_pos_wv = W2V(X_pos, istrain=False, mname=w2v_pos_name).wv
        else:
            pos_index = None
            w2v_pos_wv = None

        y_tags = []
        for s in y_train:
            y_tags.append([tag2index[t] for t in s])

        # convert label to one-hot encoding
        y_tags = to_categorical(y_tags, len(tag2index)+1)

        print(seq_index[0])
        print(pos_index[0])

        # get the word embedding
        w2v_model = W2V(X_train, istrain=False) # numpy array will cause an error ...

        model = train_model(seq_index, pos_index, y_tags, w2v_model.wv, w2v_pos_wv, word2index, pos2index, tag2index, istrain=True)
    

###################### Testing ###############################
    model = load_model(mname, custom_objects={"ignore_accuracy":ignore_class_accuracy(0)})
    # plot_model(model, to_file='model.png')

    X_test, y_test = get_data(file="data/test.json")
    X_pos = get_pos(X_test)

    seq_index = []
    # convert word to index
    for s in X_test:
        X_int = []
        for w in s:
            try:
                X_int.append(word2index[w])
            except KeyError:
                X_int.append(word2index['-OOV-'])
 
        seq_index.append(X_int)

    if(args.use_pos == 1):
        pos_index = []
        # convert pos to index
        for s in X_pos:
            X_int = []
            for w in s:
                try:
                    X_int.append(pos2index[w])
                except KeyError:
                    X_int.append(pos2index['-OOV-'])
     
            pos_index.append(X_int)
        X_pos = pad_sequences(pos_index, maxlen=max_seq, padding="post", value=-1.)

    X_test = pad_sequences(seq_index, maxlen=max_seq, padding="post", value=-1.)

    y_tags = []
    for s in y_test:
        y_tags.append([tag2index[t] for t in s])

    # convert labels to one-hot encoding
    y_tags = to_categorical(y_tags, len(tag2index)+1)

    if(args.use_pos == 1):
        y_prob = model.predict([X_test, X_pos])
        loss, accuracy, ignore_accuracy = model.evaluate([X_test, X_pos], y_tags)
    else:
        y_prob = model.predict([X_test])    
        loss, accuracy, ignore_accuracy = model.evaluate([X_test], y_tags)
    
    print(loss, accuracy, ignore_accuracy)

    y_pred = np.argmax(y_prob, axis=-1)
    y_predict = []
    for s in y_pred:
        temp = []
        for y in s:
            if(int(y) == 0):
                temp.append("-PAD-")
            else:
                temp.append(index2tag[y])
        y_predict.append(np.array(temp))

    y_predict = np.array(y_predict)
    # save the prediction
    np.save("y_predict.npy", y_predict)
