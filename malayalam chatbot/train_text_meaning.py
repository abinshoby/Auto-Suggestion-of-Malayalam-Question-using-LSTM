# -*- coding: utf-8 -*-
#By abin and hari
import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
from IPython.display import clear_output
from bs4 import BeautifulSoup
import os
os.environ['KERAS_BACKEND'] = 'theano'
import nltk
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.layers import  LSTM,  Bidirectional

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()



def save(encoded_docs,docs):
    tok_docs=[]
    tokenizer = Tokenizer()
    for d in docs:
        tok=nltk.word_tokenize(d)
        tok_docs.append(tok)

    print(tok_docs)
    word_map={}
    for (doc,seq) in zip(tok_docs,encoded_docs):
        for (w,n) in zip(doc,seq):
            word_map[w]=n
    import json

    with open('word_map.json', 'w') as fp:
        json.dump(word_map, fp)

    #print(word_map)


def remove_nonweighted(pad_docs,remove_list):
    last=[]
    for seq in pad_docs:
        sub=[]
        for i in seq:
            if i not in remove_list:
                sub.append(i)
        last.append(sub)
    return last



def train():
    from numpy import asarray
    from numpy import zeros
    import keras
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Embedding
    # define documents
    docs=[]
    labels=[]


    class PlotLearning(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.i = 0
            self.x = []
            self.losses = []
            self.val_losses = []
            self.acc = []
            self.val_acc = []
            self.fig = plt.figure()

            self.logs = []

        def on_epoch_end(self, epoch, logs={}):
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.acc.append(logs.get('acc'))
            self.val_acc.append(logs.get('val_acc'))
            self.i += 1
            f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

            clear_output(wait=True)

            ax1.set_yscale('log')
            ax1.plot(self.x, self.losses, label="loss")
            ax1.plot(self.x, self.val_losses, label="val_loss")
            ax1.legend()

            ax2.plot(self.x, self.acc, label="accuracy")
            ax2.plot(self.x, self.val_acc, label="validation accuracy")
            ax2.legend()

            plt.show();

    plot = PlotLearning()



    data_train = pd.read_csv('data/querydata.tsv', sep='\t')

    for idx in range(data_train.question.shape[0]):
        text = BeautifulSoup(data_train.question[idx])
        docs.append(clean_str(text.get_text()))
        labels.append(data_train.sentiment[idx])


    # define class labels

    labels = to_categorical(np.asarray(labels))
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(docs)

    vocab_size = len(t.word_index) + 1
    print("word_index",t.word_index)
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(docs)
    print("encoding1", encoded_docs)



    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('glove/wiki.ml.vec')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    remove_list=[]
    embedding_matrix = zeros((vocab_size, 300))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            remove_list.append(i)

    encoded_docs=remove_nonweighted(encoded_docs,remove_list)
    max_length = 50
    print("encoding2", encoded_docs)
    save(encoded_docs, docs)
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs)

    # define model
    model = Sequential()
    e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False)
             #no of distinct words,output dim,weight,max no of words in a sentence
    model.add(e)

    model.add(Bidirectional(LSTM(100,return_sequences=True)))
    model.add(Bidirectional(LSTM(100)))

    model.add(Dense(len(labels[0]), activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    # fit the model

    model.fit(padded_docs[:int(len(padded_docs)*0.8)], labels[:int(len(padded_docs)*0.8)],validation_data=(padded_docs[int(len(padded_docs)*0.8):],labels[int(len(padded_docs)*0.8):]), epochs=25, verbose=1,callbacks=[plot])
    # evaluate the model
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=1)
    print('Accuracy: %f' % (accuracy * 100))



    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5",overwrite=True)
    print("Saved model to disk")
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    print(padded_docs)
    pred=model.predict(padded_docs)
    print(pred)
    out=[]
    for doc in pred:
        out.append(doc.tolist().index(max(doc.tolist())))
    print(out)

#epoch50
#90% accuracy 53.33% validation accuracy
#trial2  loss: 0.0700 - acc: 0.9853 - val_loss: 1.6664 - val_acc: 0.6176
#trial3 loss: 0.0696 - acc: 0.9853 - val_loss: 1.9343 - val_acc: 0.5588

#epoch60
#trial4 loss: 0.0611 - acc: 0.9853 - val_loss: 1.4027 - val_acc: 0.6765     total 92.352941
#trial 7 loss: 0.0583 - acc: 0.9853 - val_loss: 1.5779 - val_acc: 0.6471    total 91.764706
#epoch25
#trial5loss: 0.2487 - acc: 0.9706 - val_loss: 2.0502 - val_acc: 0.4706   total 87.058824
#trial6 loss: 0.2440 - acc: 0.9632 - val_loss: 1.8709 - val_acc: 0.5000  total 88.235294
#trial8 oss: 0.2823 - acc: 0.9779 - val_loss: 1.6043 - val_acc: 0.5882    total 90.588235

train()
