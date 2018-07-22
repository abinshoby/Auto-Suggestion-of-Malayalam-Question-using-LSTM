# -*- coding: utf-8 -*-
#By abin and hari
import numpy as np
import pandas as pd
import re
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




def train():
    from numpy import asarray
    from numpy import zeros
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Embedding
    # define documents
    docs=[]
    labels=[]
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
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(docs)
    print(encoded_docs)
    save(encoded_docs,docs)

    max_length = 50
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs)
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
    embedding_matrix = zeros((vocab_size, 300))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
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
    model.fit(padded_docs, labels, epochs=50, verbose=1)
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




train()
