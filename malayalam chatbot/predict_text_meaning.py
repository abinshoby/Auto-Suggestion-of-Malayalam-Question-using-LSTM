# -*- coding: utf-8 -*-
#By abin and hari
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_yaml
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
import _pickle as cPickle
from collections import defaultdict
import re
import os
from keras.models import model_from_json
import nltk
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()
def avg(av,sent):
    l=[]
    for i in range(300):
        l.append((av[i]+sent[i])/2)
    return l
def sent2vec(sentgroup):
    last=[]
    for sent in sentgroup:
        av=np.asarray(sent[0])
        for i in range(1,len(sent)):
                av=avg(av,sent[i])
        last.append(av)
    #print("last",last)
    return last


def load_map():
    import json
    with open('word_map.json', 'r') as fp:
        word_map = json.load(fp)
    #print(word_map)
    return word_map
def predict(text):
    if(len(text)>0):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model.h5")
        print("Loaded model from disk")
        t = Tokenizer()
        t.fit_on_texts(text)
        #print(text)
        tok_docs = []


        print(text)


        tok = nltk.word_tokenize(text[0])
        tok_docs.append(tok)
        #print(tok_docs)
        word_map=load_map()
        max1=max(word_map.values())
        text_seq=[]
        for sent in tok_docs:
            s1=[]
            for w in sent:
                if w in word_map.keys():
                    s1.append(word_map[w])
                else:
                    #max1=max1+1
                    s1.append(0)

            text_seq.append(s1)
        #print(text_seq)
        max_length = 50
        padded_docs = pad_sequences(text_seq, maxlen=max_length, padding='post')


        pred = model.predict(padded_docs)

        print(pred)
        print(padded_docs)
        out = []
        for doc in pred:
            out.append(doc.tolist().index(max(doc.tolist())))
        print(out)
        data_train = pd.read_csv('data/querydata_to_be_suggested.tsv', sep='\t')
        return (data_train.loc[data_train['sentiment'] == out[0]]['review']).values.tolist()[0]
    else:
        return [""]
# l=predict("ഹോട്ടലുകളുടെ ")
# print(l)
# print(predict(['ഹോട്ടലുകളുടെ പട്ടിക തരുക']))