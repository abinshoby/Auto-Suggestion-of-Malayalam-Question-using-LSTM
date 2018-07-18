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
from keras.utils import plot_model

from keras.utils.vis_utils import model_to_dot

from keras.utils.vis_utils import plot_model




import re
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def predict(text):
    yaml_file = open('/home/abin/PycharmProjects/ICFOSS/query_model_test.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    MAX_SEQUENCE_LENGTH = 100
    MAX_NB_WORDS = 20000  # 20000
    EMBEDDING_DIM = 300
    VALIDATION_SPLIT = 0.2
    texts = []
   # text = BeautifulSoup(text)
    texts.append(clean_str(text))#text.get_text()

    tokenizer = Tokenizer(nb_words=20000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    #plot_model(model, to_file='model.png')
    #SVG(model_to_dot(model).create(prog='dot', format='svg'))
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    result = model.predict(data)
    print(result)
    res_list=list(result[0])
    ind=res_list.index(max(res_list))
    print(ind)
    #print("{} positive, {} negeative.".format(result[0,1], result[0,0]))
    data_train = pd.read_csv('/home/abin/PycharmProjects/ICFOSS/data/querydata_to_be_suggested.tsv', sep='\t')
    return str((data_train.loc[data_train['sentiment'] == ind]['review']).values.tolist()[0])

#l=predict("ബോട്ടിംഗ്")
#print(l)
