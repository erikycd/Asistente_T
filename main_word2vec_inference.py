# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:07:17 2022

@author: erikycd

This code simulates the inference process for the word2vec algorithm

"""

#%% IMPORTING LIBRARIES
    
import numpy as np
import pickle

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier


#%% CLASS WORD2VEC CLASSIFICATION

class Word2VecClassification:

    def __init__(self, vector_size = 10, window_size = 3, algorithm = RandomForestClassifier()):
        self.vector_size = vector_size
        self.window_size = window_size
        self.tkr = RegexpTokenizer('[a-zA-Z]+')
        self.sw = stopwords.words('spanish')
        self.algorithm = algorithm
        self.model_w2v = None
        self.model_cls = None


    def buildWordVector(self, doc):
        vec_matrix = np.zeros(shape = (self.vector_size, len(doc)))

        for idw, word in enumerate(doc):
            if word not in self.sw:      
                try:
                    vec_matrix[:, idw] = self.model_w2v.wv[word]
                except KeyError:
                    continue

        return np.mean(vec_matrix, axis = 1)


    def get_feature_from_vec(self, tokenized_corpus):
        docs = []
        for doc in tokenized_corpus:
            doc =  [x for x in doc if x not in self.sw]
            vec = self.buildWordVector(doc)
            vec = np.nan_to_num(vec.astype(np.float32))
            docs.append(vec)
        return docs


    def get_tokenized_corpus(self, corpus):
        return [self.tkr.tokenize(text.strip().lower()) for text in corpus]


    def predict(self, corpus):
        tokenized_corpus = self.get_tokenized_corpus(corpus)
        x = self.get_feature_from_vec(tokenized_corpus)
        return self.model_cls.predict(x)


#%% INFERENCE

filename_model = './word2vec_engine/word2vec_model.sav'
loaded_model = pickle.load(open(filename_model, 'rb'))

exit_commands = ('bye', 'quit')
user = ''

print('Envia "bye" o "quit" para teminar la inferencia. \n')

while user not in exit_commands:

    user = input('Usuario: ')
    reply  = loaded_model.predict([user])
    print('Clasificacion: ', reply)
    
print('Inferencia finalizada')









