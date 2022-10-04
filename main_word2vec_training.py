# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:03:13 2022

@author: erikycd

This code yields a classification model from text with Random Forest algorithm
    
"""

#%% IMPORTING LIBRARIES

import pandas as pd 
import numpy as np
import yaml
from yaml import safe_load
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from gensim.models.word2vec import Word2Vec


#%% CLASS WORD2VEC CLASSIFICATION

class Word2VecClassification:

    def __init__(self, vector_size, window_size, algorithm):
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
        print('Obteniendo vectores...')
        docs = []
        for doc in tokenized_corpus:
            doc =  [x for x in doc if x not in self.sw]
            vec = self.buildWordVector(doc)
            vec = np.nan_to_num(vec.astype(np.float32))
            docs.append(vec)
        return docs


    def get_tokenized_corpus(self, corpus):
        return [self.tkr.tokenize(text.strip().lower()) for text in corpus]


    def fit_w2v(self, tokenized_corpus):
        print('Adecuando al modelo word2vec...')
        return Word2Vec(sentences = tokenized_corpus,
                            vector_size = self.vector_size,
                            window = self.window_size,
                            min_count = 2,
                            negative = 20,
                            hs = 0,
                            ns_exponent = .5,
                            cbow_mean = 1,
                            epochs = 150,
                            sg = 0,                            
                            )


    def fit(self, corpus, y_train):
        # train w2ec
        tokenized_corpus = self.get_tokenized_corpus(corpus)
        self.model_w2v = self.fit_w2v(tokenized_corpus)
        # train classification
        x_train = self.get_feature_from_vec(tokenized_corpus)
        self.model_cls = self.algorithm.fit(x_train, y_train)
        return x_train, y_train


    def predict(self, corpus):
        tokenized_corpus = self.get_tokenized_corpus(corpus)
        x = self.get_feature_from_vec(tokenized_corpus)
        return self.model_cls.predict(x)
    
    
    def save(self, report, model):
        text_file = open("./word2vec_engine/report.txt", "w")
        text_file.write(report)
        text_file.close()
        print('Reporte de entrenamiento generado')
        filename_model = './word2vec_engine/word2vec_model.sav'
        pickle.dump(model, open(filename_model, 'wb'))
        print('Modelo word2vec_model.sav guardado en carpeta ./word2vec_engine/')
        
    
#%% CLASS DATA PROCESSING

class DataProcessing:
    
    def __init__(self, path):
        self.path = path

    def reading_data(self):
        print('Procesando los datos...')
        with open(self.path, 'r', encoding = 'utf-8') as file:
            intents = safe_load(file)
        
        data_2 = []
        labels_2 = []
        for classes in intents:
            lines = intents[classes]
            label = [classes] * len(lines)
            data_2+=lines
            labels_2+=label
            
        df_dataset = pd.DataFrame(list(zip(data_2, labels_2)), columns = ['Text', 'Label'])
        print('Clases detectadas: ', pd.unique(df_dataset['Label']))
            
        return df_dataset
    
    def split(self, df_dataset):
        train, test = train_test_split(df_dataset, test_size = 0.2, random_state = 1, shuffle = True)
        
        return train, test
        
    
#%% DEF MAIN

def main(data_file):
    
    data_proc = DataProcessing(path = data_file)
    
    df_dataset = data_proc.reading_data()
    train, test = data_proc.split(df_dataset)
    
    model_word2vec = Word2VecClassification(vector_size = 3, window_size = 5, algorithm = RandomForestClassifier())
    fv, fl = model_word2vec.fit(train['Text'], train['Label'])
    
    predictions = model_word2vec.predict(test['Text'])
    report = classification_report(test['Label'], predictions)
    
    model_word2vec.save(report, model_word2vec)

#%% MAIN

if __name__ == '__main__':
    
    data_file = 'intent_file_word2vec.yml'
    main(data_file)
    





    