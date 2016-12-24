# -*- coding: utf-8 -*-

__author__ = 'xead'
import pickle
import os

import string

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize 

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

class SentimentClassifier(object):
    def preprocess(self, doc):               
        if type(doc) is str:
            return doc.lower().translate(string.maketrans("",""), string.punctuation)    
        else:
            return doc.lower().translate({ord(c): None for c in string.punctuation})  

    def __init__(self):
        APP_ROOT = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(APP_ROOT, 'fitted_logregression.sav'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(APP_ROOT, 'fitted_count_vectorizer.sav'), 'rb') as f:
            self.vectorizer = pickle.load(f)        
        self.classes_dict = { 0: u'негативная', 1: u'позитивная', -1: u'ошибка оценки' }

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return u'нейтральная или неочевидная'
        if probability < 0.7:
            return u'вероятно'
        if probability > 0.95:
            return u'несомненно'
        else:
            return ""

    def get_feature_weights(self, vectorized):
        idx = np.nonzero(vectorized.toarray())
        feature_names = np.array(self.vectorizer.get_feature_names())[idx[1]]
        feature_weights = self.model.coef_[:,idx[1]][0]
        w = zip(feature_names, feature_weights)
        w.sort(key=lambda tup: -abs(tup[1])) 
        return w


    def vectorize(self, text):
        return self.vectorizer.transform([text])

    def predict_vectorized(self, vectorized):
        try:
            return self.model.predict(vectorized)[0],\
                   self.model.predict_proba(vectorized)[0].max()
        except:
            print "prediction error"
            return -1, 0.8

    def get_prediction_message(self, vectorized):
        prediction = self.predict_vectorized(vectorized)
        class_prediction = prediction[0]
        prediction_probability = prediction[1]
        return self.get_probability_words(prediction_probability) + " " + self.classes_dict[class_prediction]