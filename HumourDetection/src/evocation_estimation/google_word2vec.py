'''
Created on Jan 25, 2017

@author: Andrew
'''
from gensim.models import KeyedVectors
from scipy.stats import entropy
from numpy import zeros

class GoogleWord2Vec(object):
    def __init__(self, vector_loc, binary=True):
        #http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
        self.model = KeyedVectors.load_word2vec_format(vector_loc, binary=binary)
    
    def get_vector(self, word):
        vector = None
        try:
            vector = self.model[word.lower()]
        except KeyError:
            vector = zeros(300)
        return vector
    
    def get_similarity(self, word1, word2):
        similarity = 0.0
        try:
            similarity = self.model.similarity(word1, word2)
        except KeyError:
            #similarity is 0
            pass
        return similarity
    
    def get_relative_entropy(self, word1, word2):
        return entropy(self.get_vector(word1), self.get_vector(word2))