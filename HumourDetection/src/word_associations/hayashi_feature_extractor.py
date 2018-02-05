'''
Created on Jan 27, 2017

@author: Andrew
'''
from __future__ import print_function, division #for Python 2.7 compatibility
import pickle
import numpy as np
from nltk.corpus import wordnet
import logging
from sklearn.base import TransformerMixin
from util.wordnet.wordnet_graph import WordNetGraph
from util.wordnet.wordnet_utils import WordNetUtils

class HayashiFeatureExtractor(TransformerMixin):
    def __init__(self, lda_model=None, w2v_model=None, autoex_model=None, betweenness_loc=None, load_loc=None,  verbose=False):
        """
            Initialize a Word Association Strength feature extractor
            
            :param lda_model: LDA model to use for feature extraction
            :type lda_model: util.gensim_wrappers.gensim_topicsum_models.GensimTopicSumModel
            :param w2v_model: Word2Vec model to use for feature extraction
            :type w2v_model: util.gensim_wrappers.gensim_vector_models.GensimVectorModel
            :param autoex_model: AutoExtend model to use for feature extraction
            :type autoex_model: util.gensim_wrappers.gensim_vector_models.GensimVectorModel
            :param betweenness_loc: location of betweenness centrality pkl
            :type betweenness_loc: str
            :param load_loc: location of load centrality pkl
            :type load_loc: str
            :param verbose: whether verbose mode should be used or not
            :type verbose: bool
        """
        
        self.lda_model = lda_model
        self.w2v_model = w2v_model
        self.autoex_model = autoex_model
        self.betweenness_loc = betweenness_loc
        self.load_loc = load_loc
        
        #TODO: not global logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        
        self.verbose_interval = 1000
    
    def _get_synset(self,word):
        return wordnet.synset(word)
    
    def _get_name(self,word):
        return word.split(".", 1)[0]
    
    def get_num_dimensions(self):
        """
            Returns the number of feature vector dimensions.
            
            :returns: the number of feature dimensions
            :rtype: int
        """
        dimensions = 109 #the number of dimensions for all features except autoex offset
        dimensions += self.autoex_model.get_dimensions() #add the dimensionality of the autoex embeddings
        
        return dimensions
    
    def get_lda_feats(self, stimuli_response):
        features=[]
        
        processed = 0
        total = len(stimuli_response)
        for stimuli, response in stimuli_response:
            stim_word = self._get_name(stimuli)
            resp_word = self._get_name(response)
            lda_sim=self.lda_model.get_similarity(stim_word, resp_word)
            
            features.append(lda_sim)
            
            processed += 1
            if not (processed % self.verbose_interval):
                logging.debug("{}/{} done".format(processed, total))
        
        return np.vstack(features)
    
    def get_w2v_feats(self, stimuli_response):
        features = []
        for stimuli, response in stimuli_response:
            stim_word = self._get_name(stimuli)
            resp_word = self._get_name(response)
            w2v_sim = self.w2v_model.get_similarity(stim_word, resp_word)
            features.append(w2v_sim)
        
        return np.vstack(features)
    
    def get_autoex_feats(self, stimuli_response):
        features = []
        for stimuli, response in stimuli_response:
            autoex_sim = self.autoex_model.get_similarity(stimuli, response)
            autoex_offset = self.autoex_model.get_vector(stimuli) - self.autoex_model.get_vector(response)
            
            features.append((autoex_sim, autoex_offset))
        
        return np.vstack(features)
    
    def get_wn_betweenness(self, stimuli_response):
        features = []
        with open(self.betweenness_loc, "rb") as betweeneness_pkl:
            betweenness = pickle.load(betweeneness_pkl)
        
        for stimuli, response in stimuli_response:
            stim_betweenness = betweenness.get(stimuli,0.0)
            resp_betweenness = betweenness.get(response,0.0)
            
            features.append((stim_betweenness,resp_betweenness))
            
        return np.vstack(features)
    
    def get_wn_load(self, stimuli_response):
        features = []
        with open(self.load_loc, "rb") as load_pkl:
            load = pickle.load(load_pkl)
        
        for stimuli, response in stimuli_response:
            stim_load = load.get(stimuli,0.0)
            resp_load = load.get(response,0.0)
            
            features.append((stim_load, resp_load))        
        
        return np.vstack(features)
    
    def get_dir_rel(self, stimuli_response):
        features = []
        wn_graph = WordNetGraph()
        
        for stimuli, response in stimuli_response:
            dirrel = wn_graph.get_directional_relativity([stimuli],[response])
            
            features.append(dirrel)
        
        return np.vstack(features)
    
    def get_wn_feats(self, stimuli_response):
        features = []
        wnu = WordNetUtils(cache=True)
        
        total = len(stimuli_response)
        processed = 0
        for stimuli, response in stimuli_response:
            stimuli_synset = self._get_synset(stimuli)
            response_synset = self._get_synset(response)
            
            stim_lexvector = wnu.get_lex_vector([stimuli_synset])
            resp_lexvector = wnu.get_lex_vector([response_synset])
            wup_sim = wnu.wup_similarity_w_root(stimuli_synset, response_synset)
            
            features.append((stim_lexvector, resp_lexvector, wup_sim))
            
            processed += 1
            if not (processed % self.verbose_interval):
                logging.debug("{}/{} done".format(len(features), total))
        
        return np.vstack(features)
    
    def transform(self,stimuli_response):
        association_tuples = [ (stimuli.lower(), response.lower()) for stimuli, response in stimuli_response]
        
        features = []
        
        logging.debug("starting lda")
        features.append(self.get_lda_feats(association_tuples))
        logging.debug("lda done")
        
        logging.debug("starting w2v")
        features.append(self.get_w2v_feats(association_tuples))
        logging.debug("w2v done")
        
        logging.debug("starting autoex")
        features.append(self.get_autoex_feats(association_tuples))
        logging.debug("autoex done")
        
        logging.debug("starting betweenness")
        features.append(self.get_wn_betweenness(association_tuples))
        logging.debug("betweenness done")
        
        logging.debug("starting load")
        features.append(self.get_wn_load(association_tuples))
        logging.debug("load done")
        
        logging.debug("starting dirrels")
        features.append(self.get_dir_rel(association_tuples))
        logging.debug("dirrels done")
        
        logging.debug("starting wordnet feats")
        features.append(self.get_wn_feats(association_tuples))
        logging.debug("wordnet feats done")
                   
        return np.hstack(features)