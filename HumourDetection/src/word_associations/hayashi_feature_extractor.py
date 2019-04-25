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
from util.loggers import LoggerMixin
from util.wordnet.wordnet_utils import get_lex_vector, wup_similarity

class HayashiFeatureExtractor(TransformerMixin, LoggerMixin):
    def __init__(self, lda_model_getter=None, w2v_model_getter=None, autoex_model_getter=None, betweenness_loc=None, load_loc=None,  verbose=False):
        """
            Initialize a Word Association Strength feature extractor
            
            :param lda_model_getter: Getter function for LDA model to use for feature extraction
            :type lda_model_getter: Callable[[], GensimTopicSumModel]
            :param w2v_model_getter: Getter function for Word2Vec model to use for feature extraction
            :type w2v_model_getter: Callable[[], GensimTopicSumModel]
            :param autoex_model_getter: Getter function for AutoExtend model to use for feature extraction
            :type autoex_model_getter: Callable[[], GensimTopicSumModel]
            :param betweenness_loc: location of betweenness centrality pkl
            :type betweenness_loc: str
            :param load_loc: location of load centrality pkl
            :type load_loc: str
            :param verbose: whether verbose mode should be used or not
            :type verbose: bool
        """
        
        self.get_lda_model = lda_model_getter
        self.get_w2v_model = w2v_model_getter
        self.get_autoex_model = autoex_model_getter
        self.betweenness_loc = betweenness_loc
        self.load_loc = load_loc
        
        #TODO: not global logging
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        
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
        dimensions += self.get_autoex_model().get_dimensions() #add the dimensionality of the autoex embeddings
        
        return dimensions
    
    def get_lda_feats(self, stimuli_response):
        features=[]
        
        processed = 0
        total = len(stimuli_response)
        for stimuli, response in stimuli_response:
            stim_word = self._get_name(stimuli)
            resp_word = self._get_name(response)
            lda_sim=self.get_lda_model().get_similarity(stim_word, resp_word)
            
            features.append(lda_sim)
            
            processed += 1
            if not (processed % self.verbose_interval):
                self.logger.debug("{}/{} done".format(processed, total))
        
        return np.vstack(features)
    
    def get_w2v_feats(self, stimuli_response):
        features = []
        for stimuli, response in stimuli_response:
            stim_word = self._get_name(stimuli)
            resp_word = self._get_name(response)
            w2v_sim = self.get_w2v_model().get_similarity(stim_word, resp_word)
            features.append(w2v_sim)
        
        return np.vstack(features)
    
    def get_autoex_feats(self, stimuli_response):
        features = []
        for stimuli, response in stimuli_response:
            autoex_sim = self.get_autoex_model().get_similarity(stimuli, response)
            autoex_offset = self.get_autoex_model().get_vector(stimuli) - self.get_autoex_model().get_vector(response)
            
            features.append(np.hstack((autoex_sim, autoex_offset)))
        
        return np.vstack(features)
    
    def get_wn_betweenness(self, stimuli_response):
        features = []
        with open(self.betweenness_loc, "rb") as betweeneness_pkl:
            betweenness = pickle.load(betweeneness_pkl)
        
        for stimuli, response in stimuli_response:
            stim_betweenness = betweenness.get(stimuli,0.0)
            resp_betweenness = betweenness.get(response,0.0)
            
            features.append(np.hstack((stim_betweenness,resp_betweenness)))
            
        return np.vstack(features)
    
    def get_wn_load(self, stimuli_response):
        features = []
        with open(self.load_loc, "rb") as load_pkl:
            load = pickle.load(load_pkl)
        
        for stimuli, response in stimuli_response:
            stim_load = load.get(stimuli,0.0)
            resp_load = load.get(response,0.0)
            
            features.append(np.hstack((stim_load, resp_load)))        
        
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
        
        total = len(stimuli_response)
        processed = 0
        for stimuli, response in stimuli_response:
            stimuli_synset = self._get_synset(stimuli)
            response_synset = self._get_synset(response)
            
            stim_lexvector = get_lex_vector([stimuli_synset])
            resp_lexvector = get_lex_vector([response_synset])
            wup_sim = wup_similarity(stimuli_synset, response_synset, simulate_root=True)
            
            features.append(np.hstack((stim_lexvector, resp_lexvector, wup_sim)))
            
            processed += 1
            if not (processed % self.verbose_interval):
                self.logger.debug("{}/{} done".format(len(features), total))
        
        return np.vstack(features)
    
    def fit(self,X, y=None):
        return self
    
    def transform(self,stimuli_response):
        association_tuples = [ (stimuli.lower(), response.lower()) for stimuli, response in stimuli_response]
        
        features = []
        
        self.logger.debug("starting lda")
        features.append(self.get_lda_feats(association_tuples))
        self.logger.debug("lda done")
        
        self.logger.debug("starting w2v")
        features.append(self.get_w2v_feats(association_tuples))
        self.logger.debug("w2v done")
        self.get_w2v_model()._purge_model()
        
        self.logger.debug("starting autoex")
        features.append(self.get_autoex_feats(association_tuples))
        self.logger.debug("autoex done")
        
        self.logger.debug("starting betweenness")
        features.append(self.get_wn_betweenness(association_tuples))
        self.logger.debug("betweenness done")
        
        self.logger.debug("starting load")
        features.append(self.get_wn_load(association_tuples))
        self.logger.debug("load done")
        
        self.logger.debug("starting dirrels")
        features.append(self.get_dir_rel(association_tuples))
        self.logger.debug("dirrels done")
        
        self.logger.debug("starting wordnet feats")
        features.append(self.get_wn_feats(association_tuples))
        self.logger.debug("wordnet feats done")
                   
        return np.hstack(features)

if __name__ == "__main__":
#     from word_associations.association_readers.xml_readers import EvocationDataset
#     import random
#       
#     evoc = EvocationDataset("../Data/evocation").get_all_associations()
#     random.seed(10)
#     random.shuffle(evoc)
#     syn_pair, stren = zip(*evoc)
#     del evoc
#  
#     stren = np.array(stren)
#       
#     from util.model_wrappers.common_models import get_wikipedia_lda, get_google_word2vec,\
#     get_google_autoextend
#     betweenness_pkl="d:/git/HumourDetection/HumourDetection/src/word_associations/wordnet_betweenness.pkl"
#     load_pkl="d:/git/HumourDetection/HumourDetection/src/word_associations/wordnet_load.pkl"
#       
#     fe = HayashiFeatureExtractor(get_wikipedia_lda, get_google_word2vec, get_google_autoextend, betweenness_pkl, load_pkl, True)
#       
#     mat = fe.fit_transform(syn_pair, stren)
#       
#     np.save("hayashi_features.npy", mat)
#     np.save("hayashi_strengths.npy", stren)
    mat=np.load("hayashi_features.npy")
    stren=np.load("hayashi_strengths.npy")

    from keras.wrappers.scikit_learn import KerasRegressor
    from word_associations.strength_prediction import _create_mlp
    from sklearn.model_selection import cross_val_predict
    
    input_dim = mat.shape[1]
    num_units = max(int(input_dim/2), 5)
    estimator = KerasRegressor(build_fn=_create_mlp, num_units=num_units, input_dim=input_dim, epochs=100, batch_size=250, verbose=2)
    pred_y = cross_val_predict(estimator, mat, stren, cv=5)
    
    from scipy.stats.stats import pearsonr, spearmanr
    
    r = pearsonr(stren, pred_y)
    print(f"pearson r: {r[0]}\t\tp-value: {r[1]}")
    rho = spearmanr(stren, pred_y)
    print(f"spearman rho: {rho[0]}\t\tp-value: {rho[1]}")