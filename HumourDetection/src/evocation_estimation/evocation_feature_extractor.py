'''
Created on Jan 27, 2017

@author: Andrew
'''
from __future__ import print_function, division #for Python 2.7 compatibility
import pickle
import numpy as np
from nltk.corpus import wordnet as wn
import re
# import os
from extended_lesk import ExtendedLesk
from util.gensim_wrappers.gensim_vector_models import load_gensim_vector_model
from util.gensim_wrappers.gensim_topicsum_models import load_gensim_topicsum_model,\
    TYPE_LDA
from util.model_name_consts import STANFORD_GLOVE, GOOGLE_W2V, AUTOEXTEND,\
    WIKIPEDIA_LDA, WIKIPEDIA_TFIDF, WIKIPEDIA_W2G
from sklearn.base import TransformerMixin
from util.wordnet.wordnet_graph import WordNetGraph
from util.wordnet.wordnet_utils import WordNetUtils
from util.word2gauss_wrapper import load_word2gauss_model
import logging
from util.misc import mean

#TODO: Add selective feature scaling (i.e. scale everything but vectors. Or should be scale vectors too?)
#TODO: Add a lot of documentation

#Word embedding features
FEAT_W2V_SIM = "Word2Vec Similarity"
FEAT_W2V_OFFSET = "Word2Vec Offset"
FEAT_W2V_VECTORS = "Full Word2Vec Vectors"
FEAT_GLOVE_SIM = "GloVe Similarity"
FEAT_GLOVE_OFFSET = "GloVe Offset"
FEAT_GLOVE_VECTORS = "Full GloVe Vectors"
FEAT_W2G_SIM = "Word2Gauss Similarity"
FEAT_W2G_ENERGY = "Word2Gauss Energy"
FEAT_W2G_OFFSET = "Word2Gauss Offset"
FEAT_W2G_VECTORS = "Full Word2Gauss Vectors"
FEAT_MAX_AUTOEX_SIM = "Maximum AutoExtend Similarity"
FEAT_AVG_AUTOEX_SIM = "Average AutoExtend Similarity"

FEAT_LDA_SIM = "LDA Similarity"

#WordNet features
FEAT_LEXVECTORS = "WordNet Lexvectors"
FEAT_MAX_WUP_SIM = "Maximum Wu-Palmer Similarity"
FEAT_AVG_WUP_SIM = "Average Wu-Palmer Similarity"
FEAT_MAX_LCH_SIM = "Maximum Leacock-Chodorow Similarity"
FEAT_AVG_LCH_SIM = "Average Leacock-Chodorow Similarity"
FEAT_MAX_PATH_SIM = "Maximum Path Similarity"
FEAT_AVG_PATH_SIM = "Average Path Similarity"
FEAT_MAX_LOAD = "Maximum Load Centrality"
FEAT_AVG_LOAD = "Average Load Centrality"
FEAT_TOTAL_LOAD = "Total Load Centrality"
FEAT_MAX_BETWEENNESS = "Maximum Betweenness Centrality"
FEAT_AVG_BETWEENNESS = "Average Betweenness Centrality"
FEAT_TOTAL_BETWEENNESS = "Total Betweenness Centrality"
FEAT_DIR_REL = "Directional Relativity"
FEAT_LESK = "Extended Lesk"

DEFAULT_FEATS = [FEAT_W2V_SIM,
                 FEAT_W2V_OFFSET,
                 FEAT_GLOVE_SIM,
                 FEAT_W2G_SIM,
                 FEAT_W2G_ENERGY,
                 FEAT_MAX_AUTOEX_SIM,
                 FEAT_AVG_AUTOEX_SIM,
                 FEAT_LDA_SIM,
                 FEAT_LEXVECTORS,
                 FEAT_MAX_WUP_SIM,
                 FEAT_AVG_WUP_SIM,
                 FEAT_MAX_LOAD,
                 FEAT_AVG_LOAD,
                 FEAT_MAX_BETWEENNESS,
                 FEAT_AVG_BETWEENNESS,
                 FEAT_DIR_REL
                 ]


class EvocationFeatureExtractor(TransformerMixin):
    def __init__(self, features=DEFAULT_FEATS, lda_loc=None, wordids_loc=None, tfidf_loc=None, w2v_loc=None, autoex_loc=None, betweenness_loc=None, load_loc=None,  glove_loc=None, w2g_model_loc=None, w2g_vocab_loc=None, lesk_relations=None, verbose=True):
        self.features = set(features)
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        self.verbose_interval = 1000
        
        self.features = self._validate_features(features)
        self.lda_loc = lda_loc
        self.wordids_loc = wordids_loc
        self.tfidf_loc = tfidf_loc
        self.w2v_loc = w2v_loc
        self.autoex_loc = autoex_loc
        self.betweenness_loc = betweenness_loc
        self.load_loc = load_loc
        self.glove_loc = glove_loc
        self.w2g_model_loc = w2g_model_loc
        self.w2g_vocab_loc = w2g_vocab_loc
        self.lesk_relations = lesk_relations
        
            
    
    def get_num_dimensions(self):
        """
            Returns the number of feature vector dimensions.
            
            Note this is not the same as the number of features. While most
            features are 1D, some features (e.g. FEAT_w2V_OFFSET) can be more
            than that.
            
            :returns: the number of feature dimensions
            :rtype: int
        """
        feats_2d = set([FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS,
                        FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD
                        ])
        
        dimensions = 0
        for feature in self.features:
            if feature == FEAT_GLOVE_OFFSET:
                glove = load_gensim_vector_model(STANFORD_GLOVE, self.glove_loc, True)
                dimensions += glove.get_dimensions()
            elif feature == FEAT_W2V_OFFSET:
                w2v = load_gensim_vector_model(GOOGLE_W2V, self.w2v_loc, True)
                dimensions += w2v.get_dimensions()
            elif feature == FEAT_W2G_OFFSET:
                w2g = load_word2gauss_model(WIKIPEDIA_W2G, self.w2g_model_loc, self.w2g_vocab_loc)
                dimensions += w2g.get_dimensions()
            elif feature == FEAT_GLOVE_VECTORS:
                glove = load_gensim_vector_model(STANFORD_GLOVE, self.glove_loc, True)
                dimensions += glove.get_dimensions() * 2
            elif feature == FEAT_W2V_VECTORS:
                w2v = load_gensim_vector_model(GOOGLE_W2V, self.w2v_loc, True)
                dimensions += w2v.get_dimensions() * 2
            elif feature == FEAT_W2G_VECTORS:
                w2g = load_word2gauss_model(WIKIPEDIA_W2G, self.w2g_model_loc, self.w2g_vocab_loc)
                dimensions += w2g.get_dimensions() * 2
            elif feature == FEAT_LEXVECTORS:
                dimensions += 100
            elif feature in feats_2d:
                dimensions += 2
            else:
                dimensions += 1
        
        return dimensions
    
    def _get_synsets(self,word):
        return wn.synsets(re.sub(" ", "_",word))
    
    def _get_synset_names(self,word):
        return [synset.name() for synset in self._get_synsets(word)]
    
    def _space_to_underscore(self, word):
        return re.sub(" ", "_",word)
    def _underscore_to_space(self, word):
        return re.sub("_", " ",word)
    
    def fit(self,X, y=None):
        return self
    
    def get_lda_feats(self,stimuli_response):
        feature_vects = []
        if (FEAT_LDA_SIM in self.features):
                #Only load LDA model if we care about LDA features
                lda = load_gensim_topicsum_model(WIKIPEDIA_LDA, TYPE_LDA, self.lda_loc, WIKIPEDIA_TFIDF, self.wordids_loc, self.tfidf_loc)
                
                for stimuli, response in stimuli_response:
                    lda_sim = lda.get_similarity(stimuli, response)
                    lda_sim = 0.0 if np.isnan(lda_sim) else lda_sim
                    
                    feature_vects.append(lda_sim)
        
        else:
            feature_vects = [[]] * len(stimuli_response) #default to empty feature set
        
        return np.vstack(feature_vects)
    
    def get_w2v_feats(self,stimuli_response):
        feature_vects=[]
        if any([feat in self.features for feat in (FEAT_W2V_OFFSET, FEAT_W2V_SIM, FEAT_W2V_VECTORS)]):
            #Only load Word2Vec model if we care about Word2Vec features
            w2v = load_gensim_vector_model(GOOGLE_W2V, self.w2v_loc)
            
            for stimuli, response in stimuli_response:
                feature_vect = []
                santized_stimuli = self._space_to_underscore(stimuli)
                santized_response = self._space_to_underscore(response)
                
                if FEAT_W2V_SIM in self.features:
                    feature_vect.append(w2v.get_similarity(santized_stimuli, santized_response))
                
                stim_vector = w2v.get_vector(santized_stimuli)
                resp_vector = w2v.get_vector(santized_response)
                if FEAT_W2V_OFFSET in self.features:
                    feature_vect.append(stim_vector - resp_vector)
                if FEAT_W2V_VECTORS in self.features:
                    feature_vect.append(stim_vector)
                    feature_vect.append(resp_vector)
                    
                feature_vects.append(np.hstack(feature_vect))
                
        else:
            feature_vects = [[]] * len(stimuli_response) #default to empty feature set
        
        return np.vstack(feature_vects)
    
    def get_glove_feats(self,stimuli_response):
        feature_vects=[]
        if any([feat in self.features for feat in (FEAT_GLOVE_OFFSET, FEAT_GLOVE_SIM, FEAT_GLOVE_VECTORS)]):
            #Only load GloVe model if we care about GloVe features
            glove = load_gensim_vector_model(STANFORD_GLOVE, self.glove_loc, True) #https://radimrehurek.com/gensim/scripts/glove2word2vec.html
            
            for stimuli, response in stimuli_response:
                feature_vect = []
                santized_stimuli = self._space_to_underscore(stimuli)
                santized_response = self._space_to_underscore(response)
                
                if FEAT_GLOVE_SIM in self.features:
                    feature_vect.append(glove.get_similarity(santized_stimuli, santized_response))
                
                stim_vector = glove.get_vector(santized_stimuli)
                resp_vector = glove.get_vector(santized_response)
                if FEAT_GLOVE_OFFSET in self.features:
                    feature_vect.append(stim_vector - resp_vector)
                if FEAT_GLOVE_VECTORS in self.features:
                    feature_vect.append(stim_vector)
                    feature_vect.append(resp_vector)
                    
                feature_vects.append(np.hstack(feature_vect))
                  
        else:
            feature_vects = [[]] * len(stimuli_response) #default to empty feature set
        
        return np.vstack(feature_vects)
    
    def get_autoex_feats(self,stimuli_response):
        feature_vects=[]
        if any([feat in self.features for feat in (FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM)]):
            #Only load AutoExtend model if we care about AutoExtend features
            autoex = load_gensim_vector_model(AUTOEXTEND, self.autoex_loc, True)
            
            total = len(stimuli_response)
            processed = 0
            for stimuli, response in stimuli_response:
                feature_vect = []
                stimuli_synsets = self._get_synset_names(self._space_to_underscore(stimuli))
                response_synsets = self._get_synset_names(self._space_to_underscore(response))
                
                synset_sims = []
                for stimuli_synset in stimuli_synsets:
                    for response_synset in response_synsets:
                        synset_sims.append(autoex.get_similarity(stimuli_synset, response_synset))
                
                if not synset_sims:
                    #if no valid readings exist (likely because one of the words has 0 synsets), default to 0
                    synset_sims = [0.0]
                
                if FEAT_MAX_AUTOEX_SIM in self.features:
                    feature_vect.append(max(synset_sims))
                if FEAT_AVG_AUTOEX_SIM in self.features:
                    feature_vect.append(mean(synset_sims))
                    
                feature_vects.append(feature_vect)
                
                processed += 1
                if not (processed % self.verbose_interval):
                    self.logger.debug("{}/{} done".format(processed, total))
                        
        else:
            feature_vects = [[]] * len(stimuli_response) #default to empty feature set
        
        return np.vstack(feature_vects)
    
    def get_wn_betweenness(self,stimuli_response):
        feature_vects = []
        if any([feat in self.features for feat in (FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS)]):
            #Only load betweenness if we care about betweenness features
            with open(self.betweenness_loc, "rb") as betweeneness_pkl:
                betweenness = pickle.load(betweeneness_pkl)
            
            for stimuli, response in stimuli_response:
                feature_vect = []
                
                stimuli_synsets = self._get_synset_names(self._space_to_underscore(stimuli))
                stimuli_betweennesses = []
                for stimuli_synset in stimuli_synsets:
                    stimuli_betweennesses.append(betweenness.get(stimuli_synset,0.0))
                if not stimuli_synsets:
                    #if stimuli has 0 synsets, insert a dumbie value to avoid errors
                    stimuli_betweennesses.append(0.0)
                
                response_synsets = self._get_synset_names(self._space_to_underscore(response))
                response_betweennesses = []
                for response_synset in response_synsets:
                    response_betweennesses.append(betweenness.get(response_synset,0.0))
                if not response_synsets:
                    #if respose has 0 synsets, insert a dumbie value to avoid errors
                    response_betweennesses.append(0.0)
                
                if FEAT_MAX_BETWEENNESS in self.features:
                    feature_vect.append(max(stimuli_betweennesses))
                    feature_vect.append(max(response_betweennesses))
                if FEAT_TOTAL_BETWEENNESS in  self.features:
                    feature_vect.append(sum(stimuli_betweennesses))
                    feature_vect.append(sum(response_betweennesses))
                if FEAT_AVG_BETWEENNESS in self.features:
                    feature_vect.append(mean(stimuli_betweennesses))
                    feature_vect.append(mean(response_betweennesses))
                
                feature_vects.append(feature_vect)
            
        else:
            feature_vects = [[]] * len(stimuli_response) #default to empty feature set
        
        return np.vstack(feature_vects)
    
    def get_wn_load(self,stimuli_response):
        feature_vects = []
        if any([feat in self.features for feat in (FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD)]):
                #Only load load if we care about load features and we have a location
                with open(self.load_loc, "rb") as load_pkl:
                    load = pickle.load(load_pkl)
                
                for stimuli, response in stimuli_response:
                    feature_vect = []
                    
                    stimuli_synsets = self._get_synset_names(self._space_to_underscore(stimuli))
                    stimuli_loads = []
                    for stimuli_synset in stimuli_synsets:
                        stimuli_loads.append(load.get(stimuli_synset,0.0))
                    if not stimuli_synsets:
                        #if stimuli has 0 synsets, insert a dumbie value to avoid errors
                        stimuli_loads.append(0.0)
                    
                    response_synsets = self._get_synset_names(self._space_to_underscore(response))
                    response_loads = []
                    for response_synset in response_synsets:
                        response_loads.append(load.get(response_synset,0.0))
                    if not response_synsets:
                        #if respose has 0 synsets, insert a dumbie value to avoid errors
                        response_loads.append(0.0)
                    
                    if FEAT_MAX_LOAD in self.features:
                        feature_vect.append(max(stimuli_loads))
                        feature_vect.append(max(response_loads))
                    if FEAT_TOTAL_LOAD in  self.features:
                        feature_vect.append(sum(stimuli_loads))
                        feature_vect.append(sum(response_loads))
                    if FEAT_AVG_LOAD in self.features:
                        feature_vect.append(mean(stimuli_loads))
                        feature_vect.append(mean(response_loads))
                    
                    feature_vects.append(feature_vect)
                    
        else:
            feature_vects = [[]] * len(stimuli_response) #default to empty feature set
        
        return np.vstack(feature_vects)
    
    def get_dir_rel(self,stimuli_response):
        feature_vects = []
        if FEAT_DIR_REL in self.features:
            wn_graph = WordNetGraph()
            
            total = len(stimuli_response)
            processed = 0
            for stimuli, response in stimuli_response:
                stimuli_synsets = self._get_synset_names(stimuli)
                response_synsets = self._get_synset_names(response)
                
                dirrel = wn_graph.get_directional_relativity(stimuli_synsets,response_synsets)
                feature_vects.append(dirrel)
                
                processed += 1
                if not (processed % self.verbose_interval):
                    self.logger.debug("{}/{} done".format(processed, total))
        
        else:
            feature_vects = [[]] * len(stimuli_response) #default to empty feature set
        
        return np.vstack(feature_vects)
    
    def get_w2g_feats(self,stimuli_response):
        feature_vects=[]
        if any([feat in self.features for feat in (FEAT_W2G_SIM, FEAT_W2G_ENERGY, FEAT_W2G_OFFSET,FEAT_W2G_VECTORS)]):
            #Only load Word2Gauss if we care about Word2Gauss features
            w2g = load_word2gauss_model(WIKIPEDIA_W2G, self.w2g_model_loc, self.w2g_vocab_loc)
            
            for stimuli, response in stimuli_response:
                feature_vect = []
                
                #treat stimuli/response a document and let the vocab tokenize it. This way we can capture higher order ngrams
                stimuli_as_doc = self._underscore_to_space(stimuli)
                response_as_doc = self._underscore_to_space(response)
                
                if FEAT_W2G_SIM in self.features:
                    feature_vect.append(w2g.get_similarity(stimuli_as_doc, response_as_doc))
                if FEAT_W2G_OFFSET in self.features:
                    feature_vect.append(w2g.get_offset(stimuli_as_doc, response_as_doc))
                if FEAT_W2G_VECTORS in self.features:
                    feature_vect.append(w2g.get_vector(stimuli_as_doc))
                    feature_vect.append(w2g.get_vector(response_as_doc))
                if FEAT_W2G_ENERGY in self.features:
                    #assume stimuli/response are a valid ngram
                    feature_vect.append(w2g.get_energy(self._space_to_underscore(stimuli), self._space_to_underscore(response)))
                
                feature_vects.append(np.hstack(feature_vect))
         
        else:
            feature_vects = [[]] * len(stimuli_response) #default to empty feature set
        
        return np.vstack(feature_vects)
    
    def get_wn_feats(self,stimuli_response):
        feature_vects=[]
        
        #determine which features we need to extract
        wup_needed = any([feat in self.features for feat in (FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM)])
        path_needed = any([feat in self.features for feat in (FEAT_MAX_PATH_SIM, FEAT_AVG_PATH_SIM)])
        lch_needed = any([feat in self.features for feat in (FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM)])
        lexvector_needed = FEAT_LEXVECTORS in self.features
        
        if wup_needed or path_needed or lch_needed or lexvector_needed:
            wnu = WordNetUtils(cache=True)
            
            total = len(stimuli_response)
            processed = 0
            for stimuli, response in stimuli_response:
                feature_vect = []
                
                stimuli_synsets = self._get_synsets(stimuli)
                response_synsets = self._get_synsets(response)
                
                if lexvector_needed:
                    feature_vect.append(wnu.get_lex_vector(stimuli_synsets))
                    feature_vect.append(wnu.get_lex_vector(response_synsets))
                
                wup_sims = []
                path_sims = []
                lch_sims = []
                for synset1 in stimuli_synsets:
                    for synset2 in response_synsets:
                        if wup_needed:
                            wup_sims.append(wnu.wup_similarity_w_root(synset1, synset2))
                        if path_needed:
                            path_sims.append(wnu.path_similarity_w_root(synset1, synset2))
                        if lch_needed:
                            lch_sims.append(wnu.modified_lch_similarity_w_root(synset1, synset2))
                
                #if no valid values exists, default to 0.0
                if not wup_sims:
                    wup_sims = [0.0]
                if not path_sims:
                    path_sims = [0.0]
                if not lch_sims:
                    lch_sims = [0.0]
                
                if FEAT_MAX_WUP_SIM in self.features:
                    feature_vect.append(max(wup_sims))
                if FEAT_AVG_WUP_SIM in self.features:
                    feature_vect.append(mean(wup_sims))
                if FEAT_MAX_PATH_SIM in self.features:
                    feature_vect.append(max(path_sims))
                if FEAT_AVG_PATH_SIM in self.features:
                    feature_vect.append(mean(path_sims))
                if FEAT_MAX_LCH_SIM in self.features:
                    feature_vect.append(max(lch_sims))
                if FEAT_AVG_LCH_SIM in self.features:
                    feature_vect.append(mean(lch_sims))
                
                feature_vects.append(np.hstack(feature_vect))
                
                processed+=1
                if not (processed % self.verbose_interval):
                    self.logger.debug("{}/{} done".format(processed, total))
        
        else:
            feature_vects = [[]] * len(stimuli_response) #default to empty feature set
        
        return np.vstack(feature_vects)
    
    def get_extended_lesk_feats(self, stimuli_response):
        feature_vects = []
        
        if FEAT_LESK in self.features:
            el = ExtendedLesk(self.lesk_relations, cache=True) #doesn't matter if we cache since it'll be garbage collected when we're done
            
            total = len(stimuli_response)
            processed = 0
            for stimuli, response in stimuli_response:
                #don't need to sanitize stimuli and response since I'm looking them up in wordnet anyway
                extended_lesk = el.getWordRelatedness(stimuli, response)
                
                feature_vects.append(extended_lesk)
                
                processed += 1
                if not (processed % self.verbose_interval):
                    self.logger.debug("{}/{} done".format(processed, total))
        
        else:
            feature_vects = [[]] * len(stimuli_response)
        
        return np.vstack(feature_vects)
    
    def transform(self, stimuli_response):
        stimuli_response = [(stimuli.lower(), response.lower()) for stimuli, response in stimuli_response]
        feature_vects = []
        
        self.logger.debug("starting lda")
        feature_vects.append(self.get_lda_feats(stimuli_response))
        self.logger.debug("lda done")
         
        self.logger.debug("starting autoex")
        feature_vects.append(self.get_autoex_feats(stimuli_response))
        self.logger.debug("autoex done")
         
        self.logger.debug("starting betweenness")
        feature_vects.append(self.get_wn_betweenness(stimuli_response))
        self.logger.debug("betweenness done")
        
        self.logger.debug("starting load")
        feature_vects.append(self.get_wn_load(stimuli_response))
        self.logger.debug("load done")
        
        self.logger.debug("starting w2v")
        feature_vects.append(self.get_w2v_feats(stimuli_response))
        self.logger.debug("w2v done")
        
        self.logger.debug("starting glove")
        feature_vects.append(self.get_glove_feats(stimuli_response))
        self.logger.debug("glove done")
        
        self.logger.debug("starting w2g")
        feature_vects.append(self.get_w2g_feats(stimuli_response))
        self.logger.debug("w2g done")
       
        self.logger.debug("starting dirrels")
        feature_vects.append(self.get_dir_rel(stimuli_response))
        self.logger.debug("dirrels done")
       
        self.logger.debug("starting wordnet feats")
        feature_vects.append(self.get_wn_feats(stimuli_response))
        self.logger.debug("wordnet feats done")
         
        self.logger.debug("starting extended lesk")
        feature_vects.append(self.get_extended_lesk_feats(stimuli_response))
        self.logger.debug("extended lesk done")
        
        return np.hstack(feature_vects)