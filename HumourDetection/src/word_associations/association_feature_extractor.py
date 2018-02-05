'''
Created on Jan 27, 2017

@author: Andrew Cattle <acattle@cse.ust.hk>
'''
from __future__ import print_function, division #for Python 2.7 compatibility
from nltk.corpus import wordnet as wn
from extended_lesk import ExtendedLesk
from sklearn.base import TransformerMixin
from util.wordnet.wordnet_graph import WordNetGraph
from util.wordnet.wordnet_utils import WordNetUtils
from util.misc import mean
import numpy as np
import pickle
import re
import logging

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
#TODO: LDA offset and LDA vectors?

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

#All newly implemented features should be added to this list
ALL_FEATS = [FEAT_W2V_SIM,
             FEAT_W2V_OFFSET,
             FEAT_W2V_VECTORS,
             FEAT_GLOVE_SIM,
             FEAT_GLOVE_OFFSET,
             FEAT_GLOVE_VECTORS,
             FEAT_W2G_SIM,
             FEAT_W2G_ENERGY,
             FEAT_W2G_OFFSET,
             FEAT_W2G_VECTORS,
             FEAT_MAX_AUTOEX_SIM,
             FEAT_AVG_AUTOEX_SIM,
             FEAT_LDA_SIM,
             FEAT_LEXVECTORS,
             FEAT_MAX_WUP_SIM,
             FEAT_AVG_WUP_SIM,
             FEAT_MAX_LCH_SIM,
             FEAT_AVG_LCH_SIM,
             FEAT_MAX_PATH_SIM,
             FEAT_AVG_PATH_SIM,
             FEAT_MAX_LOAD,
             FEAT_AVG_LOAD,
             FEAT_TOTAL_LOAD,
             FEAT_MAX_BETWEENNESS,
             FEAT_AVG_BETWEENNESS,
             FEAT_TOTAL_BETWEENNESS,
             FEAT_DIR_REL,
             FEAT_LESK
             ]

class AssociationFeatureExtractor(TransformerMixin):
    def __init__(self, features=DEFAULT_FEATS, lda_model=None, w2v_model=None, autoex_model=None, betweenness_loc=None, load_loc=None,  glove_model=None, w2g_model=None, lesk_relations=None, verbose=False, low_memory=True):
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
            :param glove_model: GloVe model to use for feature extraction
            :type glove_model: util.gensim_wrappers.gensim_vector_models.GensimVectorModel
            :param w2g_model: Word2Gauss model to use for feature extraction
            :type w2g_model: util.word2gauss_wrapper.Word2GaussModel
            :param lesk_relations: Location of relations.dat for use with ExtendedLesk
            :type lesk_relations: str
            :param verbose: whether verbose mode should be used or not
            :type verbose: bool
            :param low_memory: specifies whether models should be purged from memory after use. This reduces memory usage but increases disk I/O as models will need to be automatically read back from disk before next use
        """
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        self.verbose_interval = 1000
        
        self.lda_model = lda_model
        self.w2v_model = w2v_model
        self.autoex_model = autoex_model
        self.betweenness_loc = betweenness_loc
        self.load_loc = load_loc
        self.glove_model = glove_model
        self.w2g_model = w2g_model
        self.lesk_relations = lesk_relations
        self.features = self._validate_features(features)
        
        self.low_memory = low_memory
        
    
    def _validate_features(self, features):
        """
            Method for identifying and removing unknown features.
            
            If this method isn't run as part of __init__(), get_num_dimensions()
            may return an incorrect number.
            
            :raises Exception: If user specifies a feature but does not provide the required models
        """
        valid_feats = set(ALL_FEATS)
        
        unknown_feats=[]
        for feat in features:
            if feat not in valid_feats:
                self.logger.warn(f"Unknown AssociationFeatureExtractor feature '{feat}' encountered. Ignoring")
                unknown_feats.append(feat)
            
            #Ensure features have the required models
            #TODO: Should I use a custom exceptions?
            elif feat == FEAT_LDA_SIM and not self.lda_model:
                raise Exception(f"'{feat}' feature is specified but no LDA model was provided")
            elif feat in (FEAT_W2V_SIM, FEAT_W2V_OFFSET, FEAT_W2V_VECTORS) and not self.w2v_model:
                raise Exception(f"'{feat}' feature is specified but no Word2Vec model was provided")
            elif feat in (FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM) and not self.autoex_model:
                raise Exception(f"'{feat}' feature is specified but no AutoEx model was provided")
            elif feat in (FEAT_GLOVE_SIM, FEAT_GLOVE_OFFSET, FEAT_GLOVE_VECTORS) and not self.glove_model:
                raise Exception(f"'{feat}' feature is specified but no GloVe model was provided")
            elif feat in (FEAT_W2G_SIM, FEAT_W2G_OFFSET, FEAT_W2G_VECTORS) and not self.w2g_model:
                raise Exception(f"'{feat}' feature is specified but no Word2Gauss model was provided")
            elif feat in (FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD) and not self.load_loc:
                raise Exception(f"'{feat}' feature is specified but no load centrality pkl location was provided")
            elif feat in (FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS) and not self.betweenness_loc:
                raise Exception(f"'{feat}' feature is specified but no betweenness centrality pkl location was provided")
            elif feat == FEAT_LESK and not self.lesk_relations:
                raise Exception(f"'{feat}' feature is specified but no lesk relation DAT location provided")
            
        feats = set(features)
        feats.difference_update(unknown_feats)
        return feats
    
    def get_num_dimensions(self):
        """
            Returns the number of feature vector dimensions.
            
            Note this is not the same as the number of features. While most
            features are 1D, some features (e.g. FEAT_w2V_OFFSET) can be more
            than that.
            
            Assumes features have been validated during initialization
            
            :returns: the number of feature dimensions
            :rtype: int
        """
        feats_2d = set([FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS,
                        FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD])
        
        dimensions = 0
        for feature in self.features:
            if feature == FEAT_GLOVE_OFFSET:
                dimensions += self.glove_model.get_dimensions()
            elif feature == FEAT_W2V_OFFSET:
                dimensions += self.w2v_model.get_dimensions()
            elif feature == FEAT_W2G_OFFSET:
                dimensions += self.w2g_model.get_dimensions()
            elif feature == FEAT_GLOVE_VECTORS:
                dimensions += self.glove_model.get_dimensions() * 2
            elif feature == FEAT_W2V_VECTORS:
                dimensions += self.w2v_model.get_dimensions() * 2
            elif feature == FEAT_W2G_VECTORS:
                dimensions += self.w2g_model.get_dimensions() * 2
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
        if FEAT_LDA_SIM in self.features:
            #Only load LDA model if we care about LDA features
            
            for stimuli, response in stimuli_response:
                lda_sim = self.lda_model.get_similarity(stimuli, response)
                lda_sim = 0.0 if np.isnan(lda_sim) else lda_sim
                
                feature_vects.append(lda_sim)
                
            if self.low_memory:
                self.lda_model._purge_model()
        
        else:
            feature_vects = [[]] * len(stimuli_response) #default to empty feature set
        
        return np.vstack(feature_vects)
    
    def get_w2v_feats(self,stimuli_response):
        feature_vects=[]
        full_vects_needed = (FEAT_W2V_OFFSET in self.features) or (FEAT_W2V_VECTORS in self.features)
        if (FEAT_W2V_SIM in self.features) or full_vects_needed:
            #Only load Word2Vec model if we care about Word2Vec features
            
            for stimuli, response in stimuli_response:
                feature_vect = []
                santized_stimuli = self._space_to_underscore(stimuli)
                santized_response = self._space_to_underscore(response)
                
                if FEAT_W2V_SIM in self.features:
                    feature_vect.append(self.w2v_model.get_similarity(santized_stimuli, santized_response))
                
                if full_vects_needed:
                    stim_vector = self.w2v_model.get_vector(santized_stimuli)
                    resp_vector = self.w2v_model.get_vector(santized_response)
                    if FEAT_W2V_OFFSET in self.features:
                        feature_vect.append(stim_vector - resp_vector)
                    if FEAT_W2V_VECTORS in self.features:
                        feature_vect.append(stim_vector)
                        feature_vect.append(resp_vector)
                    
                feature_vects.append(np.hstack(feature_vect))
                
            if self.low_memory:
                self.w2v_model._purge_model()
                
        else:
            feature_vects = [[]] * len(stimuli_response) #default to empty feature set
        
        return np.vstack(feature_vects)
    
    def get_glove_feats(self,stimuli_response):
        feature_vects=[]
        full_vects_needed = (FEAT_GLOVE_OFFSET in self.features) or (FEAT_GLOVE_VECTORS in self.features)
        if (FEAT_GLOVE_SIM in self.features) or full_vects_needed:
            #Only load GloVe model if we care about GloVe features
            
            for stimuli, response in stimuli_response:
                feature_vect = []
                santized_stimuli = self._space_to_underscore(stimuli)
                santized_response = self._space_to_underscore(response)
                
                if FEAT_GLOVE_SIM in self.features:
                    feature_vect.append(self.glove_model.get_similarity(santized_stimuli, santized_response))
                
                if full_vects_needed:
                    stim_vector = self.glove_model.get_vector(santized_stimuli)
                    resp_vector = self.glove_model.get_vector(santized_response)
                    if FEAT_GLOVE_OFFSET in self.features:
                        feature_vect.append(stim_vector - resp_vector)
                    if FEAT_GLOVE_VECTORS in self.features:
                        feature_vect.append(stim_vector)
                        feature_vect.append(resp_vector)
                    
                feature_vects.append(np.hstack(feature_vect))
                
            if self.low_memory:
                self.glove_model._purge_model()
                  
        else:
            feature_vects = [[]] * len(stimuli_response) #default to empty feature set
        
        return np.vstack(feature_vects)
    
    def get_autoex_feats(self,stimuli_response):
        feature_vects=[]
        if (FEAT_MAX_AUTOEX_SIM in self.features) or (FEAT_AVG_AUTOEX_SIM in self.features):
            #Only load AutoExtend model if we care about AutoExtend features
            
            total = len(stimuli_response)
            processed = 0
            for stimuli, response in stimuli_response:
                feature_vect = []
                stimuli_synsets = self._get_synset_names(self._space_to_underscore(stimuli))
                response_synsets = self._get_synset_names(self._space_to_underscore(response))
                
                synset_sims = []
                for stimuli_synset in stimuli_synsets:
                    for response_synset in response_synsets:
                        synset_sims.append(self.autoex_model.get_similarity(stimuli_synset, response_synset))
                
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
                
            if self.low_memory:
                self.autoex_model._purge_model()
                        
        else:
            feature_vects = [[]] * len(stimuli_response) #default to empty feature set
        
        return np.vstack(feature_vects)
    
    def get_wn_betweenness(self,stimuli_response):
        feature_vects = []
        if FEAT_MAX_BETWEENNESS in self.features or FEAT_AVG_BETWEENNESS in self.features or FEAT_TOTAL_BETWEENNESS in self.features:
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
        if FEAT_MAX_LOAD in self.features or FEAT_AVG_LOAD in self.features or FEAT_TOTAL_LOAD in self.features:
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
        if (FEAT_W2G_SIM in self.features) or (FEAT_W2G_ENERGY in self.features) or (FEAT_W2G_OFFSET in self.features) or (FEAT_W2G_VECTORS in self.features):
            #Only load Word2Gauss if we care about Word2Gauss features
            
            for stimuli, response in stimuli_response:
                feature_vect = []
                
                #treat stimuli/response a document and let the vocab tokenize it. This way we can capture higher order ngrams
                stimuli_as_doc = self._underscore_to_space(stimuli)
                response_as_doc = self._underscore_to_space(response)
                
                if FEAT_W2G_SIM in self.features:
                    feature_vect.append(self.w2g_model.get_similarity(stimuli_as_doc, response_as_doc))
                if FEAT_W2G_OFFSET in self.features:
                    feature_vect.append(self.w2g_model.get_offset(stimuli_as_doc, response_as_doc))
                if FEAT_W2G_VECTORS in self.features:
                    feature_vect.append(self.w2g_model.get_vector(stimuli_as_doc))
                    feature_vect.append(self.w2g_model.get_vector(response_as_doc))
                if FEAT_W2G_ENERGY in self.features:
                    #assume stimuli/response are a valid ngrams
                    feature_vect.append(self.w2g_model.get_energy(self._space_to_underscore(stimuli), self._space_to_underscore(response)))
                
                feature_vects.append(np.hstack(feature_vect))
                
            if self.low_memory:
                self.w2g_model._purge_model()
         
        else:
            feature_vects = [[]] * len(stimuli_response) #default to empty feature set
        
        return np.vstack(feature_vects)
    
    def get_wn_feats(self,stimuli_response):
        feature_vects=[]
        
        #determine which features we need to extract
        wup_needed = (FEAT_MAX_WUP_SIM in self.features) or (FEAT_AVG_WUP_SIM in self.features)
        path_needed = (FEAT_MAX_PATH_SIM in self.features) or (FEAT_AVG_PATH_SIM in self.features)
        lch_needed = (FEAT_MAX_LCH_SIM in self.features) or (FEAT_AVG_LCH_SIM in self.features) 
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
        """
            Extract Association Strength features
            
            :param stimuli_response: The stimuli/response word pairs
            :type stimuli_response: Iterable[Tuple[str,str]]
            
            :returns: A numpy array representing the extracted features
            :rtype: numpy.array
        """
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