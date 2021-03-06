'''
Created on Jan 27, 2017

@author: Andrew Cattle <acattle@connect.ust.hk>
'''
from __future__ import print_function, division #for Python 2.7 compatibility
from extended_lesk import ExtendedLesk
from sklearn.base import TransformerMixin
from util.wordnet.wordnet_graph import WordNetGraph
from util.wordnet.wordnet_utils import get_lex_vector, wup_similarity,\
    path_similarity, lch_similarity, get_synsets
from util.misc import mean
from util.loggers import LoggerMixin
import numpy as np
import pickle
import re
import logging

#TODO: Add selective feature scaling (i.e. scale everything but vectors. Or should be scale vectors too?)
#TODO: Add a lot of documentation

_underscore_pat = re.compile("_")
_space_pat = re.compile(" ")

#Word embedding features
EMBEDDING_W2V = "Word2Vec"
EMBEDDING_GLOVE = "Glove"
EMBEDDING_W2G_COS = "Word2Gauss Cosine"
EMBEDDING_W2G_KL = "Word2Gauss KL"
FEAT_EMBEDDING_SIM = "Word Embedding Similarity"
FEAT_EMBEDDING_OFFSET = "Word Embedding Offset"
FEAT_EMBEDDING_VECTORS = "Word Embedding Vectors"
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

class AssociationFeatureExtractor(TransformerMixin, LoggerMixin):
    def __init__(self, features=DEFAULT_FEATS, lda_model_getter=None, w2v_model_getter=None, autoex_model_getter=None, betweenness_loc=None, load_loc=None,  glove_model_getter=None, w2g_model_getter=None, lesk_relations=None, verbose=False, low_memory=True):
        """
            Initialize a Word Association Strength feature extractor
            
            :param features: list of association features to extract
            :type features: Iterable[str]
            :param lda_model_getter: function for retrieving LDA model to use for feature extraction
            :type lda_model_getter: Callable[[], GensimTopicSumModel]
            :param w2v_model_getter: function for retrieving Word2Vec model to use for feature extraction
            :type w2v_model_getter: Callable[[], GensimVectorModel]
            :param autoex_model_getter: function for retrieving AutoExtend model to use for feature extraction
            :type autoex_model_getter: Callable[[], GensimVectorModel]
            :param betweenness_loc: location of betweenness centrality pkl
            :type betweenness_loc: str
            :param load_loc: location of load centrality pkl
            :type load_loc: str
            :param glove_model_getter: function for retrieving GloVe model to use for feature extraction
            :type glove_model_getter: Callable[[], GensimVectorModel]
            :param w2g_model_getter: function for retrieving Word2Gauss model to use for feature extraction
            :type w2g_model_getter: Callable[[], Word2GaussModel]
            :param lesk_relations: Location of relations.dat for use with ExtendedLesk
            :type lesk_relations: str
            :param verbose: whether verbose mode should be used or not
            :type verbose: bool
            :param low_memory: specifies whether models should be purged from memory after use. This reduces memory usage but increases disk I/O as models will need to be automatically read back from disk before next use
            :type low_memory: bool
        """
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        self.verbose_interval = 1000
        
        self._get_lda_model = lda_model_getter
        self._get_w2v_model = w2v_model_getter
        self._get_autoex_model = autoex_model_getter
        self.betweenness_loc = betweenness_loc
        self.load_loc = load_loc
        self._get_glove_model = glove_model_getter
        self._get_w2g_model = w2g_model_getter
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
                self.logger.warning(f"Unknown AssociationFeatureExtractor feature '{feat}' encountered. Ignoring")
                unknown_feats.append(feat)
            
            #Ensure features have the required models
            #TODO: Should I use a custom exceptions?
            elif feat == FEAT_LDA_SIM and not self._get_lda_model:
                raise Exception(f"'{feat}' feature is specified but no LDA model was provided")
            elif feat in (FEAT_W2V_SIM, FEAT_W2V_OFFSET, FEAT_W2V_VECTORS) and not self._get_w2v_model:
                raise Exception(f"'{feat}' feature is specified but no Word2Vec model was provided")
            elif feat in (FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM) and not self._get_autoex_model:
                raise Exception(f"'{feat}' feature is specified but no AutoEx model was provided")
            elif feat in (FEAT_GLOVE_SIM, FEAT_GLOVE_OFFSET, FEAT_GLOVE_VECTORS) and not self._get_glove_model:
                raise Exception(f"'{feat}' feature is specified but no GloVe model was provided")
            elif feat in (FEAT_W2G_SIM, FEAT_W2G_OFFSET, FEAT_W2G_VECTORS) and not self._get_w2g_model:
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
                dimensions += self._get_glove_model().get_dimensions()
            elif feature == FEAT_W2V_OFFSET:
                dimensions += self._get_w2v_model().get_dimensions()
            elif feature == FEAT_W2G_OFFSET:
                dimensions += self._get_w2g_model().get_dimensions()
            elif feature == FEAT_GLOVE_VECTORS:
                dimensions += self._get_glove_model().get_dimensions() * 2
            elif feature == FEAT_W2V_VECTORS:
                dimensions += self._get_w2v_model().get_dimensions() * 2
            elif feature == FEAT_W2G_VECTORS:
                dimensions += self._get_w2g_model().get_dimensions() * 2
            elif feature == FEAT_LEXVECTORS:
                dimensions += 100
            elif feature in feats_2d:
                dimensions += 2
            else:
                dimensions += 1
        
        #TODO: unload any loaded models
        
        return dimensions
    
    def _get_synsets(self,word):
        return get_synsets(self._space_to_underscore(word)) #uses cached function
    def _space_to_underscore(self, word):
        return _space_pat.sub("_",word)
    def _underscore_to_space(self, word):
        return _underscore_pat.sub(" ",word)
    
    def _initialize_feature_vects(self,labels):
        return {label:[] for label in labels if label in self.features}
    def _convert_feats_to_numpy(self, feature_vects):
        return {feat : np.vstack(vect) for feat, vect in feature_vects.items()}
    
    def fit(self,X, y=None):
        return self
    
    def get_lda_feats(self,stimuli_response):
        """
            Get LDA similarity.
            
            :param stimuli_response: list of tuples representing the stimuli and response words
            :type stimuli_response: Iterable[Tuple[str, str]]
            
            :returns: A matrix representing the calculated LDA similarities
            :rtype: numpy.array
        """
        feature_vects = self._initialize_feature_vects([FEAT_LDA_SIM])
        if feature_vects:
            #Only load LDA model if we care about LDA features
            for stimuli, response in stimuli_response:
                lda_sim = self._get_lda_model().get_similarity(stimuli, response)
                lda_sim = 0.0 if np.isnan(lda_sim) else lda_sim
                
                feature_vects[FEAT_LDA_SIM].append(lda_sim)
                
            if self.low_memory:
                self._get_lda_model()._purge_model()
        
        return {feat : np.vstack(vect) for feat, vect in feature_vects.items()}
    
    def get_w2v_feats(self,stimuli_response):
        """
            Get Word2Vec-based features. These include similarity, vector offsets,
            and/or full vectors.
            
            :param stimuli_response: list of tuples representing the stimuli and response words
            :type stimuli_response: Iterable[Tuple[str, str]]
            
            :returns: A matrix representing the extracted Word2Vec features
            :rtype: numpy.array
        """
        feature_vects=self._initialize_feature_vects([FEAT_W2V_SIM, FEAT_W2V_OFFSET, FEAT_W2V_VECTORS])
        full_vects_needed = (FEAT_W2V_OFFSET in self.features) or (FEAT_W2V_VECTORS in self.features)
        if feature_vects:
            #Only load Word2Vec model if we care about Word2Vec features
            
            for stimuli, response in stimuli_response:
                santized_stimuli = self._space_to_underscore(stimuli)
                santized_response = self._space_to_underscore(response)
                
                if FEAT_W2V_SIM in self.features:
                    feature_vects[FEAT_W2V_SIM].append(self._get_w2v_model().get_similarity(santized_stimuli, santized_response))
                
                if full_vects_needed:
                    stim_vector = self._get_w2v_model().get_vector(santized_stimuli)
                    resp_vector = self._get_w2v_model().get_vector(santized_response)
                    if FEAT_W2V_OFFSET in self.features:
                        feature_vects[FEAT_W2V_OFFSET].append(stim_vector - resp_vector)
                    if FEAT_W2V_VECTORS in self.features:
                        feature_vects[FEAT_W2V_VECTORS].append(np.hstack([stim_vector, resp_vector]))
                    
            if self.low_memory:
                self._get_w2v_model()._purge_model()
                
        return self._convert_feats_to_numpy(feature_vects)
    
    def get_glove_feats(self,stimuli_response):
        """
            Get GloVe-based features. These include similarity, vector offsets,
            and/or full vectors.
            
            :param stimuli_response: list of tuples representing the stimuli and response words
            :type stimuli_response: Iterable[Tuple[str, str]]
            
            :returns: A matrix representing the extracted GloVe features
            :rtype: numpy.array
        """
        feature_vects=self._initialize_feature_vects([FEAT_GLOVE_SIM, FEAT_GLOVE_OFFSET, FEAT_GLOVE_VECTORS])
        full_vects_needed = (FEAT_GLOVE_OFFSET in self.features) or (FEAT_GLOVE_VECTORS in self.features)
        if feature_vects:
            #Only load GloVe model if we care about GloVe features
            
            for stimuli, response in stimuli_response:
                santized_stimuli = self._space_to_underscore(stimuli)
                santized_response = self._space_to_underscore(response)
                
                if FEAT_GLOVE_SIM in self.features:
                    feature_vects[FEAT_GLOVE_SIM].append(self._get_glove_model().get_similarity(santized_stimuli, santized_response))
                
                if full_vects_needed:
                    stim_vector = self._get_glove_model().get_vector(santized_stimuli)
                    resp_vector = self._get_glove_model().get_vector(santized_response)
                    if FEAT_GLOVE_OFFSET in self.features:
                        feature_vects[FEAT_GLOVE_OFFSET].append(stim_vector - resp_vector)
                    if FEAT_GLOVE_VECTORS in self.features:
                        feature_vects[FEAT_GLOVE_VECTORS].append(np.hstack([stim_vector, resp_vector]))
                    
            if self.low_memory:
                self._get_glove_model()._purge_model()
                
        return self._convert_feats_to_numpy(feature_vects)
    
    def get_autoex_feats(self,stimuli_response_synset_names):
        """
            Extract AutoExtend similarity features
            
            :param stimuli_response_synset_names: the names of thesynsets which each stimuli and response belong to
            :type stimuli_response_synset_names: Iterable[Tuple[Iterable[str], Iterable[str]]]
            
            :returns: A matrix representing the extracted AutoExtend features
            :rtype: numpy.array
        """
        feature_vects=self._initialize_feature_vects([FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM])
        if feature_vects:
            #Only load AutoExtend model if we care about AutoExtend features
            
            total = len(stimuli_response_synset_names)
            processed = 0
            for stimuli_synset_names, response_synset_names in stimuli_response_synset_names:
                synset_sims = [ self._get_autoex_model().get_similarity(s, r) for s in stimuli_synset_names for r in response_synset_names]
#                 for stimuli_synset in stimuli_synsets:
#                     for response_synset in response_synsets:
#                         #TODO: get the names in advance?
#                         synset_sims.append(self._get_autoex_model().get_similarity(stimuli_synset.name(), response_synset.name()))
                
                if not synset_sims:
                    #if no valid readings exist (likely because one of the words has 0 synsets), default to 0
                    synset_sims = [0.0]
                
                if FEAT_MAX_AUTOEX_SIM in self.features:
                    feature_vects[FEAT_MAX_AUTOEX_SIM].append(max(synset_sims))
                if FEAT_AVG_AUTOEX_SIM in self.features:
                    feature_vects[FEAT_AVG_AUTOEX_SIM].append(mean(synset_sims))
                    
                processed += 1
                if not (processed % self.verbose_interval):
                    self.logger.debug("{}/{} done".format(processed, total))
                
            if self.low_memory:
                self._get_autoex_model()._purge_model()
        
        return self._convert_feats_to_numpy(feature_vects)
    
    def get_wn_betweenness(self,stimuli_response_synset_names):
        """
            Extract WordNet betweenness centrality features.
            
            :param stimuli_response_synset_names: the names of the synsets which each stimuli and response belong to
            :type stimuli_response_synset_names: Iterable[Tuple[Iterable[str], Iterable[str]]]
            
            :returns: A matrix representing the extracted WordNet betweenness centralities
            :rtype: numpy.array
        """
        feature_vects = self._initialize_feature_vects([FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS])
        if feature_vects:
            #Only load betweenness if we care about betweenness features
            with open(self.betweenness_loc, "rb") as betweeneness_pkl:
                betweenness = pickle.load(betweeneness_pkl)
            
            for stimuli_synset_names, response_synset_names in stimuli_response_synset_names:
                stimuli_betweennesses = [betweenness.get(stimuli_synset_name,0.0) for stimuli_synset_name in stimuli_synset_names]
#                 for stimuli_synset_name in stimuli_synset_names:
#                     stimuli_betweennesses.append(betweenness.get(stimuli_synset_name,0.0))
                if not stimuli_synset_names:
                    #if stimuli has 0 synsets, insert a dumbie value to avoid errors
                    stimuli_betweennesses.append(0.0)
                
                response_betweennesses = [betweenness.get(response_synset_name,0.0) for response_synset_name in response_synset_names]
#                 for response_synset_name in response_synset_names:
#                     response_betweennesses.append(betweenness.get(response_synset_name,0.0))
                if not response_synset_names:
                    #if respose has 0 synsets, insert a dumbie value to avoid errors
                    response_betweennesses.append(0.0)
                
                if FEAT_MAX_BETWEENNESS in self.features:
                    feature_vects[FEAT_MAX_BETWEENNESS].append(np.hstack([max(stimuli_betweennesses),max(response_betweennesses)]))
                if FEAT_TOTAL_BETWEENNESS in  self.features:
                    feature_vects[FEAT_TOTAL_BETWEENNESS].append(np.hstack([sum(stimuli_betweennesses),sum(response_betweennesses)]))
                if FEAT_AVG_BETWEENNESS in self.features:
                    feature_vects[FEAT_AVG_BETWEENNESS].append(np.hstack([mean(stimuli_betweennesses), mean(response_betweennesses)]))
                    
        return self._convert_feats_to_numpy(feature_vects)
    
    def get_wn_load(self,stimuli_response_synset_names):
        """
            Extract WordNet load centrality features.
            
            :param stimuli_response_synset_names: the names of the synsets which each stimuli and response belong to
            :type stimuli_response_synset_names: Iterable[Tuple[Iterable[str], Iterable[str]]]
            
            :returns: A matrix representing the extracted WordNet load centralities
            :rtype: numpy.array
        """
        feature_vects = self._initialize_feature_vects([FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD])
        if feature_vects:
                #Only load load if we care about load features and we have a location
                with open(self.load_loc, "rb") as load_pkl:
                    load = pickle.load(load_pkl)
                
                for stimuli_synset_names, response_synset_names in stimuli_response_synset_names:
                    stimuli_loads = [load.get(stimuli_synset_name,0.0) for stimuli_synset_name in stimuli_synset_names]
#                     for stimuli_synset_name in stimuli_synset_names:
#                         stimuli_loads.append(load.get(stimuli_synset_name,0.0))
                    if not stimuli_synset_names:
                        #if stimuli has 0 synsets, insert a dumbie value to avoid errors
                        stimuli_loads.append(0.0)
                    
                    response_loads = [load.get(response_synset_name,0.0) for response_synset_name in response_synset_names]
#                     for response_synset_name in response_synset_names:
#                         response_loads.append(load.get(response_synset_name,0.0))
                    if not response_synset_names:
                        #if respose has 0 synsets, insert a dumbie value to avoid errors
                        response_loads.append(0.0)
                    
                    if FEAT_MAX_LOAD in self.features:
                        feature_vects[FEAT_MAX_LOAD].append(np.hstack([max(stimuli_loads),max(response_loads)]))
                    if FEAT_TOTAL_LOAD in  self.features:
                        feature_vects[FEAT_TOTAL_LOAD].append(np.hstack([sum(stimuli_loads),sum(response_loads)]))
                    if FEAT_AVG_LOAD in self.features:
                        feature_vects[FEAT_AVG_LOAD].append(np.hstack([mean(stimuli_loads),mean(response_loads)]))
        
        return self._convert_feats_to_numpy(feature_vects)
    
    def get_dir_rel(self,stimuli_response_synset_names):
        """ 
            Extract WordNet directional relativity (Hayashi, 2016) features.
            
                Hayashi, Y. (2016). Predicting the Evocation Relation between
                Lexicalized Concepts. In Proceedings of COLING 2016, the 26th
                International Conference on Computational Linguistics: Technical
                Papers (pp. 1657-1668).
            
            :param stimuli_response_synset_names: the names of the synsets which each stimuli and response belong to
            :type stimuli_response_synset_names: Iterable[Tuple[Iterable[str], Iterable[str]]]
            
            :returns: A matrix representing the extracted WordNet betweenness centralities
            :rtype: numpy.array
        """
        feature_vects = self._initialize_feature_vects([FEAT_DIR_REL])
        if feature_vects:
            wn_graph = WordNetGraph()
            
            for stimuli_names, response_names in stimuli_response_synset_names:
#                 stimuli_names = [s.name() for s in stimuli_synsets]
#                 response_names = [s.name() for s in response_synsets]
                
                dirrel = wn_graph.get_directional_relativity(stimuli_names,response_names)
                feature_vects[FEAT_DIR_REL].append(dirrel)
        
        return self._convert_feats_to_numpy(feature_vects)
    
    def get_w2g_feats(self,stimuli_response):
        """
            Get Word2Gauss-based features. These include similarity, energy,
            vector offsets, and/or full vectors.
            
            :param stimuli_response: list of tuples representing the stimuli and response words
            :type stimuli_response: Iterable[Tuple[str, str]]
            
            :returns: A matrix representing the extracted Word2Gauss features
            :rtype: numpy.array
        """
        feature_vects=self._initialize_feature_vects([FEAT_W2G_SIM,FEAT_W2G_ENERGY,FEAT_W2G_OFFSET,FEAT_W2G_VECTORS])
        if feature_vects:
            #Only load Word2Gauss if we care about Word2Gauss features
            
            for stimuli, response in stimuli_response:
                #treat stimuli/response a document and let the vocab tokenize it. This way we can capture higher order ngrams
                stimuli_as_doc = self._underscore_to_space(stimuli)
                response_as_doc = self._underscore_to_space(response)
                
                if FEAT_W2G_SIM in self.features:
                    feature_vects[FEAT_W2G_SIM].append(self._get_w2g_model().get_similarity(stimuli_as_doc, response_as_doc))
                if FEAT_W2G_OFFSET in self.features:
                    feature_vects[FEAT_W2G_OFFSET].append(self._get_w2g_model().get_offset(stimuli_as_doc, response_as_doc))
                if FEAT_W2G_VECTORS in self.features:
                    feature_vects[FEAT_W2G_VECTORS].append(np.hstack([self._get_w2g_model().get_vector(stimuli_as_doc), self._get_w2g_model().get_vector(response_as_doc)]))
                if FEAT_W2G_ENERGY in self.features:
                    #assume stimuli/response are a valid ngrams
                    feature_vects[FEAT_W2G_ENERGY].append(self._get_w2g_model().get_energy(self._space_to_underscore(stimuli), self._space_to_underscore(response)))
                
            if self.low_memory:
                self._get_w2g_model()._purge_model()
        
        return self._convert_feats_to_numpy(feature_vects)
    
    def get_wn_feats(self,stimuli_response_synsets):
        """
            Extract WordNet-based features according to the instance's feature
            list. This includes path, Wu-Palmer, and Leacock-Chandra
            similarities as well as lexvector.
            
            :param stimuli_response_synsets: the synsets which each stimuli and response belong to
            :type stimuli_response_synsets: Iterable[Tuple[Iterable[Synset], Iterable[Synset]]]
            
            :returns: A matrix representing the extracted WordNet features
            :rtype: numpy.array
        """
        feature_vects=self._initialize_feature_vects([FEAT_MAX_WUP_SIM,FEAT_AVG_WUP_SIM,FEAT_MAX_PATH_SIM,FEAT_AVG_PATH_SIM,FEAT_MAX_LCH_SIM,FEAT_AVG_LCH_SIM,FEAT_LEXVECTORS])
        
        #determine which features we need to extract
        wup_needed = (FEAT_MAX_WUP_SIM in self.features) or (FEAT_AVG_WUP_SIM in self.features)
        path_needed = (FEAT_MAX_PATH_SIM in self.features) or (FEAT_AVG_PATH_SIM in self.features)
        lch_needed = (FEAT_MAX_LCH_SIM in self.features) or (FEAT_AVG_LCH_SIM in self.features) 
        lexvector_needed = FEAT_LEXVECTORS in self.features
        
        if feature_vects:
            total = len(stimuli_response_synsets)
            processed = 0
            for stimuli_synsets, response_synsets in stimuli_response_synsets:
                if lexvector_needed:
                    feature_vects[FEAT_LEXVECTORS].append(np.hstack([get_lex_vector(stimuli_synsets), get_lex_vector(response_synsets)]))
                
                wup_sims = []
                path_sims = []
                lch_sims = []
                for synset1 in stimuli_synsets:
                    for synset2 in response_synsets:
                        if wup_needed:
                            wup_sims.append(wup_similarity(synset1, synset2, simulate_root=True))
                        if path_needed:
                            path_sims.append(path_similarity(synset1, synset2, simulate_root=True))
                        if lch_needed:
                            lch_sims.append(lch_similarity(synset1, synset2, simulate_root=True))
                
                #if no valid values exists, default to 0.0
                if not wup_sims:
                    wup_sims = [0.0]
                if not path_sims:
                    path_sims = [0.0]
                if not lch_sims:
                    lch_sims = [0.0]
                
                if FEAT_MAX_WUP_SIM in self.features:
                    feature_vects[FEAT_MAX_WUP_SIM].append(max(wup_sims))
                if FEAT_AVG_WUP_SIM in self.features:
                    feature_vects[FEAT_AVG_WUP_SIM].append(mean(wup_sims))
                if FEAT_MAX_PATH_SIM in self.features:
                    feature_vects[FEAT_MAX_PATH_SIM].append(max(path_sims))
                if FEAT_AVG_PATH_SIM in self.features:
                    feature_vects[FEAT_AVG_PATH_SIM].append(mean(path_sims))
                if FEAT_MAX_LCH_SIM in self.features:
                    feature_vects[FEAT_MAX_LCH_SIM].append(max(lch_sims))
                if FEAT_AVG_LCH_SIM in self.features:
                    feature_vects[FEAT_AVG_LCH_SIM].append(mean(lch_sims))
                
                processed+=1
                if not (processed % self.verbose_interval):
                    self.logger.debug("{}/{} done".format(processed, total))
        
        return self._convert_feats_to_numpy(feature_vects)
    
    def get_extended_lesk_feats(self, stimuli_response_synsets):
        """
            Extract Extended Lesk relatedness. This feature is only extracted if
            FEAT_LESK exists in the feature list specified during __init__().
            Calculates Extended Lesk according to the relations DAT file
            specfied during __init__().
            
            :param stimuli_response_synsets: the synsets which each stimuli and response belong to
            :type stimuli_response_synsets: Iterable[Tuple[Iterable[Synset], Iterable[Synset]]]
            
            :returns: A matrix representing the extracted Extended Lesk relatedness
            :rtype: numpy.array
        """
        feature_vects = self._initialize_feature_vects([FEAT_LESK])
        if feature_vects:
            el = ExtendedLesk(self.lesk_relations)
            
            total = len(stimuli_response_synsets)
            processed = 0
            for stimuli_synsets, response_synsets in stimuli_response_synsets:
                #don't need to sanitize stimuli and response since I'm looking them up in wordnet anyway
                extended_lesk = el.getSynsetRelatedness(stimuli_synsets, response_synsets)
                
                feature_vects[FEAT_LESK].append(extended_lesk)
                
                processed += 1
                if not (processed % self.verbose_interval):
                    self.logger.debug("{}/{} done".format(processed, total))
        
        return self._convert_feats_to_numpy(feature_vects)
    
    def extract_features(self, stimuli_response):
        """
            Extract Association Strength features
            
            :param stimuli_response: The stimuli/response word pairs
            :type stimuli_response: Iterable[Tuple[str,str]]
            
            :returns: A numpy array representing the extracted features
            :rtype: numpy.array
        """
        feature_vects = {}
        
        stimuli_response = [(stimuli.lower(), response.lower()) for stimuli, response in stimuli_response]
        
        #raw word features
        for name, func in [("lda", self.get_lda_feats), ("w2v", self.get_w2v_feats), ("glove", self.get_glove_feats), ("w2g", self.get_w2g_feats)]:
            self.logger.debug(f"starting {name}")
            feature_vects.update(func(stimuli_response))
            self.logger.debug(f"{name} done")
        

        
        #prefetch all the synsets. Should speed things up
        #TODO: use a Dict? to reduce the number of wordnet.synsets calls further?
        stimuli_response_synsets = [(self._get_synsets(stimuli), self._get_synsets(response)) for stimuli, response in stimuli_response]
        
        for name, func in [("wordnet feats", self.get_wn_feats), ("extended lesk", self.get_extended_lesk_feats)]:
            self.logger.debug(f"starting {name}")
            feature_vects.update(func(stimuli_response_synsets))
            self.logger.debug(f"{name} done")
       
        #prefetch the names
        stimuli_response_synset_names = [([s.name() for s in stimuli_syns], [r.name() for r in response_syns]) for stimuli_syns, response_syns in stimuli_response_synsets]
        
        for name, func in [("autoex", self.get_autoex_feats), ("betweenness", self.get_wn_betweenness), ("load", self.get_wn_load), ("dirrel", self.get_dir_rel)]:
            self.logger.debug(f"starting {name}")
            feature_vects.update(func(stimuli_response_synset_names))
            self.logger.debug(f"{name} done")
        
        #failsafe to ensure make sure there's no nan or inf that would hurt training
        for feat in feature_vects.values():
            #np.nan_to_num won't work because the in/-inf values still hurt training
            feat[np.logical_not(np.isfinite(feat))] = 0
        
        return feature_vects
        
    def transform(self, stimuli_response):
        feature_vects = self.extract_features(stimuli_response)

        #enforce cinsistent word order
        sorted_keys = sorted(self.features)
        feature_vects = [feature_vects[key] for key in sorted_keys]
        
        return np.hstack(feature_vects)
    