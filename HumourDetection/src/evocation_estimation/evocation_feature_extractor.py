'''
Created on Jan 27, 2017

@author: Andrew
'''
from __future__ import print_function, division #for Python 2.7 compatibility
import pickle
import numpy as np
from nltk.corpus import wordnet as wn
import warnings
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


class EvocationFeatureExtractor(TransformerMixin):
    def __init__(self, lda_loc=None, wordids_loc=None, tfidf_loc=None, w2v_loc=None, autoex_loc=None, betweenness_loc=None, load_loc=None, wordnetgraph_loc=None, glove_loc=None, w2g_model_loc=None, w2g_vocab_loc=None, lesk_relations=None, dtype=np.float32, verbose=True):
        self.lda_loc = lda_loc
        self.wordids_loc = wordids_loc
        self.tfidf_loc = tfidf_loc
        self.w2v_loc = w2v_loc
        self.autoex_loc = autoex_loc
        self.betweenness_loc = betweenness_loc
        self.load_loc = load_loc
#         self.wordnetgraph_loc = wordnetgraph_loc
        if wordnetgraph_loc != None:
            warnings.warn("Pickling wordnet no longer necessary")
        self.glove_loc = glove_loc
        self.w2g_model_loc = w2g_model_loc
        self.w2g_vocab_loc = w2g_vocab_loc
        self.lesk_relations=lesk_relations
        
#         self.lda = GensimLDAModel(lda_loc, wordids_loc, tfidf_loc)
#         self.word2vec = GoogleWord2Vec(w2v_loc)
#         with open(autoex_loc, "rb") as autoex_pkl:
#             self.autoex = pickle.load(autoex_pkl)
#         with open(betweenness_loc, "rb") as betweenness_pkl:
#             self.betweenness = pickle.load(betweenness_pkl)
#         with open(load_loc, "rb") as load_pkl:
#             self.load = pickle.load(load_pkl)
#         with open(wordnetgraph_loc, "rb") as wordnetgraph_pkl:
#             self.wn_graph = pickle.load(wordnetgraph_pkl)
        self.dtype=dtype
        
        #https://stackoverflow.com/questions/5980042/how-to-implement-the-verbose-or-v-option-into-a-script
        self.verboseprint = print if verbose else lambda *a, **k: None
        self.verbose_interval = 1000
    
    
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
        features = []
        if (self.lda_loc != None) and (self.wordids_loc!=None) and(self.tfidf_loc!=None):
            lda = load_gensim_topicsum_model(WIKIPEDIA_LDA, TYPE_LDA, self.lda_loc, WIKIPEDIA_TFIDF, self.wordids_loc, self.tfidf_loc)
            
            
            total = len(stimuli_response)
            for stimuli, response in stimuli_response:
                lda_sim =lda.get_similarity(stimuli, response)
                
                features.append(lda_sim)
                
                if (len(features) % self.verbose_interval == 0):
                    self.verboseprint("{}/{} done".format(len(features), total))
            
            
            del lda #clear from memory
        
        else:
            warnings.warn("One or more of lda_loc, wordids_loc, or tfidf_loc is unspecified. Returning blank features")
            features = [[] * len(stimuli_response)]
        
        return np.vstack(features)
    
    def get_w2v_feats(self,stimuli_response):
        features=[]
        if (self.w2v_loc != None):
            w2v = load_gensim_vector_model(GOOGLE_W2V, self.w2v_loc)
            
            for stimuli, response in stimuli_response:
                santized_stimuli = self._space_to_underscore(stimuli)
                santized_response = self._space_to_underscore(response)
                
                w2v_sim = w2v.get_similarity(santized_stimuli, santized_response)
                w2v_offset = w2v.get_vector(santized_stimuli) - w2v.get_vector(santized_response)
                features.append(np.hstack((w2v_sim, w2v_offset)))
                
            w2v._purge_model() #free up memory. We can reload it from disk if needed
        
        else:
            warnings.warn("No word2vec location specified. Returning blank features")
            features = [[] * len(stimuli_response)]
        
        return np.vstack(features)
    
    def get_glove_feats(self,stimuli_response):
        features = []
        if (self.glove_loc != None):
            glove = load_gensim_vector_model(STANFORD_GLOVE, self.glove_loc, True) #https://radimrehurek.com/gensim/scripts/glove2word2vec.html
            
            for stimuli, response in stimuli_response:
                santized_stimuli = self._space_to_underscore(stimuli)
                santized_response = self._space_to_underscore(response)
                
                glove_sim = glove.get_similarity(santized_stimuli, santized_response)
                glove_offset = glove.get_vector(santized_stimuli) - glove.get_vector(santized_response)
                features.append(np.hstack((glove_sim, glove_offset)))
            
            glove._purge_model() #free up memory. We can reload it from disk if needed
        
        else:
            warnings.warn("No GloVe location specified. Returning blank features")
            features = [[] * len(stimuli_response)]
        
        return np.vstack(features)
    
    def get_autoex_feats(self,stimuli_response):
        features = []
        if (self.autoex_loc != None):
#             with open(self.autoex_loc, "rb") as autoex_pkl:
#                 autoex = pickle.load(autoex_pkl)
            autoex = load_gensim_vector_model(AUTOEXTEND, self.autoex_loc, True)
            
            for stimuli, response, in stimuli_response:
                stimuli_synsets = self._get_synset_names(stimuli)
                response_synsets = self._get_synset_names(response)
                
                synset_sims = []
                for stimuli_synset in stimuli_synsets:
                    for response_synset in response_synsets:
                        synset_sims.append(autoex.get_similarity(stimuli_synset, response_synset))
                
                if (len(stimuli_synsets) == 0) or (len(response_synsets) == 0):
                    #if one of the words has 0 synsets, insert a dumbie value to avoid errors
                    synset_sims.append(np.nan)
                
                max_sim = max(synset_sims)
                avg_sim = np.mean(synset_sims)
                
                features.append((max_sim,avg_sim))
                            
            autoex._purge_model() #free up memory. We can reload it from disk if needed
        
        else:
            warnings.warn("No AutoEx location specified. Returning blank features")
            features = [[] * len(stimuli_response)]
        
        return np.vstack(features)
    
    def get_wn_betweenness(self,stimuli_response):
        features = []
        if (self.betweenness_loc != None):
            with open(self.betweenness_loc, "rb") as betweeneness_pkl:
                betweenness = pickle.load(betweeneness_pkl)
            
            for stimuli, response in stimuli_response:
                stimuli_synsets = self._get_synset_names(stimuli)
                stimuli_betweennesses = []
                for stimuli_synset in stimuli_synsets:
                    stimuli_betweennesses.append(betweenness.get(stimuli_synset,0.0))
                if len(stimuli_synsets) == 0:
                    #if stimuli has 0 synsets, insert a dumbie value to avoid errors
                    stimuli_betweennesses.append(0)
                
                max_stim_betweenness = max(stimuli_betweennesses)
                total_stim_betweenness = sum(stimuli_betweennesses)
                avg_stim_betweenness = np.mean(stimuli_betweennesses)
                
                response_synsets = self._get_synset_names(response)
                response_betweennesses = []
                for response_synset in response_synsets:
                    response_betweennesses.append(betweenness.get(response_synset,0.0))
                if len(response_synsets) == 0:
                    #if respose has 0 synsets, insert a dumbie value to avoid errors
                    response_betweennesses.append(0)
                
                max_resp_betweenness = max(response_betweennesses)
                total_resp_betweenness = sum(response_betweennesses)
                avg_resp_betweenness = np.mean(response_betweennesses)
                
                features.append((max_stim_betweenness, max_resp_betweenness,
                                 total_stim_betweenness, total_resp_betweenness,
                                 avg_stim_betweenness, avg_resp_betweenness
                                 ))
            
            del betweenness #clear from memory
        
        else:
            warnings.warn("No Betweenness Centrality location specified. Returning blank features")
            features = [[] * len(stimuli_response)]
        
        return np.vstack(features)
    
    def get_wn_load(self,stimuli_response):
        features = []
        if (self.load_loc != None):
            with open(self.load_loc, "rb") as load_pkl:
                load = pickle.load(load_pkl)
            
            for stimuli, response in stimuli_response:
                stimuli_synsets = self._get_synset_names(stimuli)
                stimuli_loads = []
                for stimuli_synset in stimuli_synsets:
                    stimuli_loads.append(load.get(stimuli_synset,0.0))
                if len(stimuli_synsets) == 0:
                    #if stimuli has 0 synsets, insert a dumbie value to avoid errors
                    stimuli_loads.append(0)
                
                max_stim_load = max(stimuli_loads)
                total_stim_load = sum(stimuli_loads)
                avg_stim_load = np.mean(stimuli_loads)
                
                response_synsets = self._get_synset_names(response)
                response_loads = []
                for response_synset in response_synsets:
                    response_loads.append(load.get(response_synset,0.0))
                if len(response_synsets) == 0:
                    #if response has 0 synsets, insert a dumbie value to avoid errors
                    response_loads.append(0)
                
                max_resp_load = max(response_loads)
                total_resp_load = sum(response_loads)
                avg_resp_load = np.mean(response_loads)
                
                features.append((max_stim_load, max_resp_load,
                                 total_stim_load, total_resp_load,
                                 avg_stim_load, avg_resp_load
                                 ))
            
            del load #clear from memory
        
        else:
            warnings.warn("No Load Centrality location specified. Returning blank features")
            features = [[] * len(stimuli_response)]
        
        return np.vstack(features)
    
    def get_dir_rel(self,stimuli_response):
#         if (self.wordnetgraph_loc != None):
#             with open(self.wordnetgraph_loc, "rb") as wordnetgraph_pkl:
#                 wn_graph = pickle.load(wordnetgraph_pkl)
        wn_graph = WordNetGraph()
        features =[]
        
        total = len(stimuli_response)
        for stimuli, response in stimuli_response:
            stimuli_synsets = self._get_synset_names(stimuli)
            response_synsets = self._get_synset_names(response)
            
            dirrel = wn_graph.get_directional_relativity(stimuli_synsets,response_synsets)
            features.append(dirrel)
            
            if (len(features) % self.verbose_interval == 0):
                self.verboseprint("{}/{} done".format(len(features), total))
        
        del wn_graph #clear from memory
    
#         else:
#             warnings.warn("No WordNetGraph location specified. Returning blank features")
        
        return np.vstack(features)
    
    def get_w2g_feats(self,stimuli_response):
        features=[]
        
        if (self.w2g_model_loc != None) and (self.w2g_vocab_loc!=None):
            w2g = load_word2gauss_model(WIKIPEDIA_W2G, self.w2g_model_loc, self.w2g_vocab_loc)
#             import sys;sys.path.append(r'/mnt/c/Users/Andrew/.p2/pool/plugins/org.python.pydev_6.2.0.201711281614/pysrc')
#             import pydevd;pydevd.settrace()
            
            for stimuli, response in stimuli_response:
#                 stimuli = stimuli.encode()
#                 response= response.encode()
                #treat stimuli/response a document and let the vocab tokenize it. This way we can capture higher order ngrams
                stimuli_as_doc = self._underscore_to_space(stimuli)
                response_as_doc = self._underscore_to_space(response)
                
                w2g_sim = w2g.get_similarity(stimuli_as_doc, response_as_doc)
                w2g_offset = w2g.get_offset(stimuli_as_doc, response_as_doc)
                
                #assume stimuli/response are a valid ngram
                w2g_energy = w2g.get_energy(self._space_to_underscore(stimuli), self._space_to_underscore(response))
                
                features.append(np.hstack((w2g_energy,w2g_sim,w2g_offset)))
            
            w2g._purge_model() #clear from memory
        
        else:
            warnings.warn("One or more of w2g_model_loc or w2g_vocab_loc is unspecified. Returning blank features")
            features = [[] * len(stimuli_response)]
        
        return np.vstack(features)
    
    def get_wn_feats(self,stimuli_response):
        features=[]
        wnu = WordNetUtils(cache=True)
#         el=None
#         if (self.lesk_relations != None):
#             el = ExtendedLesk(self.lesk_relations)
#         else:
#             warnings.warn("No lesk relation file location specified. Returning blank features")
            
        total = len(stimuli_response)
        for stimuli, response in stimuli_response:
            stimuli_synsets = self._get_synsets(stimuli)
            response_synsets = self._get_synsets(response)
            
            stim_lexvector = wnu.get_lex_vector(stimuli_synsets)
            resp_lexvector = wnu.get_lex_vector(response_synsets)
            
            wup_sims = []
            path_sims = []
            lch_sims = []
            for synset1 in stimuli_synsets:
                for synset2 in response_synsets:
                    wup_sims.append(wnu.wup_similarity_w_root(synset1, synset2))
                    path_sims.append(wnu.path_similarity_w_root(synset1, synset2))
                    lch_sims.append(wnu.modified_lch_similarity_w_root(synset1, synset2))
            
            if len(wup_sims) == 0:
                wup_sims.append(np.nan)
            if len(path_sims) == 0:
                path_sims.append(np.nan)
            if len(lch_sims) == 0:
                lch_sims.append(np.nan)
            
            wup_max = max(wup_sims)
            wup_avg = np.mean(wup_sims)
            path_max = max(path_sims)
            path_avg = np.mean(path_sims)
            lch_max = max(lch_sims)
            lch_avg = np.mean(lch_sims)
            
            features.append(np.hstack((stim_lexvector, resp_lexvector,
                                       wup_max, wup_avg,
                                       path_max, path_avg,
                                       lch_max, lch_avg
                                       )))
            
            if (len(features) % self.verbose_interval == 0):
                self.verboseprint("{}/{} done".format(len(features), total))
        
        del wnu
        
        return np.vstack(features)
    
    def get_extended_lesk_feats(self, stimuli_response):
        features = []
        
        if (self.lesk_relations != None):
            el = ExtendedLesk(self.lesk_relations, cache=True) #doesn't matter if we cache since we deleted it when we're done.
            
            total = len(stimuli_response)
            for stimuli, response in stimuli_response:
                #don't need to sanitize stimuli and response since I'm looking them up in wordnet anyway
                extended_lesk = el.getWordRelatedness(stimuli, response)
                
                features.append(extended_lesk)
            
                if (len(features) % self.verbose_interval == 0):
                    self.verboseprint("{}/{} done".format(len(features), total))
            
            del el #clear from memory
        
        else:
            warnings.warn("No lesk relation file location specified. Returning blank features")
            features = [[] * len(stimuli_response)]
        
        return np.vstack(features)
    
    def transform(self, stimuli_response):
        stimuli_response = [(stimuli.lower(), response.lower()) for stimuli, response in stimuli_response]
        features = []
        
        self.verboseprint("starting lda")
        features.append(self.get_lda_feats(stimuli_response))
        self.verboseprint("lda done")
        
        self.verboseprint("starting betweenness")
        features.append(self.get_wn_betweenness(stimuli_response))
        self.verboseprint("betweenness done")
       
        self.verboseprint("starting load")
        features.append(self.get_wn_load(stimuli_response))
        self.verboseprint("load done")
       
        self.verboseprint("starting w2v")
        features.append(self.get_w2v_feats(stimuli_response))
        self.verboseprint("w2v done")
       
        self.verboseprint("starting glove")
        features.append(self.get_glove_feats(stimuli_response))
        self.verboseprint("glove done")
       
        self.verboseprint("starting autoex")
        features.append(self.get_autoex_feats(stimuli_response))
        self.verboseprint("autoex done")
       
        self.verboseprint("starting dirrels")
        features.append(self.get_dir_rel(stimuli_response))
        self.verboseprint("dirrels done")
      
        self.verboseprint("starting wordnet feats")
        features.append(self.get_wn_feats(stimuli_response))
        self.verboseprint("wordnet feats done")
        
        self.verboseprint("starting w2g")
        features.append(self.get_w2g_feats(stimuli_response))
        self.verboseprint("w2g done")
        
        self.verboseprint("starting extended lesk")
        features.append(self.get_extended_lesk_feats(stimuli_response))
        self.verboseprint("extended lesk done")
  
        return np.hstack(features)