'''
Created on Jan 27, 2017

@author: Andrew
'''
from __future__ import print_function, division #for Python 2.7 compatibility
import pickle
import numpy as np
from nltk.corpus import wordnet as wn
import warnings
from vocab import Vocabulary
from word2gauss import GaussianEmbedding
import re
# import os
from extended_lesk import ExtendedLesk
from util.gensim_wrappers.gensim_vector_models import load_gensim_vector_model
from util.gensim_wrappers.gensim_docsum_models import load_gensim_docsum_model,\
    TYPE_LDA
from util.model_name_consts import STANFORD_GLOVE, GOOGLE_W2V, AUTOEXTEND,\
    WIKIPEDIA_LDA, WIKIPEDIA_TFIDF
from scipy.spatial.distance import cosine
from sklearn.base import TransformerMixin
from util.wordnet.wordnet_graph import WordNetGraph
from util.wordnet.wordnet_utils import WordNetUtils


class EvocationFeatureExtractor(TransformerMixin):
    def __init__(self, lda_loc=None, wordids_loc=None, tfidf_loc=None, w2v_loc=None, autoex_loc=None, betweenness_loc=None, load_loc=None, wordnetgraph_loc=None, glove_loc=None, w2g_model_loc=None, w2g_vocab_loc=None, lesk_relations=None, dtype=np.float32, full_dir=""):
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
        self.full_dir=full_dir
    
    
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
    
    def get_lda_feats(self,stimuli_response, verbose=True):
        features = []
        if (self.lda_loc != None) and (self.wordids_loc!=None) and(self.tfidf_loc!=None):
            lda = load_gensim_docsum_model(WIKIPEDIA_LDA, TYPE_LDA, self.lda_loc, WIKIPEDIA_TFIDF, self.wordids_loc, self.tfidf_loc)
            
            
            total = len(stimuli_response)
            for stimuli, response in stimuli_response:
                lda_sim =lda.get_similarity(stimuli, response)
                
                features.append(lda_sim)
                
                if verbose and (len(features) % 500 == 0):
                    print("{}/{} done".format(len(features), total))
            
            
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
    
    def get_dir_rel(self,stimuli_response, verbose=True):
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
            
            if verbose and (len(features) % 500 == 0):
                print("{}/{} done".format(len(features), total))
        
        del wn_graph #clear from memory
    
#         else:
#             warnings.warn("No WordNetGraph location specified. Returning blank features")
        
        return np.vstack(features)
    
    def get_w2g_feats(self,stimuli_response):
        features=[]
        
        if (self.w2g_model_loc != None) and (self.w2g_vocab_loc!=None):
            voc = Vocabulary.load(self.w2g_vocab_loc)
            w2g = GaussianEmbedding.load(self.w2g_model_loc)
#             import sys;sys.path.append(r'/mnt/c/Users/Andrew/.p2/pool/plugins/org.python.pydev_6.2.0.201711281614/pysrc')
#             import pydevd;pydevd.settrace()
            
            for stimuli, response in stimuli_response:
#                 stimuli = stimuli.encode()
#                 response= response.encode()
                #treat stimuli/response a document and let the vocab tokenize it. This way we can capture higher order ngrams
                stimuli_tokens = voc.tokenize(self._underscore_to_space(stimuli))
                response_tokens = voc.tokenize(self._underscore_to_space(response))
                #compute cosine similarity and offset between vectors
                sim = 1-cosine(w2g.phrases_to_vector([stimuli_tokens, []],vocab=voc), w2g.phrases_to_vector([response_tokens,[]], vocab=voc))
                offset = w2g.phrases_to_vector([stimuli_tokens, response_tokens], vocab=voc)
                
                #We want to get the energy function between vectors (either KL or IP depending on how we trained the model)
                #but I can't see an easy way to calculate it from the vectors themselves, 
                stimuli_id=None
                response_id=None
                try:
                    stimuli_id = voc.word2id(self._space_to_underscore(stimuli))
                except KeyError:
                    warnings.warn("{} not in W2G vocab".format(stimuli))
                
                try:
                    response_id = voc.word2id(self._space_to_underscore(response))
                except KeyError:
                    warnings.warn("{} not in W2G vocab".format(response))
                energy = 0
                if (stimuli_id!=None) and (response_id!=None):
                    energy = w2g.energy(stimuli_id, response_id)
                    
                w2g_energy = -energy #energy is -KL, so -1x to make it normal KL
                w2g_sim = sim
                w2g_offset = offset
                features.append(np.hstack((w2g_energy,w2g_sim,w2g_offset)))
            
            del w2g #clear from memory
            del voc
        
        else:
            warnings.warn("One or more of w2g_model_loc or w2g_vocab_loc is unspecified. Returning blank features")
            features = [[] * len(stimuli_response)]
        
        return np.vstack(features)
    
    def get_wn_feats(self,stimuli_response, verbose = True):
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
            
            if verbose and (len(features) % 500 == 0):
                print("{}/{} done".format(len(features), total))
        
        del wnu
        
        return np.vstack(features)
    
    def get_extended_lesk_feats(self, stimuli_response, verbose=True):
        features = []
        
        if (self.lesk_relations != None):
            el = ExtendedLesk(self.lesk_relations, cache=True) #doesn't matter if we cache since we deleted it when we're done.d
            
            total = len(stimuli_response)
            for stimuli, response in stimuli_response:
                #don't need to sanitize stimuli and response since I'm looking them up in wordnet anyway
                extended_lesk = el.getWordRelatedness(stimuli, response)
                
                features.append(extended_lesk)
            
                if verbose and (len(features) % 500 == 0):
                    print("{}/{} done".format(len(features), total))
            
            del el #clear from memory
        
        else:
            warnings.warn("No lesk relation file location specified. Returning blank features")
            features = [[] * len(stimuli_response)]
        
        return np.vstack(features)
    
    def transform(self, stimuli_response):
        stimuli_response = [ (stimuli.lower(), response.lower()) for stimuli, response in stimuli_response]
#         association_tuples = []
#         synset_map = {}
#         synset_name_map = {}
#         for stimuli, response, strength in stimuli_response_strength:
#             stimuli = stimuli.lower()
#             response = response.lower()
#             association_tuples.append( (stimuli.lower(), response.lower(), {}, strength))
#             if 
        
#         import random
#         random.seed(10)
#         random.shuffle(association_tuples)
# 
#         strengths=[]
#         word_pairs = []
#         for stimuli,responce in stimuli_response:
#             strengths.append(strength)
#             word_pairs.append((stim,resp))
# 
#         strengths=vstack(strengths).astype(float32)
#         strengths=strengths.reshape((strengths.shape[0],1))
# 
#         with open(os.path.join(self.full_dir, "strengths.pkl"), "wb") as strength_file:
#             pickle.dump(strengths, strength_file)
#         print("strengths saved")
#         with open(os.path.join(self.full_dir, "word_pairs.pkl"), "wb") as strength_file:
#             pickle.dump(word_pairs, strength_file)
#         print("word pairs saved")
        
        features = []
        
        features.append(self.get_lda_feats(stimuli_response))
        print("lda done")
   
#         writePickle(association_tuples, self.full_dir)
   
        features.append(self.get_wn_betweenness(stimuli_response))
        print("betweenness done")
   
#         writePickle(association_tuples, self.full_dir)
   
        features.append(self.get_wn_load(stimuli_response))
        print("load done")
              
#         writePickle(association_tuples, self.full_dir)
           
        features.append(self.get_w2v_feats(stimuli_response))
        print("w2v done")
   
#         writePickle(association_tuples, self.full_dir)
               
        features.append(self.get_glove_feats(stimuli_response))
        print("glove done")
   
#         writePickle(association_tuples, self.full_dir)
           
        features.append(self.get_autoex_feats(stimuli_response))
        print("autoex done")
   
#         writePickle(association_tuples, self.full_dir)
           
        features.append(self.get_dir_rel(stimuli_response))
        print("dirrels done")
  
#         writePickle(association_tuples, self.full_dir)
           
        features.append(self.get_wn_feats(stimuli_response))
        print("wordnet feats done")
   
#         writePickle(association_tuples, self.full_dir)
          
        features.append(self.get_w2g_feats(stimuli_response))
        print("w2g done")
   
#         writePickle(association_tuples, self.full_dir)
          
        features.append(self.get_extended_lesk_feats(stimuli_response))
        print("extended lesk done")
  
#         writePickle(association_tuples, self.full_dir)
        
        #TODO: include [selective] feature scaling?
        
        return np.hstack(features)

# def writePickle(association_tuples, full_dir):
#     #deletes features after writing them
#     feat_dicts = []
#     for _, _, feat_dict, _ in association_tuples:
#         feat_dicts.append(feat_dict)
#     feats = feat_dicts[0].keys()
#     for feat in feats:
#         feat_vect = []
#         for feat_dict in feat_dicts:
#             val = feat_dict[feat]
#             if ("lexvector" not in feat) and ("offset" not in feat) and ((val == np.inf) or (val ==-np.inf)):
#                 val = 0
#             feat_vect.append(val)
#             del feat_dict[feat]
#         feat_vect =np.nan_to_num(np.vstack(feat_vect).astype(np.float32))
#     
#         with open(os.path.join(full_dir, "{}.pkl".format(feat)), "wb") as feat_file:
#             pickle.dump(feat_vect, feat_file)
#         print("{} done: {}".format(feat, feat_vect.shape))

