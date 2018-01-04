'''
Created on Jan 27, 2017

@author: Andrew
'''
from __future__ import print_function, division #for Python 2.7 compatibility
import pickle
from lda_vector import LDAVectorizer
from google_word2vec import GoogleWord2Vec
from autoextend import AutoExtendEmbeddings
from wordnet_graph import WordNetGraph
from numpy import hstack, zeros, float32, empty, vstack, nan,nan_to_num,inf
import numpy as np
from nltk.corpus import wordnet as wn
import warnings
from vocab import Vocabulary
from word2gauss import GaussianEmbedding
from wordnet_utils import WordNetUtils
import re
from scipy.spatial.distance import cosine
# import os
from extended_lesk import ExtendedLesk
from util.gensim_wrappers.gensim_vector_models import load_gensim_vector_model
from util.model_name_consts import STANFORD_GLOVE, GOOGLE_W2V, AUTOEXTEND,\
    WIKIPEDIA_LDA, WIKIPEDIA_TFIDF


class FeatureExtractor(object):
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
        
#         self.lda = LDAVectorizer(lda_loc, wordids_loc, tfidf_loc)
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
    
    
    def _get_synsets(self,word):
        return wn.synsets(re.sub(" ", "_",word))
    
    def _get_synset_names(self,word):
        return [synset.name() for synset in self._get_synsets(word)]
    
    def _space_to_underscore(self, word):
        return re.sub(" ", "_",word)
    def _underscore_to_space(self, word):
        return re.sub("_", " ",word)
    
    def _add_lda_feats(self,association_tuples, verbose=True):
        if (self.lda_loc != None) and (self.wordids_loc!=None) and(self.tfidf_loc!=None):
            lda = LDAVectorizer(self.lda_loc, self.wordids_loc, self.tfidf_loc, word_separator="_")
            
            processed = 0
            total = len(association_tuples)
            for stimuli, response, features_dict, _ in association_tuples:
                features_dict["lda sim"]=lda.get_similarity(stimuli, response)
                
                processed += 1
                if verbose and (processed % 500 == 0):
                    print("{}/{} done".format(processed, total))
            
            del lda #clear from memory
        
        else:
            warnings.warn("One or more of lda_loc, wordids_loc, or tfidf_loc is unspecified. Returning blank features")
        
        return association_tuples
    
    def _add_w2v_feats(self,association_tuples):
        if (self.w2v_loc != None):
            w2v = load_gensim_vector_model(GOOGLE_W2V, self.w2v_loc)
            
            for stimuli, response, features_dict, _ in association_tuples:
                santized_stimuli = self._space_to_underscore(stimuli)
                santized_response = self._space_to_underscore(response)
                features_dict["w2v sim"] = w2v.get_similarity(santized_stimuli, santized_response)
                features_dict["w2v offset"] = w2v.get_vector(santized_stimuli) - w2v.get_vector(santized_response)
                
            w2v._purge_model() #free up memory. We can reload it from disk if needed
        
        else:
            warnings.warn("No word2vec location specified. Returning blank features")
        
        return association_tuples
    
    def _add_glove_feats(self,association_tuples):
        if (self.glove_loc != None):
            glove = load_gensim_vector_model(STANFORD_GLOVE, self.glove_loc, True) #https://radimrehurek.com/gensim/scripts/glove2word2vec.html
            
            for stimuli, response, features_dict, _ in association_tuples:
                santized_stimuli = self._space_to_underscore(stimuli)
                santized_response = self._space_to_underscore(response)
                features_dict["glove sim"] = glove.get_similarity(santized_stimuli, santized_response)
                features_dict["glove offset"] = glove.get_vector(santized_stimuli) - glove.get_vector(santized_response)
            
            glove._purge_model() #free up memory. We can reload it from disk if needed
        
        else:
            warnings.warn("No GloVe location specified. Returning blank features")
        
        return association_tuples
    
    def _add_autoex_feats(self,association_tuples):
        if (self.autoex_loc != None):
#             with open(self.autoex_loc, "rb") as autoex_pkl:
#                 autoex = pickle.load(autoex_pkl)
            autoex = load_gensim_vector_model(AUTOEXTEND, self.autoex_loc, True)
            
            for stimuli, response, features_dict, _ in association_tuples:
                stimuli_synsets = self._get_synset_names(stimuli)
                response_synsets = self._get_synset_names(response)
                
                synset_sims = []
                for stimuli_synset in stimuli_synsets:
                    for response_synset in response_synsets:
                        synset_sims.append(autoex.get_similarity(stimuli_synset, response_synset))
                
                if (len(stimuli_synsets) == 0) or (len(response_synsets) == 0):
                    #if one of the words has 0 synsets, insert a dumbie value to avoid errors
                    synset_sims.append(nan)
                
                features_dict["max autoex sim"] = max(synset_sims)
                features_dict["avg autoex sim"] = sum(synset_sims)/len(synset_sims)
            
            autoex._purge_model() #free up memory. We can reload it from disk if needed
        
        else:
            warnings.warn("No AutoEx location specified. Returning blank features")
        
        return association_tuples
    
    def _add_wn_betweenness(self,association_tuples):
        if (self.betweenness_loc != None):
            with open(self.betweenness_loc, "rb") as betweeneness_pkl:
                betweenness = pickle.load(betweeneness_pkl)
            
            for stimuli, response, features_dict, _ in association_tuples:
                stimuli_synsets = self._get_synset_names(stimuli)
                stimuli_betweennesses = []
                for stimuli_synset in stimuli_synsets:
                    stimuli_betweennesses.append(betweenness.get(stimuli_synset,0.0))
                if len(stimuli_synsets) == 0:
                    #if stimuli has 0 synsets, insert a dumbie value to avoid errors
                    stimuli_betweennesses.append(0)
                
                features_dict["max stimuli betweenness"] = max(stimuli_betweennesses)
                features_dict["total stimuli betweenness"] = sum(stimuli_betweennesses)
                features_dict["avg stimuli betweenness"] = sum(stimuli_betweennesses)/len(stimuli_betweennesses)
                
                response_synsets = self._get_synset_names(response)
                response_betweennesses = []
                for response_synset in response_synsets:
                    response_betweennesses.append(betweenness.get(response_synset,0.0))
                if len(response_synsets) == 0:
                    #if respose has 0 synsets, insert a dumbie value to avoid errors
                    response_betweennesses.append(0)
                
                features_dict["max response betweenness"] = max(response_betweennesses)
                features_dict["total response betweenness"] = sum(response_betweennesses)
                features_dict["avg response betweenness"] = sum(response_betweennesses)/len(response_betweennesses)
            
            del betweenness #clear from memory
        
        else:
            warnings.warn("No Betweenness Centrality location specified. Returning blank features")
        
        return association_tuples
    
    def _add_wn_load(self,association_tuples):
        if (self.load_loc != None):
            with open(self.load_loc, "rb") as load_pkl:
                load = pickle.load(load_pkl)
            
            for stimuli, response, features_dict, _ in association_tuples:
                stimuli_synsets = self._get_synset_names(stimuli)
                stimuli_loads = []
                for stimuli_synset in stimuli_synsets:
                    stimuli_loads.append(load.get(stimuli_synset,0.0))
                if len(stimuli_synsets) == 0:
                    #if stimuli has 0 synsets, insert a dumbie value to avoid errors
                    stimuli_loads.append(0)
                
                features_dict["max stimuli load"] = max(stimuli_loads)
                features_dict["total stimuli load"] = sum(stimuli_loads)
                features_dict["avg stimuli load"] = sum(stimuli_loads)/len(stimuli_loads)
                
                response_synsets = self._get_synset_names(response)
                response_loads = []
                for response_synset in response_synsets:
                    response_loads.append(load.get(response_synset,0.0))
                if len(response_synsets) == 0:
                    #if response has 0 synsets, insert a dumbie value to avoid errors
                    response_loads.append(0)
                
                features_dict["max response load"] = max(response_loads)
                features_dict["total response load"] = sum(response_loads)
                features_dict["avg response load"] = sum(response_loads)/len(response_loads)
            
            del load #clear from memory
        
        else:
            warnings.warn("No Load Centrality location specified. Returning blank features")
        
        return association_tuples
    
    def _add_dir_rel(self,association_tuples, verbose=True):
#         if (self.wordnetgraph_loc != None):
#             with open(self.wordnetgraph_loc, "rb") as wordnetgraph_pkl:
#                 wn_graph = pickle.load(wordnetgraph_pkl)
        wn_graph = WordNetGraph()
        
        processed = 0
        total = len(association_tuples)
        for stimuli, response, features_dict, _ in association_tuples:
            stimuli_synsets = self._get_synset_names(stimuli)
            response_synsets = self._get_synset_names(response)
            
            features_dict["dirrel"] = wn_graph.get_directional_relativity(stimuli_synsets,response_synsets)
            
            processed += 1
            if verbose and (processed % 500 == 0):
                print("{}/{} done".format(processed, total))
        
        del wn_graph #clear from memory
    
#         else:
#             warnings.warn("No WordNetGraph location specified. Returning blank features")
        
        return association_tuples
    
    def _add_w2g_feats(self,association_tuples):
        if (self.w2g_model_loc != None) and (self.w2g_vocab_loc!=None):
            voc = Vocabulary.load(self.w2g_vocab_loc)
            w2g = GaussianEmbedding.load(self.w2g_model_loc)
#             import sys;sys.path.append(r'/mnt/c/Users/Andrew/.p2/pool/plugins/org.python.pydev_6.2.0.201711281614/pysrc')
#             import pydevd;pydevd.settrace()
            
            for stimuli, response, features_dict, _ in association_tuples:
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
                    
                features_dict["w2g energy"] = -energy#energy is -KL, so -1x to make it normal KL
                features_dict["w2g sim"] = sim
                features_dict["w2g offset"] = offset
            
            del w2g #clear from memory
            del voc
        
        else:
            warnings.warn("One or more of w2g_model_loc or w2g_vocab_loc is unspecified. Returning blank features")
        
        return association_tuples
    
    def _add_wn_feats(self,association_tuples, verbose = True):
        wnu = WordNetUtils(cache=True)
#         el=None
#         if (self.lesk_relations != None):
#             el = ExtendedLesk(self.lesk_relations)
#         else:
#             warnings.warn("No lesk relation file location specified. Returning blank features")
            
        processed = 0
        total = len(association_tuples)
        for stimuli, response, features_dict, _ in association_tuples:
            stimuli_synsets = self._get_synsets(stimuli)
            response_synsets = self._get_synsets(response)
            features_dict["stimuli lexvector"] = wnu.get_lex_vector(stimuli_synsets)
            features_dict["response lexvector"] = wnu.get_lex_vector(response_synsets)
            
            wup_sims = []
            path_sims = []
            lch_sims = []
            for synset1 in stimuli_synsets:
                for synset2 in response_synsets:
                    wup_sims.append(wnu.wup_similarity_w_root(synset1, synset2))
                    path_sims.append(wnu.path_similarity_w_root(synset1, synset2))
                    lch_sims.append(wnu.modified_lch_similarity_w_root(synset1, synset2))
            
            if len(wup_sims) == 0:
                wup_sims.append(nan)
            if len(path_sims) == 0:
                path_sims.append(nan)
            if len(lch_sims) == 0:
                lch_sims.append(nan)
            
            features_dict["wup max"] = max(wup_sims)
            features_dict["wup avg"] = sum(wup_sims)/len(wup_sims)
            features_dict["path max"] = max(path_sims)
            features_dict["path avg"] = sum(path_sims)/len(path_sims)
            features_dict["lch max"] = max(lch_sims)
            features_dict["lch avg"] = sum(lch_sims)/len(lch_sims)
            
            processed+=1
            if verbose and (processed % 500 == 0):
                print("{}/{} done".format(processed, total))
        
        del wnu
        
        return association_tuples
    
    def _add_extended_lesk_feats(self, association_tuples, verbose=True):
        if (self.lesk_relations != None):
            el = ExtendedLesk(self.lesk_relations, cache=True) #doesn't matter if we cache since we deleted it when we're done.
            
            processed = 0
            total = len(association_tuples)
            for stimuli, response, features_dict, _ in association_tuples:
                #don't need to sanitize stimuli and response since I'm looking them up in wordnet anyway
                features_dict["extended lesk"] = el.getWordRelatedness(stimuli, response)
            
                processed+=1
                if verbose and (processed % 500 == 0):
                    print("{}/{} done".format(processed, total))
            
            del el #clear from memory
        
        else:
            warnings.warn("No lesk relation file location specified. Returning blank features")
        
        return association_tuples
    
    def get_features(self,stimuli_response_strength,full_dir):
        association_tuples = [ (stimuli.lower(), response.lower(), {}, strength) for stimuli, response, strength in stimuli_response_strength]
#         association_tuples = []
#         synset_map = {}
#         synset_name_map = {}
#         for stimuli, response, strength in stimuli_response_strength:
#             stimuli = stimuli.lower()
#             response = response.lower()
#             association_tuples.append( (stimuli.lower(), response.lower(), {}, strength))
#             if 
        
        import random
        random.seed(10)
        random.shuffle(association_tuples)

        strengths=[]
        word_pairs = []
        for stim,resp,feat_dict,strength in association_tuples:
            strengths.append(strength)
            word_pairs.append((stim,resp))

        strengths=vstack(strengths).astype(float32)
        strengths=strengths.reshape((strengths.shape[0],1))

        with open(os.path.join(full_dir, "strengths.pkl"), "wb") as strength_file:
            pickle.dump(strengths, strength_file)
        print("strengths saved")
        with open(os.path.join(full_dir, "word_pairs.pkl"), "wb") as strength_file:
            pickle.dump(word_pairs, strength_file)
        print("word pairs saved")
               
        association_tuples = self._add_lda_feats(association_tuples)
        print("lda done")
   
        writePickle(association_tuples, full_dir)
   
        association_tuples = self._add_wn_betweenness(association_tuples)
        print("betweenness done")
   
        writePickle(association_tuples, full_dir)
   
        association_tuples = self._add_wn_load(association_tuples)
        print("load done")
              
        writePickle(association_tuples, full_dir)
           
        association_tuples = self._add_w2v_feats(association_tuples)
        print("w2v done")
   
        writePickle(association_tuples, full_dir)
               
        association_tuples = self._add_glove_feats(association_tuples)
        print("glove done")
   
        writePickle(association_tuples, full_dir)
           
        association_tuples = self._add_autoex_feats(association_tuples)
        print("autoex done")
   
        writePickle(association_tuples, full_dir)
           
        association_tuples = self._add_dir_rel(association_tuples)
        print("dirrels done")
  
        writePickle(association_tuples, full_dir)
           
        association_tuples = self._add_wn_feats(association_tuples)
        print("wordnet feats done")
   
        writePickle(association_tuples, full_dir)
          
        association_tuples = self._add_w2g_feats(association_tuples)
        print("w2g done")
   
        writePickle(association_tuples, full_dir)
          
        association_tuples = self._add_extended_lesk_feats(association_tuples)
        print("extended lesk done")
  
        writePickle(association_tuples, full_dir)
                   
        return association_tuples

def writePickle(association_tuples, full_dir):
    #deletes features after writing them
    feat_dicts = []
    for _, _, feat_dict, _ in association_tuples:
        feat_dicts.append(feat_dict)
    feats = feat_dicts[0].keys()
    for feat in feats:
        feat_vect = []
        for feat_dict in feat_dicts:
            val = feat_dict[feat]
            if ("lexvector" not in feat) and ("offset" not in feat) and ((val == inf) or (val ==-inf)):
                val = 0
            feat_vect.append(val)
            del feat_dict[feat]
        feat_vect =nan_to_num(vstack(feat_vect).astype(float32))
    
        with open(os.path.join(full_dir, "{}.pkl".format(feat)), "wb") as feat_file:
            pickle.dump(feat_vect, feat_file)
        print("{} done: {}".format(feat, feat_vect.shape))

