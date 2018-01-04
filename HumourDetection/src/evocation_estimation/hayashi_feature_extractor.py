'''
Created on Jan 27, 2017

@author: Andrew
'''
from __future__ import print_function, division #for Python 2.7 compatibility
import pickle
from lda_vector import LDAVectorizer
from autoextend import AutoExtendEmbeddings
from wordnet_graph import WordNetGraph
import numpy as np
from nltk.corpus import wordnet
import warnings
from wordnet_utils import WordNetUtils
from util.gensim_wrappers.gensim_vector_models import load_gensim_vector_model
from util.model_name_consts import AUTOEXTEND, GOOGLE_W2V, WIKIPEDIA_LDA,\
    WIKIPEDIA_TFIDF
from sklearn.base import TransformerMixin


class HayashiFeatureExtractor(TransformerMixin):
    def __init__(self, lda_loc=None, wordids_loc=None, tfidf_loc=None, w2v_loc=None, autoex_loc=None, betweenness_loc=None, load_loc=None, wordnetgraph_loc=None, glove_loc=None, w2g_model_loc=None, w2g_vocab_loc=None, dtype=np.float32):
        self.lda_loc = lda_loc
        self.wordids_loc = wordids_loc
        self.tfidf_loc = tfidf_loc
        self.w2v_loc = w2v_loc
        self.autoex_loc = autoex_loc
        self.betweenness_loc = betweenness_loc
        self.load_loc = load_loc
        if wordnetgraph_loc != None:
            warnings.warn("wordnetgraph_loc is deprecated")
#         self.wordnetgraph_loc = wordnetgraph_loc
#         self.glove_loc = glove_loc
#         self.w2g_model_loc = w2g_model_loc
#         self.w2g_vocab_loc = w2g_vocab_loc
        
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
    
    
    def _get_synset(self,word):
        return wordnet.synset(word)
    
    def _get_name(self,word):
        return word.split(".", 1)[0]
    
#     def _space_to_underscore(self, word):
#         return re.sub(" ", "_",word)
    
    def get_lda_feats(self, stimuli_response, verbose=True):
        features=[]
        if (self.lda_loc != None) and (self.wordids_loc!=None) and(self.tfidf_loc!=None):
            lda = LDAVectorizer(self.lda_loc, self.wordids_loc, self.tfidf_loc)
            
            processed = 0
            total = len(stimuli_response)
            for stimuli, response in stimuli_response:
                stim_word = self._get_name(stimuli)
                resp_word = self._get_name(response)
                lda_sim=lda.get_similarity(stim_word, resp_word)
                
                features.append(lda_sim)
                
                processed += 1
                if verbose and (processed % 500 == 0):
                    print("{}/{} done".format(processed, total))
            
            del lda #clear from memory
        
        else:
            warnings.warn("One or more of lda_loc, wordids_loc, or tfidf_loc is unspecified. Returning blank features")
            features = [[] * len(stimuli_response)]
        
        return np.vstack(features)
    
    def get_w2v_feats(self, stimuli_response):
        features = []
        if (self.w2v_loc != None):
            w2v = load_gensim_vector_model(GOOGLE_W2V, self.w2v_loc)
            
            for stimuli, response in stimuli_response:
                stim_word = self._get_name(stimuli)
                resp_word = self._get_name(response)
                w2v_sim = w2v.get_similarity(stim_word, resp_word)
#                 features_dict["w2v offset"] = w2v.get_vector(stim_word) - w2v.get_vector(resp_word)
                
                features.append(w2v_sim)
#             del w2v #clear from memory
        
        else:
            warnings.warn("No word2vec location specified. Returning blank features")
            features = [[] * len(stimuli_response)]
        
        return np.vstack(features)
    
#     def get_glove_feats(self, stimuli_response):
#         if (self.glove_loc != None):
#             glove = GoogleWord2Vec(self.glove_loc, True) #https://radimrehurek.com/gensim/scripts/glove2word2vec.html
#             
#             for stimuli, response in stimuli_response:
#                 santized_stimuli = self._space_to_underscore(stimuli)
#                 santized_response = self._space_to_underscore(response)
#                 features_dict["glove sim"] = glove.get_similarity(santized_stimuli, santized_response)
#                 features_dict["glove offset"] = glove.get_vector(santized_stimuli) - glove.get_vector(santized_response)
#             
#             del glove #clear from memory
#         
#         else:
#             warnings.warn("No GloVe location specified. Returning blank features")
#         
#         return np.vstack(features)
    
    def get_autoex_feats(self, stimuli_response):
        features = []
        if (self.autoex_loc != None):
#             with open(self.autoex_loc, "rb") as autoex_pkl:
#                 autoex = pickle.load(autoex_pkl)
            autoex = load_gensim_vector_model(AUTOEXTEND, self.autoex_loc, True)
            
            for stimuli, response in stimuli_response:
                autoex_sim = autoex.get_similarity(stimuli, response)
                autoex_offset = autoex.get_vector(stimuli) - autoex.get_vector(response)
                
                features.append((autoex_sim, autoex_offset))
#             del autoex #clear from memory
        
        else:
            warnings.warn("No AutoEx location specified. Returning blank features")
            features = [[] * len(stimuli_response)]
        
        return np.vstack(features)
    
    def get_wn_betweenness(self, stimuli_response):
        features = []
        if (self.betweenness_loc != None):
            with open(self.betweenness_loc, "rb") as betweeneness_pkl:
                betweenness = pickle.load(betweeneness_pkl)
            
            for stimuli, response in stimuli_response:
                stim_betweenness = betweenness.get(stimuli,0.0)
                resp_betweenness = betweenness.get(response,0.0)
                
                features.append((stim_betweenness,resp_betweenness))
            
            del betweenness #clear from memory
        
        else:
            warnings.warn("No Betweenness Centrality location specified. Returning blank features")
            features = [[] * len(stimuli_response)]
        
        return np.vstack(features)
    
    def get_wn_load(self, stimuli_response):
        features = []
        if (self.load_loc != None):
            with open(self.load_loc, "rb") as load_pkl:
                load = pickle.load(load_pkl)
            
            for stimuli, response in stimuli_response:
                stim_load = load.get(stimuli,0.0)
                resp_load = load.get(response,0.0)
                
                features.append((stim_load, resp_load))
            
            del load #clear from memory
        
        else:
            warnings.warn("No Load Centrality location specified. Returning blank features")
            features = [[] * len(stimuli_response)]
        
        return np.vstack(features)
    
    def get_dir_rel(self, stimuli_response):
#         if (self.wordnetgraph_loc != None):
#             with open(self.wordnetgraph_loc, "rb") as wordnetgraph_pkl:
#                 wn_graph = pickle.load(wordnetgraph_pkl)
        features = []
        wn_graph = WordNetGraph()
        
        for stimuli, response in stimuli_response:
            dirrel = wn_graph.get_directional_relativity([stimuli],[response])
            
            features.append(dirrel)
        
        del wn_graph #clear from memory
        
#         else:
#             warnings.warn("No WordNetGraph location specified. Returning blank features")
        
        return np.vstack(features)
    
#     def get_w2g_feats(self, stimuli_response):
#         if (self.w2g_model_loc != None) and (self.w2g_vocab_loc!=None):
#             voc = Vocabulary.load(self.w2g_vocab_loc)
#             w2g = GaussianEmbedding.load(self.w2g_model_loc)
#             
#             for stimuli, response in stimuli_response:
#                 santized_stimuli = self._space_to_underscore(stimuli)
#                 santized_response = self._space_to_underscore(response)
#                 stimuli_id=None
#                 try:
#                     stimuli_id = voc.word2id(santized_stimuli)
#                 except KeyError:
#                     warnings.warn("{} not in W2G vocab".format(santized_stimuli))
#                 
#                 response_id=None
#                 try:
#                     response_id = voc.word2id(santized_response)
#                 except KeyError:
#                     warnings.warn("{} not in W2G vocab".format(santized_response))
#                 
#                 energy = 0
#                 sim=0
#                 if (stimuli_id!=None) and (response_id!=None):
#                     energy = w2g.energy(stimuli_id, response_id)
#                     sim=1-cosine(w2g.phrases_to_vector([[santized_stimuli], []],vocab=voc), w2g.phrases_to_vector([[santized_response],[]], vocab=voc))
#                     
#                 features_dict["w2g energy"] = -energy#energy is -KL, so -1x to make it normal KL
#                 features_dict["w2g sim"] = sim
#                 features_dict["w2g offset"] = w2g.phrases_to_vector([[santized_stimuli], [santized_response]], vocab=voc)
#             
#             del w2g #clear from memory
#             del voc
#         
#         else:
#             warnings.warn("One or more of w2g_model_loc or w2g_vocab_loc is unspecified. Returning blank features")
#         
#         return np.vstack(features)
    
    def get_wn_feats(self, stimuli_response, verbose = True):
        features = []
        wnu = WordNetUtils(cache=True)
        
        total = len(stimuli_response)
        for stimuli, response in stimuli_response:
            stimuli_synset = self._get_synset(stimuli)
            response_synset = self._get_synset(response)
            
            stim_lexvector = wnu.get_lex_vector([stimuli_synset])
            resp_lexvector = wnu.get_lex_vector([response_synset])
            wup_sim = wnu.wup_similarity_w_root(stimuli_synset, response_synset)
            
            features.append((stim_lexvector, resp_lexvector, wup_sim))
            
            if verbose and (len(features) % 500 == 0):
                print("{}/{} done".format(len(features), total))
        
        del wnu
        
        return np.vstack(features)
    
    def transform(self,stimuli_response):
        association_tuples = [ (stimuli.lower(), response.lower()) for stimuli, response in stimuli_response]
        
        features = []
        
        features.append(self.get_lda_feats(association_tuples))
        print("lda done")
                       
        features.append(self.get_w2v_feats(association_tuples))
        print("w2v done")
               
#         features.append(self.get_glove_feats(association_tuples))
#         print("glove done")
         
        features.append(self.get_autoex_feats(association_tuples))
        print("autoex done")
          
        features.append(self.get_wn_betweenness(association_tuples))
        print("betweenness done")
              
        features.append(self.get_wn_load(association_tuples))
        print("load done")
              
        features.append(self.get_dir_rel(association_tuples))
        print("dirrels done")
          
        features.append(self.get_wn_feats(association_tuples))
        print("wordnet feats done")
        
#         features.append(self.get_w2g_feats(association_tuples))
#         print("w2g done")
                   
        return np.hstack(features)