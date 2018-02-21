'''
    Created on Jan 25, 2017

    :author: Andrew Cattle <acattle@cse.ust.hk>
    
    This module provides a wrapper for Gensim vector space models. This allows
    models to fail silently when OOV words encountered as well as provides lazy
    loading.
    
    This module also provides singleton-like handling of Gensim vector models to
    more efficiently use system memory.
'''

from gensim.models import KeyedVectors
import numpy as np
import logging

_models = {} #holds models in form {model_name:GensimVectorModel}
#By using a module-level variable, we can easily share singleton-like instances across various other modules
#See https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons

def load_gensim_vector_model(model_name, vector_loc, binary=True, lazy_load=True):
    """
        Loads Gensim vector space model from Google-style vectors and stores it
        as a singleton-like instance.
        
        If model_name has already been loaded, the existing instance will be
        returned.
        
        :param model_name: the name of the model to be loaded
        :type model_name: str
        :param vector_loc: the location of the Google-style vectors
        :type vector_loc: str
        :param binary: whether the file specified by vector_loc is in binary format
        :type binary: bool
        :param lazy_load: specifies whether the model should be lazy_loaded
        :type lazy_load: bool
        
        :returns: the loaded model
        :rtype: GensimVectorModel
    """
    if model_name in _models:
        logging.info("'{}' already loaded. Will use existing instance.".format(model_name))
    else:
        _models[model_name] = GensimVectorModel(vector_loc, binary)
    
    return _models[model_name]

def get_gensim_vector_model(model_name):
    """
        Gets singleton-like model_name instance
        
        :param model_name: the name of the model to be returned
        :type model_name: str
        
        :returns: the model corresponding to model_name
        :rtype: GensimVectorModel
        
        :raises Exception: If model_name has not been loaded already using load_gensim_vector_model()
    """
    if model_name not in _models:
        raise Exception("Model '{}' must be initialized before use. Please call load_gensim_vector_model() first.".format(model_name))
    
    return _models[model_name]

def purge_gensim_vector_model(model_name):
    """
        Convenience method for removing model specified by model_name from
        memory.
        
        Note: model will lazy load itself back into memory from disk the next
        time it is called.
        
        :param model_name: the name of the model to be returned
        :type model_name: str
        
        :raises Exception: If model_name has not been loaded already using load_gensim_vector_model()
    """
    if model_name not in _models:
        raise Exception("Model '{}' not currently loaded. Please call load_gensim_vector_model() first.".format(model_name))
    
    _models[model_name]._purge_model()

def purge_all_gensim_vector_models():
    """
        Convenience method for removing all models from memory.
        
        Note: models will lazy load itself back into memory from disk the next
        time they are called.
    """
    for model in _models.values():
        model._purge_model()





class GensimVectorModel():
    """
        Wrapper for Gensim Word2Vec models. Provides silent failing for OOV
        words as well as lazy loading for more efficient memory management
    """
    
    #TODO: set_vector_loc()?
    
    def __init__(self, vector_loc, binary=True, lazy_load=True):
        """
            Initialize options for a Gensim vector space model
            
            :param vector_loc: the location of the Google-style vectors
            :type vector_loc: str
            :param binary: whether the file specified by vector_loc is in binary format
            :type binary: bool
            :param lazy_load: specifies whether the model should be lazy_loaded
            :type lazy_load: bool
        """
        self.vector_loc = vector_loc
        self.binary = binary
        
        self.model = None
        if not lazy_load:
            self._get_model()

    def _get_model(self):
        """
            Handles model access and lazy loading
            
            :returns: the model
            :rtype: gensim.models.KeyedVectors
        """
        if self.model == None:
            #http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
            self.model = KeyedVectors.load_word2vec_format(self.vector_loc, binary=self.binary)
        return self.model
    
    def _purge_model(self):
        """
            Removes model from active memory but still allows for it to be read
            back from disk later (assuming the files have not moved)
        """
        self.model = None
    
    def get_dimensions(self):
        """
            Get the vector size
            
            :returns: number of vector dimensions
            :rtype: int
        """
        return self._get_model().vector_size
    
    def get_vector(self, word):
        """
            Get the vector corresponding to word.
            
            Fails silently and returns an all 0 vector if word is OOV.
            
            :param word: word to retreive vector for
            :type word: str
            
            :returns: the vector corresponding to word
            :rtype: np.array
        """
        vector = None
        try:
            vector = self._get_model()[word.lower()]
        except KeyError:
            vector = np.zeros(self.get_dimensions())
        return vector
    
    def get_similarity(self, word1, word2):
        """
            Get the cosine similarity between vectors corresponding to word1
            and word2.
            
            :param word1: the first word to compare
            :type word1: str
            :param word2: the second word to compare
            :type word2: str
            
            :returns: the cosine similarity between word1 and word2
            :rtype: float
        """
        similarity = 0.0
        try:
            similarity = self._get_model().similarity(word1, word2)
        except KeyError:
            #similarity is 0
            pass
        return similarity

if __name__ == "__main__":
#     from util.model_name_consts import GOOGLE_W2V
#     vector_loc = "c:/vectors/GoogleNews-vectors-negative300.bin"
#     
#     w2v = load_gensim_vector_model(GOOGLE_W2V, vector_loc, True)
    from util.common_models import get_google_word2vec
    w2v=get_google_word2vec()
    
    oov_v = w2v.get_vector("afasdfasgasdfgasfgasdfasdfadfs") #try to get the vector for an OOV word
    
    print("all zeros? {}".format(np.array_equal(oov_v, np.zeros(300))))
    
    purge_gensim_vector_model("Google pretrained Word2Vec")
    
    import time
    time.sleep(10)
    
    w2v.get_vector("king")