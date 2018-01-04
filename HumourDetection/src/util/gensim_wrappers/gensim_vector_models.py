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
import warnings

_models = {} #holds models in form {model_name:GensimVectorModel}
#By using a module-level variable, we can easily share singleton-like instances across various other modules
#See https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons

def load_gensim_vector_model(model_name, vector_loc, binary=True):
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
        
        :returns: the loaded model
        :rtype: GensimVectorModel
    """
    if model_name in _models:
        warnings.warn("'{}' already loaded. Will use existing instance.".format(model_name), RuntimeWarning)
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
        
        Note: model will lazy load itself back into memory  from disk the next
        time it is called.
        
        :param model_name: the name of the model to be returned
        :type model_name: str
        
        :raises Exception: If model_name has not been loaded already using load_gensim_vector_model()
    """
    if model_name not in _models:
        raise Exception("Model '{}' not currently loaded. Please call load_gensim_vector_model() first.".format(model_name))
    
    _models[model_name]._purge_model()

class GensimVectorModel():
    """
        Wrapper for Gensim Word2Vec models. Provides silent failing for OOV
        words as well as lazy loading for more efficient memory management
    """
    
    def __init__(self, vector_loc, binary=True, lazy_load=True):
        self.vector_loc = vector_loc
        self.binary = binary
        
        self.model = None
        if not lazy_load:
            self._get_model()

    def _get_model(self):
        if self.model == None:
            #http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
            self.model = KeyedVectors.load_word2vec_format(self.vector_loc, binary=self.binary)
        return self.model
    
    def _purge_model(self):
        self.model = None
    
    def get_vector(self, word):
        vector = None
        try:
            vector = self._get_model()[word.lower()]
        except KeyError:
            vector = np.zeros(300)
        return vector
    
    def get_similarity(self, word1, word2):
        similarity = 0.0
        try:
            similarity = self._get_model().similarity(word1, word2)
        except KeyError:
            #similarity is 0
            pass
        return similarity