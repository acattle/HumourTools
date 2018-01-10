'''
    Created on Jan 4, 2018
    
    :author: Andrew Cattle <acattle@cse.ust.hk>
    
    This module provides a wrapper for Gensim TFIDF models. This lazy loading.
    
    This module also provides singleton-like handling of Gensim models to more
    efficiently use system memory.
'''
import logging
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

_models =  {} #holds models in form {model_name:GensimTfidfModel}
#By using a module-level variables, we can easily share singleton-like instances across various other modules
#See https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons

def load_gensim_tfidf_model(model_name, word_ids_loc, tfidf_model_loc, tokenizer=lambda x: x.split(), cache=True, lazy_load=True):
    """
        Loads Gensim document summarization model disk from and stores it as
        a singleton-like instance.
        
        If model_name has already been loaded, the existing instance will be
        returned.
        
        :param model_name: the name of the model to be loaded
        :type model_name: str
        :param word_ids_loc: the location of the Gensim Dictionary used by the TFIDF model
        :type word_ids_loc: str
        :param tfidf_model_loc: the location of the Gensim TFIDF model
        :type tfidf_model_loc: str
        :param tokenizer: the document tokenization function. Must take a single str argument and return a list of strs
        :type tokenizer: function(str)
        :param lazy_load: specifies whether the model should be lazy_loaded
        :type lazy_load: bool
        
        :returns: the loaded model
        :rtype: GensimTfidfModel
    """
    if model_name in _models:
        logging.warning("'{}' already loaded. Will use existing instance.".format(model_name), RuntimeWarning)
    else:
        _models[model_name] = GensimTFIDFModel(word_ids_loc, tfidf_model_loc, tokenizer, cache, lazy_load)
        
    return _models[model_name]

def get_gensim_tfidf_model(model_name):
    """
        Gets singleton-like model_name instance
        
        :param model_name: the name of the model to be returned
        :type model_name: str
        
        :returns: the model corresponding to model_name
        :rtype: GensimTfidfModel
        
        :raises Exception: If model_name has not been loaded already using load_gensim_tfidf_model()
    """
    if model_name not in _models:
        raise Exception("Model '{}' must be initialized before use. Please call load_gensim_tfidf_model() first.".format(model_name))
    
    return _models[model_name]

def purge_gensim_tfidf_model(model_name):
    """
        Convenience method for removing model specified by model_name from
        memory.
        
        Note: model will lazy load itself back into memory  from disk the next
        time it is called.
        
        :param model_name: the name of the model to be returned
        :type model_name: str
        
        :raises Exception: If model_name has not been loaded already using load_gensim_tfidf_model()
    """
    if model_name not in _models:
        raise Exception("Model '{}' not currently loaded. Please call load_gensim_tfidf_model() first.".format(model_name))
    
    _models[model_name]._purge_model()

def purge_all_gensim_tfidf_models():
    """
        Convenience method for removing all models from memory.
        
        Note: models will lazy load itself back into memory from disk the next
        time they are called.
    """
    for model in _models.values():
        model._purge_model()





class GensimTFIDFModel(object):
    """
        Convenience class for wrapping Gensim TFIDF models. Provides lazy
        loading functionality for more efficient memory management.
    """
    def __init__(self, word_ids_loc, tfidf_model_loc, tokenizer=lambda x: x.split(), cache=True, lazy_load=True):
        """
            Initialize Gensim TFIDF model options
            
            :param word_ids_loc: the location of the Gensim Dictionary used by the TFIDF model
            :type word_ids_loc: str
            :param tfidf_model_loc: the location of the Gensim TFIDF model
            :type tfidf_model_loc: str
            :param tokenizer: the document tokenization function. Must take a single str argument and return a list of strs
            :type tokenizer: function(str)
            :param lazy_load: specifies whether the model should be lazy_loaded
            :type lazy_load: bool
        """
        
        self.word_ids_loc = word_ids_loc
        self.tfidf_model_loc = tfidf_model_loc
        self.tokenize = tokenizer
        
        self._cache=None
        if cache==True:
            self._cache = {}
        
        self.id2word = None
        self.tfidf = None
        if not lazy_load:
            self._get_id2word()
            self._get_tfidf()
    
    def _get_id2word(self):
        """
            Handles Dictionary access and lazy loading
            
            :returns: the word2id dictionary
            :rtype: gensim.corpora.Dictionary
        """
        if self.id2word == None:
            self.id2word = Dictionary.load_from_text(self.word_ids_loc)
        return self.id2word
    
    def _get_tfidf(self):
        """
            Handles TFIDF model access and lazy loading
            
            :returns: the TFIDF model
            :rtype: gensim.models.TfidfModel
        """
        if self.tfidf == None:
            self.tfidf = TfidfModel.load(self.tfidf_model_loc)
        return self.tfidf
    
    def _purge_model(self):
        """
            Removes model from active memory but still allows for it to be read
            back from disk later (assuming the files have not moved)
        """
        self.id2word = None
        self.tfidf = None
    
    def get_tfidf_vector(self, document):
        """
            Get TFIDF vector for document
            
            :param document: the document to calculate TFIDF vector for
            :type document: str
            
            :returns: a TFIDF representation of document
            :rtype: list
        """
        
        tfidf_vector = None
        
        if (self._cache != None) and (document in self._cache):
            tfidf_vector = self._cache[document]
        else:
            tokens = self.tokenize(document)
            bow = self._get_id2word().doc2bow(tokens)
            tfidf_vector =  self._get_tfidf()[bow]
            
            if self._cache != None:
                self._cache[document] = tfidf_vector
        
        return tfidf_vector