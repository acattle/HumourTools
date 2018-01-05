'''
    Created on Jan 4, 2018
    
    :author: Andrew Cattle <acattle@cse.ust.hk>
'''
import warnings
from word2gauss import GaussianEmbedding
from vocab import Vocabulary
from scipy.spatial.distance import cosine
import numpy as np

_models = {} #holds models in form {model_name:GensimDocSumModel}
#By using a module-level variables, we can easily share singleton-like instances across various other modules
#See https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons

def load_word2gauss_model(model_name, model_loc, vocab_loc, lazy_load=True):
    """
        Loads Word2Gauss model disk from and stores it as a singleton-like
        instance.
        
        If model_name has already been loaded, the existing instance will be
        returned.
        
        :param model_name: the name of the model to be loaded
        :type model_name: str
        :param model_loc: the location of the Word2Gauss model
        :type model_loc: str
        :param vocab_loc: the location of the Vocabulary used by the Word2Gauss model
        :type vocab_loc: str
        :param lazy_load: specifies whether the model should be lazy_loaded
        :type lazy_load: bool
        
        :returns: the loaded model
        :rtype: Word2GaussModel
    """
    if model_name in _models:
        warnings.warn("'{}' already loaded. Will use existing instance.".format(model_name), RuntimeWarning)
    else:
        _models[model_name] = Word2GaussModel(model_loc, vocab_loc, lazy_load)
        
    return _models[model_name]

def get_word2gauss_model(model_name):
    """
        Gets singleton-like model_name instance
        
        :param model_name: the name of the model to be returned
        :type model_name: str
        
        :returns: the model corresponding to model_name
        :rtype: Word2GaussModel
        
        :raises Exception: If model_name has not been loaded already using load_word2gauss_model()
    """
    if model_name not in _models:
        raise Exception("Model '{}' must be initialized before use. Please call load_word2gauss_model() first.".format(model_name))
    
    return _models[model_name]

def purge_word2gauss_model(model_name):
    """
        Convenience method for removing model specified by model_name from
        memory.
        
        Note: model will lazy load itself back into memory  from disk the next
        time it is called.
        
        :param model_name: the name of the model to be returned
        :type model_name: str
        
        :raises Exception: If model_name has not been loaded already using load_word2gauss_model()
    """
    if model_name not in _models:
        raise Exception("Model '{}' not currently loaded. Please call load_word2gauss_model() first.".format(model_name))
    
    _models[model_name]._purge_model()

class Word2GaussModel(object):
    def __init__(self, model_loc, vocab_loc, lazy_load=True):
        """
            Initialize options for a Word2Gauss model and respective Vocabulary
            
            :param model_loc: the location of the Word2Gauss model
            :type model_loc: str
            :param vocab_loc: the location of the Vocabulary used by the Word2Gauss model
            :type vocab_loc: str
            :param lazy_load: specifies whether the model should be lazy_loaded
            :type lazy_load: bool
            
            :returns: the loaded model
            :rtype: Word2GaussModel
        """
        self.vocab_loc = vocab_loc
        self.model_loc = model_loc
            
        self.voc = None
        self.w2g = None
        if not lazy_load:
            self._get_voc()
            self._get_w2g()
    
    
    def _get_w2g(self):
        """
            Handles Word2Gauss model access and lazy loading
            
            :returns: the Word2Gauss model
            :rtype: word2gauss.GaussianEmbedding
        """
        if self.w2g == None:
            self.w2g = GaussianEmbedding.load(self.model_loc)
        return self.w2g
        
    def _get_voc(self):
        """
            Handles vocabulary access and lazy loading
            
            :returns: the vocabulary
            :rtype: vocab.Vocabulary
        """
        if self.voc == None:
            self.voc = Vocabulary.load(self.vocab_loc)
        return self.voc
    
    def _purge_model(self):
        """
            Removes model from active memory but still allows for it to be read
            back from disk later (assuming the files have not moved)
        """
        self.voc = None
        self.w2g = None
    
    def get_vector(self,document):
        """
            Get document vector according to the Word2Gauss model.
            
            :param document: document to retreive vector for
            :type document: str
            
            :returns: the vector corresponding to document
            :rtype: np.array
        """
        #TODO: OOV?
        
        tokens = self._get_voc().tokenize(document)
        return self._get_w2g().phrases_to_vector([tokens, []],vocab=self._get_voc())
    
    def get_offset(self, document1, document2):
        """
            Get the offset (differecne) between vectors corresponding to
            document1 and document2.
            
            :param document1: the first document to compare
            :type document1: str
            :param document2: the second document to compare
            :type document2: str
            
            :returns: the vector offset between document1 and document2
            :rtype: np.array()
        """
        document1_tokens = self._get_voc().tokenize(document1)
        document2_tokens = self._get_voc().tokenize(document2)
        
        return self._get_w2g().phrases_to_vector([document1_tokens, document2_tokens], vocab=self._get_voc())
    
    def get_similarity(self, document1, document2):
        """
            Get the cosine similarity between vectors corresponding to
            document1 and document2.
            
            :param document1: the first document to compare
            :type document1: str
            :param document2: the second document to compare
            :type document2: str
            
            :returns: the cosine similarity between word1 and word2
            :rtype: float
        """
        sim = 1-cosine(self.get_vector(document1), self.get_vector(document2)) #since it's cosine distance
        
        return sim if not np.isnan(sim) else 0.0 #default to 0 is sim is np.nan
    
    def get_energy(self, word1, word2):
        """
            Get the cosine similarity between vectors corresponding to
            word1 and word2.
            
            Note that energy is only calcuable if both word1 and word2 are
            valid ngrams in the vocabulary. If not, an engergy of 0.0 is
            returned
            
            :param word1: the first word to compare
            :type word1: str
            :param word2: the second word to compare
            :type word2: str
            
            :returns: the energy (either KL divergence or inner product, depending on the model) between word1 and word2
            :rtype: float
        """
        energy = 0.0
        
        word1_id = None
        word2_id = None
        try:
            word1_id = self._get_voc().word2id(word1)
        except KeyError:
            warnings.warn("{} not in W2G vocab".format(word1))
        try:
            word2_id = self._get_voc().word2id(word2)
        except KeyError:
            warnings.warn("{} not in W2G vocab".format(word2))
            
        if (word1_id!=None) and (word2_id!=None):
            energy = self._get_w2g().energy(word1_id, word2_id)
        
        #energy is -KL, so -1x to make it normal KL
        return -energy

if __name__ == "__main__":
    from util.model_name_consts import WIKIPEDIA_W2G
    w2g_vocab_loc="c:/vectors/wiki.moreselective.gz"
    w2g_model_loc="c:/vectors/wiki.hyperparam.selectivevocab.w2g"
    
    w2g = load_word2gauss_model(WIKIPEDIA_W2G, w2g_model_loc, w2g_vocab_loc)
    
    oov_v = w2g.get_vector("afasdfasgasdfgasfgasdfasdfadfs") #try to get the vector for an OOV word
    
    print("all zeros? {}".format(np.array_equal(oov_v, np.zeros(300))))
    
    print("similarity queen and king? {}".format(w2g.get_similarity("queen", "king")))
    print("similarity with gibberish? {}".format(w2g.get_similarity("afasdfasgasdfgasfgasdfasdfadfs", "king")))
    print("similarity with only some gibberish? {}".format(w2g.get_similarity("afasdfasgasdfgasfgasdfasdfadfs queen", "king")))
    print("energy queen and king? {}".format(w2g.get_energy("queen", "king")))
    
    #try loading a second model
    w2g2 = load_word2gauss_model(WIKIPEDIA_W2G, w2g_model_loc, w2g_vocab_loc)
    print("same model? {}".format(w2g == w2g2))
    
    import time
    time.sleep(10)
    
    purge_word2gauss_model(WIKIPEDIA_W2G)
          
    w2g.get_vector("king")