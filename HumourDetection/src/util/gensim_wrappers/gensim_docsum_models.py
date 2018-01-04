'''
Created on Jan 22, 2017

    :author: Andrew Cattle <acattle@cse.ust.hk>
    
    This module provides a wrapper for Gensim document summarization models.
    This allows models to fail silently when OOV words encountered as well as
    provides lazy loading.
    
    This module also provides singleton-like handling of Gensim models to more
    efficiently use system memory.
'''

from gensim.models import LsiModel, LdaModel
import numpy as np
from scipy.spatial.distance import cosine
import warnings
from util.gensim_wrappers.gensim_tfidf_models import load_gensim_tfidf_model

TYPE_LSI = "lsi"
TYPE_LDA = "lda"

_models = {} #holds models in form {model_name:GensimDocSumModel}
#By using a module-level variables, we can easily share singleton-like instances across various other modules
#See https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons

def load_gensim_docsum_model(model_name, model_type, model_loc, tfidf_model_name, word_ids_loc, tfidf_model_loc, dimensions=300, tokenizer=lambda x: x.split(), cache=True, lazy_load=True):
    """
        Loads Gensim document summarization model disk from and stores it as
        a singleton-like instance.
        
        If model_name has already been loaded, the existing instance will be
        returned.
        
        :param model_name: the name of the model to be loaded
        :type model_name: str
        :param model_type: the type of document summarization model. Must be "lsi" or "lda"
        :type model_type: str
        :param model_loc: the location of the Gensim Document Summarization model
        :type model_loc: str
        :param tfidf_model_name: the name of the tfidf model to be loaded
        :type tfidf_model_name: str
        :param word_ids_loc: the location of the Gensim Dictionary used by the TFIDF model
        :type word_ids_loc: str
        :param tfidf_model_loc: the location of the Gensim TFIDF model
        :type tfidf_model_loc: str
        :param dimensions: the number of dimensions of the document summarization model
        :type dimensions: int
        :param tokenizer: the document tokenization function. Must take a single str argument and return a list of strs
        :type tokenizer: function(str)
        :param cache: specifies whether doc summary results should be cached
        :type cache: bool
        :param lazy_load: specifies whether the model should be lazy_loaded
        :type lazy_load: bool
        
        :returns: the loaded model
        :rtype: GensimDocSumModel
        
        :raises ValueError: when model_type is not either "lsi" or "lda"
    """
    if model_name in _models:
        warnings.warn("'{}' already loaded. Will use existing instance.".format(model_name), RuntimeWarning)
    else:
        model_type = model_type.lower()
        if model_type == TYPE_LSI:
            _models[model_name] = GensimLSIModel(model_loc, tfidf_model_name, word_ids_loc, tfidf_model_loc, dimensions, tokenizer, cache, lazy_load)
        elif model_type == TYPE_LDA:
            _models[model_name] = GensimLDAModel(model_loc, tfidf_model_name, word_ids_loc, tfidf_model_loc, dimensions, tokenizer, cache, lazy_load)
        else:
            raise ValueError("Unknown Gensim Document Summarization Model type '{}'".format(model_type))
        
    return _models[model_name]

def get_gensim_docsum_model(model_name):
    """
        Gets singleton-like model_name instance
        
        :param model_name: the name of the model to be returned
        :type model_name: str
        
        :returns: the model corresponding to model_name
        :rtype: GensimDocSumModel
        
        :raises Exception: If model_name has not been loaded already using load_gensim_docsum_model()
    """
    if model_name not in _models:
        raise Exception("Model '{}' must be initialized before use. Please call load_gensim_docsum_model() first.".format(model_name))
    
    return _models[model_name]

def purge_gensim_vector_model(model_name):
    """
        Convenience method for removing model specified by model_name from
        memory.
        
        Note: model will lazy load itself back into memory  from disk the next
        time it is called.
        
        :param model_name: the name of the model to be returned
        :type model_name: str
        
        :raises Exception: If model_name has not been loaded already using load_gensim_docsum_model()
    """
    if model_name not in _models:
        raise Exception("Model '{}' not currently loaded. Please call load_gensim_docsum_model() first.".format(model_name))
    
    _models[model_name]._purge_model()

class GensimDocSumModel(object):
    def __init__(self, tfidf_model_name, word_ids_loc, tfidf_model_loc, dimensions=300, tokenizer=lambda x: x.split(), cache=True, lazy_Load=True):
        self.tfidf_model = load_gensim_tfidf_model(tfidf_model_name, word_ids_loc, tfidf_model_loc, tokenizer, cache, lazy_Load)
        self.dimensions = dimensions
            
        self._cache=None
        if cache==True:
            self._cache = {}
        
        self.model=None
    
    
    def _purge_model(self, tfidf_too=True):
        self.model = None
        
        if tfidf_too:
            self.tfidf_model._purge_model()
    
    def _convert_to_numpy_array(self, tuples):
        #initialize vector to all 0s
        vec = np.zeros(self.dimensions)
        
        for i, val in tuples:
            vec[i] = val
        
        return vec
    
    def get_vector(self,document):
        vector = None
        if (self._cache != None) and (document in self._cache):
            vector = self._cache[document]
        else:
            tfidf_vector = self.tfidf_model.get_tfidf_vector(document)
            tuples = self._get_model()[tfidf_vector]
            vector = self._convert_to_numpy_array(tuples)
            
            if self._cache != None:
                self._cache[document] = vector
            
        return vector
    
    def get_similarity(self, word1, word2):
        return 1-cosine(self.get_vector(word1), self.get_vector(word2)) #since it's cosine distance

class GensimLDAModel(GensimDocSumModel):
    def __init__(self, lda_fileloc, tfidf_model_name, word_ids_loc, tfidf_model_loc, dimensions=300, tokenizer=lambda x: x.split(), cache=True, lazy_load=True):
        super(GensimLDAModel, self).__init__(tfidf_model_name, word_ids_loc, tfidf_model_loc, dimensions, tokenizer, cache, lazy_load)
        
        self.lda_fileloc = lda_fileloc
        if not lazy_load:
            self._get_model()
    
    def _get_model(self):
        if self.model == None:
            self.model = LdaModel.load(self.lda_fileloc, mmap="r")
        return self.model

class GensimLSIModel(GensimDocSumModel):
    def __init__(self, lsi_fileloc, tfidf_model_name, word_ids_loc, tfidf_model_loc, dimensions=300, tokenizer=lambda x: x.split(), cache=True, lazy_load=True):
        super(GensimLSIModel, self).__init__(tfidf_model_name, word_ids_loc, tfidf_model_loc, dimensions, tokenizer, cache, lazy_load)
        
        self.lsi_fileloc = lsi_fileloc
        if not lazy_load:
            self._get_model()
        
    def _get_model(self):
        if self.model == None:
            self.model = LsiModel.load(self.lsi_fileloc)
        return self.model

#TODO: remove
# if __name__ == "__main__":
# #     lsi = GensimLSIModel(r"C:\Users\Andrew\Desktop\vectors\lda_prep_no_lemma\no_lemma.lsi", r"C:\Users\Andrew\Desktop\vectors\lda_prep_no_lemma\lda_no_lemma_wordids.txt.bz2", r"C:\Users\Andrew\Desktop\vectors\lda_prep_no_lemma\lda_no_lemma.tfidf_model")
# #     print(lsi.get_vector("The quick brown fox"))
# #     
#     lda = GensimLDAModel(r"C:\Users\Andrew\Desktop\vectors\lda_prep_no_lemma\no_lemma.lda", r"C:\Users\Andrew\Desktop\vectors\lda_prep_no_lemma\lda_no_lemma_wordids.txt.bz2", r"C:\Users\Andrew\Desktop\vectors\lda_prep_no_lemma\lda_no_lemma.tfidf_model")
# #     print(lda.get_vector("The quick brown fox"))
#     
# #     # load id->word mapping (the dictionary),
# #     id2word = gensim.corpora.Dictionary.load_from_text(bz2.BZ2File('lda_no_lemma_wordids.txt.bz2'))
# #     # load tfidf model
# #     tfidf = gensim.models.TfidfModel.load('lda_no_lemma.tfidf_model')
# #     #load TYPE_LSI model
# #     lsi = gensim.models.lsimodel.LSIModel.load("no_lemma.lsi")
# #     
# #     vector = lsi[tfidf[id2word.doc2bow(["apple"])]] #as [(topic_id, topic_score)]
#     english_stopwords = stopwords.words("english")
#     pos_to_ignore = ["D","P","X","Y", "T", "&", "~", ",", "!", "U", "E"]
#     semeval_dir = r"c:/Users/Andrew/Desktop/SemEval Data"
#     dirs = [r"trial_dir/trial_data",
#             r"train_dir/train_data",
#             r"evaluation_dir/evaluation_data"]
#     tagged_dir = "tagged"
#      
#     for d in dirs:
#         os.chdir(os.path.join(semeval_dir, d, tagged_dir))
#         for f in glob.glob("*.tsv"):
#             name = os.path.splitext(os.path.basename(f))[0]
#             hashtag = "#{}".format(re.sub("_", "", name.lower()))
#             hashtag_words = name.split("_")        
#             #remove swords that don't give you some idea of the domain
#             hashtag_words = [word.lower() for word in hashtag_words if word.lower() not in english_stopwords]
#             #the next 3 are to catch "<blank>In#Words" type hashtags
#             hashtag_words = [word for word in hashtag_words if word != "in"]
#             hashtag_words = [word for word in hashtag_words if not ((len(word) == 1) and (word.isdigit()))]
#             hashtag_words = [word for word in hashtag_words if word != "words"]
#             
#             print("{}\tprocessing {}".format(strftime("%y-%m-%d_%H:%M:%S"),name))
#             tweet_ids = []
#             tweet_tokens = []
#             tweet_pos = []
#             with codecs.open(f, "r", encoding="utf-8") as tweet_file:
#                 for line in tweet_file:
#                     line=line.strip()
#                     if line == "":
#                         continue
#                     line_split = line.split("\t")
#                     tweet_tokens.append(line_split[0].split())
#                     tweet_pos.append(line_split[1].split())
#                     tweet_ids.append(line_split[3])
#             
#             already_collected = set()
#             lines_to_rewrite = []
#             #check if file exists to avoid redoing a lot of effort
#             lda_fileloc = "{}.lda_sim".format(f)
#             if os.path.isfile(lda_fileloc): #if the file exists
#                 with codecs.open(lda_fileloc, "r", encoding="utf-8") as resume_file:
#                     header = resume_file.readline().strip()
#                     if header.split() == hashtag_words: #only if the header matches what we've extracted
#                         for line in resume_file:
#                             line_split = line.split("\t")
#                             if len(line_split) != (len(hashtag_words) +2): #if we don't have enough columns
#                                 print(u"ERROR - previously collected tweet is incomplet: {}".format(line))
#                                 continue
#                             
#                             tweet_id = line_split[0]
#                             min_val = line_split[1].split()[0]
#                             
#                             if min_val == "0":
#                                 print(u"Tweet {} has a 0 value. Will retry".format(tweet_id))
#                                 continue
#                             
#                             already_collected.add(tweet_id)
#                             lines_to_rewrite.append(line)
#             
#             done = 0
#             with codecs.open(lda_fileloc, "w", encoding="utf-8") as out_file:
#                 out_file.write(u"{}\n".format(u" ".join(hashtag_words)))
#                 for line in lines_to_rewrite:
#                     out_file.write(line)
#                 
#                 for tokens, pos, tweet_id in zip(tweet_tokens,tweet_pos, tweet_ids):
#                     if tweet_id in already_collected: #skip it if we already have a valid reading
#                         done+=1
#                         continue
#                     
#                     lda_results_by_word = []
#                     for word in hashtag_words:
#                         ldas_by_hashtag_word=[]
#                         for token, tag in zip(tokens, pos):
#                             token=token.lower()
#                             if (tag in pos_to_ignore) or (token in english_stopwords):
#                                 continue
#                             if (token == "@midnight") or (token == hashtag): #if it's the @midnight account of the game's hashtag
#                                     continue #we don't want to process it
#                             
#                             lda_val = lda.get_similarity(word, token)
#                             ldas_by_hashtag_word.append(lda_val)
#                             
#                         if len(ldas_by_hashtag_word) == 0:
#                             print(u"ERRORL no valid tokens\t{}".format(u" ".join(tokens)))
#                             ldas_by_hashtag_word = [0]
#                         
#                         
#                         lda_results_by_word.append((min(ldas_by_hashtag_word), np.mean(ldas_by_hashtag_word), max(ldas_by_hashtag_word)))
#                     
#                     mins, avgs, maxes = zip(*lda_results_by_word) #separate out the columns
#                     
#                     overall = (min(mins), np.mean(avgs), max(maxes))
#                 
#                     per_word_ldas = u"\t".join([u"{} {} {}".format(*res) for res in lda_results_by_word])
#                     overall_ldas = u"{} {} {}".format(*overall)
#                     line = u"{}\t{}\t{}\n".format(tweet_id, overall_ldas, per_word_ldas) 
#                     out_file.write(line)
#                     done+=1
#                     if done % 20 == 0:
#                         print("{}\t{}\t{} of {} completed".format(strftime("%y-%m-%d_%H:%M:%S"), name, done, len(tweet_ids)))
#             print("{}\tfinished {}".format(strftime("%y-%m-%d_%H:%M:%S"),name))