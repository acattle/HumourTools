'''
    Created on Dec 28, 2017
    
    :author: Andrew Cattle <acattle@cse.ust.hk>
    
    Convenience methods for loading commonly used models 
'''
from util.model_wrappers.gensim_wrappers.gensim_vector_models import load_gensim_vector_model
from util.model_wrappers.word2gauss_wrapper import load_word2gauss_model
from util.model_wrappers.gensim_wrappers.gensim_topicsum_models import load_gensim_topicsum_model,\
    TYPE_LDA, TYPE_LSI

############################# Gensim Vector Models #############################
def get_google_word2vec(lazy_load=True):
    return load_gensim_vector_model("Google pretrained Word2Vec", "c:/vectors/GoogleNews-vectors-negative300.bin", lazy_load=lazy_load)

def get_stanford_glove(lazy_load=True):
    return load_gensim_vector_model("Stanford pretrained GloVe", "c:/vectors/glove.840B.300d.withheader.bin", lazy_load=lazy_load)

def get_google_autoextend(lazy_load=True):
    return load_gensim_vector_model("AutoExtend synset vectors", "c:/vectors/autoextend.word2vecformat.bin", lazy_load=lazy_load)





############################## Word2Gauss Models ###############################
def get_wikipedia_word2gauss(lazy_load=True):
    return load_word2gauss_model("Wikipedia Word2Gauss", "c:/vectors/wiki.hyperparam.selectivevocab.w2g", "c:/vectors/wiki.moreselective.gz", lazy_load=lazy_load)





############################ Gensim TopicSum Models ############################
#Common TFIDF model parameters
_WIKIPEDIA_TFIDF = ("Wikipedia TFIDF Model", "c:/vectors/lda_prep_no_lemma/lda_no_lemma_wordids.txt.bz2", "c:/vectors/lda_prep_no_lemma/lda_no_lemma.tfidf_model")

def get_wikipedia_lda(lazy_load=True):
    return load_gensim_topicsum_model("Wikipedia LDA", TYPE_LDA, "c:/vectors/lda_prep_no_lemma/no_lemma.101.lda", *_WIKIPEDIA_TFIDF, lazy_load=lazy_load)

def get_wikipedia_lsi(lazy_load=True):
    return load_gensim_topicsum_model("Wikipedia LSI", TYPE_LSI, "c:/vectors/lda_prep_no_lemma/no_lemma.lsi", *_WIKIPEDIA_TFIDF, lazy_load=lazy_load)
