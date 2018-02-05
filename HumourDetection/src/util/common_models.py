'''
    Created on Dec 28, 2017
    
    :author: Andrew Cattle <acattle@cse.ust.hk>
    
    Convenience methods for loading commonly used models 
'''
from util.gensim_wrappers.gensim_vector_models import load_gensim_vector_model
from util.word2gauss_wrapper import load_word2gauss_model
from util.gensim_wrappers.gensim_topicsum_models import load_gensim_topicsum_model,\
    TYPE_LDA, TYPE_LSI
MODEL_NAME = "model_name"
MODEL_LOC = "model_loc"

############################# Gensim Vector Models #############################
def get_google_word2vec():
    return load_gensim_vector_model("Google pretrained Word2Vec", "c:/vectors/GoogleNews-vectors-negative300.bin")

def get_stanford_glove():
    return load_gensim_vector_model("Stanford pretrained GloVe", "c:/vectors/glove.840B.300d.withheader.bin")

def get_google_autoextend():
    return load_gensim_vector_model("AutoExtend synset vectors", "c:/vectors/autoextend.word2vecformat.bin")



############################## Word2Gauss Models ###############################
def get_wikipedia_word2gauss():
    return load_word2gauss_model("Wikipedia Word2Gauss", "c:/vectors/wiki.moreselective.gz", "c:/vectors/wiki.hyperparam.selectivevocab.w2g",False)



############################ Gensim TopicSum Models ############################
#Common TFIDF model parameters
_WIKIPEDIA_TFIDF = ("Wikipedia TFIDF Model", "c:/vectors/lda_prep_no_lemma/lda_no_lemma_wordids.txt.bz2", "c:/vectors/lda_prep_no_lemma/lda_no_lemma.tfidf_model")

def get_wikipedia_lda():
    return load_gensim_topicsum_model("Wikipedia LDA", TYPE_LDA, "c:/vectors/lda_prep_no_lemma/no_lemma.101.lda", *_WIKIPEDIA_TFIDF)

def get_wikipedia_lsi():
    return load_gensim_topicsum_model("Wikipedia LSI", TYPE_LSI, "c:/vectors/lda_prep_no_lemma/no_lemma.lsi", *_WIKIPEDIA_TFIDF)
