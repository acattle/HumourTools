'''
Created on Dec 8, 2017

@author: Andrew
'''
from __future__ import print_function
from lda_vector import LDAVectorizer
import pickle
from numpy import vstack, float32, hstack
import os
from google_word2vec import GoogleWord2Vec
import re
from vocab import Vocabulary
from word2gauss import GaussianEmbedding
from util.gensim_wrappers.gensim_vector_models import load_gensim_vector_model
from util.model_name_consts import GOOGLE_W2V, STANFORD_GLOVE, AUTOEXTEND,\
    WIKIPEDIA_LDA, WIKIPEDIA_TFIDF

lda_loc="/mnt/c/vectors/lda_prep_no_lemma/no_lemma.101.lda"
wordids_loc="/mnt/c/vectors/lda_prep_no_lemma/lda_no_lemma_wordids.txt.bz2"
tfidf_loc="/mnt/c/vectors/lda_prep_no_lemma/lda_no_lemma.tfidf_model"
w2v_loc="/mnt/c/vectors/GoogleNews-vectors-negative300.bin"
glove_loc="/mnt/c/vectors/glove.840B.300d.withheader.bin"
w2g_vocab_loc="/mnt/c/vectors/wiki.moreselective.gz"
w2g_model_loc="/mnt/c/vectors/wiki.hyperparam.selectivevocab.w2g"
autoex_loc = "/mnt/c/vectors/autoextend.word2vecformat.bin"
 
# lda_loc="c:/vectors/lda_prep_no_lemma/no_lemma.101.lda"
# wordids_loc="c:/vectors/lda_prep_no_lemma/lda_no_lemma_wordids.txt.bz2"
# tfidf_loc="c:/vectors/lda_prep_no_lemma/lda_no_lemma.tfidf_model"
# w2v_loc="c:/vectors/GoogleNews-vectors-negative300.bin"
# glove_loc="c:/vectors/glove.840B.300d.withheader.bin"
# w2g_vocab_loc="c:/vectors/wiki.moreselective.gz"
# w2g_model_loc="c:/vectors/wiki.hyperparam.selectivevocab.w2g"
# autoex_loc = "c:/vectors/autoextend.word2vecformat.bin"

def load_word_pairs(folder_loc):
    with open(os.path.join(folder_loc, "word_pairs.pkl"), "rb") as wp_f:
        wordpairs = pickle.load(wp_f)#, encoding="latin1")
    
    return wordpairs

def pickle_vectors(vectors_loc, vectors):
    vectors = vstack(vectors).astype(float32)
    
    with open(vectors_loc, "wb") as v_f:
        pickle.dump(vectors, v_f, protocol=2)

feature_folder = "features"
datasets = ["evoc","usf","eat"]

print("lda")
lda = LDAVectorizer(lda_loc, wordids_loc, tfidf_loc, word_separator="_")
for dataset in datasets:
    word_pairs = load_word_pairs(os.path.join(feature_folder, dataset))
     
    vectors = []
    for word1, word2 in word_pairs:
        word1 = re.sub(" ", "_", word1.split(".")[0])
        word2 = re.sub(" ", "_", word2.split(".")[0])
        w1_vec = lda.get_vector(word1)
        w2_vec = lda.get_vector(word2)
         
        vectors.append(hstack((w1_vec, w2_vec)))
     
    pickle_vectors(os.path.join(feature_folder,dataset,"lda vectors.pkl"), vectors)
    print("\t{}".format(dataset))
 
del lda

print("w2v")
w2v = load_gensim_vector_model(GOOGLE_W2V,w2v_loc)
for dataset in datasets:
    word_pairs = load_word_pairs(os.path.join(feature_folder, dataset))
    
    vectors = []
    for word1, word2 in word_pairs:
        word1 = re.sub(" ", "_", word1.split(".")[0])
        word2 = re.sub(" ", "_", word2.split(".")[0])
        w1_vec = w2v.get_vector(word1)
        w2_vec = w2v.get_vector(word2)
        
        vectors.append(hstack((w1_vec, w2_vec)))
    
    pickle_vectors(os.path.join(feature_folder,dataset,"w2v vectors.pkl"), vectors)
    print("\t{}".format(dataset))
 
del w2v

print("glove")
glove = load_gensim_vector_model(STANFORD_GLOVE, glove_loc, True)
for dataset in datasets:
    word_pairs = load_word_pairs(os.path.join(feature_folder, dataset))
     
    vectors = []
    for word1, word2 in word_pairs:
        word1 = re.sub(" ", "_", word1.split(".")[0])
        word2 = re.sub(" ", "_", word2.split(".")[0])
        w1_vec = glove.get_vector(word1)
        w2_vec = glove.get_vector(word2)
         
        vectors.append(hstack((w1_vec, w2_vec)))
     
    pickle_vectors(os.path.join(feature_folder,dataset,"glove vectors.pkl"), vectors)
    print("\t{}".format(dataset))
 
del glove

print("autoex")
autoex = load_gensim_vector_model(AUTOEXTEND, autoex_loc, True)
for dataset in datasets:
    word_pairs = load_word_pairs(os.path.join(feature_folder, dataset))
     
    vectors = []
    for word1, word2 in word_pairs:
        word1 = re.sub(" ", "_", word1.split(".")[0])
        word2 = re.sub(" ", "_", word2.split(".")[0])
        w1_vec = autoex.get_vector(word1)
        w2_vec = autoex.get_vector(word2)
         
        vectors.append(hstack((w1_vec, w2_vec)))
     
    pickle_vectors(os.path.join(feature_folder,dataset,"autoex vectors.pkl"), vectors)
    print("\t{}".format(dataset))
 
del autoex
 

print("w2g")
voc = Vocabulary.load(w2g_vocab_loc)
w2g = GaussianEmbedding.load(w2g_model_loc)    
for dataset in datasets:
    word_pairs = load_word_pairs(os.path.join(feature_folder, dataset))
      
    vectors = []
    for word1, word2 in word_pairs:
        #treat stimuli/response a document and let the vocab tokenize it. This way we can capture higher order ngrams
        word1_tokens = voc.tokenize(re.sub("_", " ", word1.split(".")[0]))
        word2_tokens = voc.tokenize(re.sub("_", " ", word2.split(".")[0]))
        #compute cosine similarity and offset between vectors
        word1_vec = w2g.phrases_to_vector([word1_tokens, []],vocab=voc)
        word2_vec = w2g.phrases_to_vector([word2_tokens,[]], vocab=voc)
         
        vectors.append(hstack((word1_vec, word2_vec)))
    
    pickle_vectors(os.path.join(feature_folder,dataset,"w2g vectors.pkl"), vectors)
    print("\t{}".format(dataset))