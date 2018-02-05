'''
Created on Apr 8, 2017

@author: Andrew
'''
from __future__ import print_function #for Python 2.7 compatibility
from word_associations.association_readers import EATGraph, USFGraph,EvocationDataset
from word_associations.association_feature_extractor import AssociationFeatureExtractor
from numpy import float32
import pickle
from util.common_models import get_wikipedia_word2gauss, get_stanford_glove,\
    get_google_autoextend, get_google_word2vec, get_wikipedia_lda



lda_loc="/mnt/c/vectors/lda_prep_no_lemma/no_lemma.101.lda"
wordids_loc="/mnt/c/vectors/lda_prep_no_lemma/lda_no_lemma_wordids.txt.bz2"
tfidf_loc="/mnt/c/vectors/lda_prep_no_lemma/lda_no_lemma.tfidf_model"
w2v_loc="/mnt/c/vectors/GoogleNews-vectors-negative300.bin"
glove_loc="/mnt/c/vectors/glove.840B.300d.withheader.bin"
# w2g_model_loc="/mnt/c/vectors/wiki.biggervocab.w2g"
# w2g_vocab_loc="/mnt/c/vectors/wiki.biggersize.gz"
w2g_vocab_loc="/mnt/c/vectors/wiki.moreselective.gz"
w2g_model_loc="/mnt/c/vectors/wiki.hyperparam.selectivevocab.w2g"
autoex_loc = "/mnt/c/vectors/autoextend.word2vecformat.bin"
lesk_loc = "/mnt/d/workspace/PyWordNetSimilarity/src/lesk-relation.dat"
 
# lda_loc="c:/vectors/lda_prep_no_lemma/no_lemma.101.lda"
# wordids_loc="c:/vectors/lda_prep_no_lemma/lda_no_lemma_wordids.txt.bz2"
# tfidf_loc="c:/vectors/lda_prep_no_lemma/lda_no_lemma.tfidf_model"
# w2v_loc="c:/vectors/GoogleNews-vectors-negative300.bin"
# glove_loc="c:/vectors/glove.840B.300d.withheader.bin"
# # w2g_model_loc="c:/vectors/wiki.biggervocab.w2g"
# # w2g_vocab_loc="c:/vectors/wiki.biggersize.gz"
# w2g_vocab_loc="c:/vectors/wiki.moreselective.gz"
# w2g_model_loc="c:/vectors/wiki.hyperparam.selectivevocab.w2g"
# # autoex_pkl="autoextend.pkl"
# autoex_loc = "c:/vectors/autoextend.word2vecformat.bin"
# lesk_loc = "d:/workspace/PyWordNetSimilarity/src/lesk-relation.dat"

# lda_loc="/home/andrew/vectors/lda_prep_no_lemma/no_lemma.101.lda"
# wordids_loc="/home/andrew/vectors/lda_prep_no_lemma/lda_no_lemma_wordids.txt.bz2"
# tfidf_loc="/home/andrew/vectors/lda_prep_no_lemma/lda_no_lemma.tfidf_model"
# w2v_loc="/home/andrew/vectors/GoogleNews-vectors-negative300.bin"
# glove_loc="/home/andrew/vectors/glove.840B.300d.withheader.bin"
# # w2g_model_loc="/home/Andrew/vectors/wiki.biggervocab.w2g"
# # w2g_vocab_loc="/home/c/Users/Andrew/vectors/wiki.biggersize.gz"
# w2g_vocab_loc="/home/andrew/vectors/wiki.moreselective.gz"
# w2g_model_loc="/home/andrew/vectors/wiki.hyperparam.selectivevocab.w2g"
# autoex_loc = "/home/andrew/vectors/autoex.extracted.googleformat.bin"

betweenness_pkl="wordnet_betweenness.pkl"
load_pkl="wordnet_load.pkl"
# wordnetgraph_pkl="wordnet_graph.pkl"



usf = USFGraph("../Data/PairsFSG2.net")
usf_associations=usf.get_all_associations()
del usf


# #remove synset info
# evoc_associations_no_synset = []
# for evoc_assoc in evoc_associations:
#     l, r, score = evoc_assoc
#     evoc_associations_no_synset.append((l.split(".")[0],r.split(".")[0],score)) #take just the head noun
# evoc_associations = evoc_associations_no_synset

feature_extractor = AssociationFeatureExtractor(lda_model=get_wikipedia_lda(),
                                                w2v_model=get_google_word2vec(),
                                                autoex_model=get_google_autoextend(),
                                                betweenness_loc=betweenness_pkl,
                                                load_loc=load_pkl,
                                                glove_model=get_stanford_glove(),
                                                w2g_model=get_wikipedia_word2gauss(),
#                                                 lesk_relations=lesk_loc,
                                                verbose=True)
usf_feats = feature_extractor.get_features(usf_associations, "features/usf")
# del feature_extractor 
del usf_associations 
# # with open("usf_feats.pkl", "rb") as usf_feats_file:
# #     usf_feats = pickle.load(usf_feats_file )
# # print("usf loaded")
# # usf_feats = feature_extractor._add_w2g_feats(usf_feats)
# print("feat extracted")
# with open("usf_feats.pkl", "wb") as evoc_feats_file:
#     pickle.dump(usf_feats, evoc_feats_file )
# del usf_feats
# print("\nevoc done")
print("usf done")
  
eat = EATGraph("../Data/eat/pajek/EATnew2.net")
# del eat
eat_associations = eat.get_all_associations()
# # feature_extractor = FeatureExtractor(lda_loc=lda_loc,
# #                                      wordids_loc=wordids_loc,
# #                                      tfidf_loc=tfidf_loc,
# #                                      w2v_loc=w2v_loc,
# # #                                      autoex_loc=autoex_pkl,
# #                                      autoex_loc=autoex_loc,
# #                                      betweenness_loc=betweenness_pkl,
# #                                      load_loc=load_pkl,
# #                                      wordnetgraph_loc=wordnetgraph_pkl,
# #                                      glove_loc=glove_loc,
# #                                      w2g_model_loc=w2g_model_loc,
# #                                      w2g_vocab_loc=w2g_vocab_loc,
# #                                      dtype=float32)
eat_feats = feature_extractor.get_features(eat_associations,"features/eat")
# del feature_extractor
del eat_associations
print("eat done")
# with open("eat_feats_all3.pkl", "rb") as eat_feats_file:
#     eat_feats = pickle.load(eat_feats_file )
# # print("eat loaded")
# # eat_feats = feature_extractor._add_w2g_feats(eat_feats)
# print("feat extracted")
# with open("eat_feats.pkl", "wb") as eat_feats_file:
#     pickle.dump(eat_feats, eat_feats_file )
del eat_feats

evoc = EvocationDataset("../Data/evocation")
evoc_associations=evoc.get_all_associations()
del evoc
#convert from synsets to headwords
evoc_associations = [(stimuli.split(".")[0], response.split(".")[0], strength) for stimuli, response, strength in evoc_associations]
evoc_feats = feature_extractor.get_features(evoc_associations,"features/evoc")
del feature_extractor
del evoc_associations

print("\n\ndone\n")

