'''
Created on Jan 9, 2017

@author: Andrew
'''
from evocation_reader import EATGraph, USFGraph
from evocation_feature_extractor import FeatureExtractor
from numpy import array, vstack
import pickle
from time import strftime
#needed for unpickling to work
from autoextend import AutoExtendEmbeddings
from wordnet_graph import WordNetGraph
from keras.models import Sequential
from keras.layers.core import Dense

from functools import partial
from multiprocessing import Pool
def get_feature_vectors(self, associations, feature_extractor):
    data = []
    target = []
    for stimuli, response, strength in associations:
        data.append(feature_extractor.get_feature_vector(stimuli, response))
        target.append(strength)
    data = array(data)
    target = array(target)
    
    return data, target

if __name__ == "__main__":
    # Load EAT data
    print("{}\tloading EAT".format(strftime("%y-%m-%d_%H:%M:%S")))
    eat = EATGraph("../Data/eat/pajek/EATnew.net")
#     usf = USFGraph("../Data/PairsFSG.net")
    associations = eat.get_all_associations()
    print("{}\tEAT loaded".format(strftime("%y-%m-%d_%H:%M:%S")))

    #load feature extractors
    print("{}\tLoading feature extractor".format(strftime("%y-%m-%d_%H:%M:%S")))
    lda_loc="/mnt/c/Users/Andrew/Desktop/vectors/lda_prep_no_lemma/no_lemma.lda"
    wordids_loc="/mnt/c/Users/Andrew/Desktop/vectors/lda_prep_no_lemma/lda_no_lemma_wordids.txt.bz2"
    tfidf_loc="/mnt/c/Users/Andrew/Desktop/vectors/lda_prep_no_lemma/lda_no_lemma.tfidf_model"
    w2v_loc="/mnt/c/Users/Andrew/Desktop/vectors/GoogleNews-vectors-negative300.bin"
    autoex_pkl="autoextend.pkl"
    betweenness_pkl="wordnet_betweenness.pkl"
    load_pkl="wordnet_load.pkl"
    wordnetgraph_pkl="wordnet_graph.pkl"
    feature_extractor = FeatureExtractor(lda_loc, wordids_loc, tfidf_loc, w2v_loc, autoex_pkl, betweenness_pkl, load_pkl, wordnetgraph_pkl)
    print("{}\tFeature extractor loaded".format(strftime("%y-%m-%d_%H:%M:%S")))
      
    print("{}\tExtracting features".format(strftime("%y-%m-%d_%H:%M:%S")))
    data = []
    target = []
    for stimuli, response, strength in associations:
        data.append(feature_extractor.get_feature_vector(stimuli, response))
        target.append(strength)
    data = array(data)
    target = array(target)
#     lda_loc="/home/acattle/lda/no_lemma.lda"
#     wordids_loc="/home/acattle/lda/lda_no_lemma_wordids.txt.bz2"
#     tfidf_loc="/home/acattle/lda/lda_no_lemma.tfidf_model"
#     w2v_loc="/home/acattle/vectors/word2vec_main_GoogleNews-vectors-negative300.bin"
#     autoex_pkl="autoextend.pkl"
#     betweenness_pkl="wordnet_betweenness.pkl"
#     load_pkl="wordnet_load.pkl"
#     wordnetgraph_pkl="wordnet_graph.pkl"
#     feature_extractor = FeatureExtractor(lda_loc, wordids_loc, tfidf_loc, w2v_loc, autoex_pkl, betweenness_pkl, load_pkl, wordnetgraph_pkl)
#     print("{}\tFeature extractor loaded".format(strftime("%y-%m-%d_%H:%M:%S")))
#     
#     print("{}\tExtracting features".format(strftime("%y-%m-%d_%H:%M:%S")))
#     
#     partial_extractor = partial(get_feature_vectors, feature_extractor=feature_extractor)
#     association_chunks = [[associations[i] for i in xrange(len(associations)) if (i % 3) == r] for r in range(3)]
#     p=Pool(3)
#     data = []
#     target = []
#     for data_p, target_p in p.map(partial_extractor, association_chunks):
#         data.append(data_p)
#         target.append(target_p)
#     p.close()
#     
#     data = vstack(data)
#     target = vstack(target)
#     print("{}\tFeatures extracted".format(strftime("%y-%m-%d_%H:%M:%S")))
     
    with open("eat_feature_matrix.pkl", "wb") as data_pkl:
        pickle.dump(data, data_pkl)
    with open("eat_target_vector.pkl", "wb") as target_pkl:
        pickle.dump(target, target_pkl)
#     with open("usf_feature_matrix.pkl", "rb") as data_pkl:
#         data = pickle.load(data_pkl)
#     with open("usf_target_vector.pkl", "rb") as target_pkl:
#         target = pickle.load(target_pkl)
    
    # create model
    input_dim = data.shape[1]
    print input_dim
    num_dim = 1000
    model = Sequential()
    model.add(Dense(num_dim, input_dim=input_dim, init='uniform', activation='relu'))
    model.add(Dense(num_dim, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='linear'))
    
    print("{}\tStarting training".format(strftime("%y-%m-%d_%H:%M:%S")))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(data, target, nb_epoch=20, batch_size=100)
    print("{}\tTraining finished".format(strftime("%y-%m-%d_%H:%M:%S")))
    
    scores = model.evaluate(data, target)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    model.save("eat_keras.h5")
    
    