'''
Created on Jan 9, 2017

@author: Andrew
'''
import argparse
from time import strftime
from scipy.stats.stats import spearmanr, pearsonr
from keras.models import Sequential
from keras.layers import Dense, Dropout
from os.path import join
from keras.wrappers.scikit_learn import KerasRegressor
from evocation_feature_extractor import EvocationFeatureExtractor, DEFAULT_FEATS
from sklearn.pipeline import Pipeline
import pickle
from util.keras_pipeline_persistance import save_keras_pipeline,\
    load_keras_pipeline

def _create_mlp(num_units=None, input_dim=None):
    if num_units == None:
        raise ValueError("num_units cannot be None. Please specify a value.")
    if input_dim == None:
        raise ValueError("input_dim cannot be None. Please specify a value.")
    model = Sequential()
    model.add(Dense(num_units, input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(num_units, input_dim=num_units))
    model.add(Dropout(0.5))
    model.add(Dense(1, input_dim=num_units, activation="relu"))
    model.compile(loss="mse", optimizer="adam")
    return model

def train_evocation_estimation_pipeline(X, y, num_units = None, epochs=50, batchsize=5000, features=DEFAULT_FEATS, lda_loc=None, wordids_loc=None, tfidf_loc=None, w2v_loc=None, autoex_loc=None, betweenness_loc=None, load_loc=None,  glove_loc=None, w2g_model_loc=None, w2g_vocab_loc=None, lesk_relations=None, verbose=0):
    evoc_feat_ext = EvocationFeatureExtractor(features=features,
                                              lda_loc=lda_loc,
                                              wordids_loc=wordids_loc,
                                              tfidf_loc=tfidf_loc,
                                              w2v_loc=w2v_loc,
                                              autoex_loc=autoex_loc,
                                              betweenness_loc=betweenness_loc,
                                              load_loc=load_loc,
                                              glove_loc=glove_loc,
                                              w2g_model_loc=w2g_model_loc,
                                              w2g_vocab_loc=w2g_vocab_loc,
                                              lesk_relations=lesk_relations,
                                              verbose=verbose
                                              )
       
    input_dim = evoc_feat_ext.get_num_dimensions()
    if num_units == None:
        #if num_units not specified, default to half of the input (minimum 5 units)
        num_units = max(int(input_dim/2), 5)
    estimator = KerasRegressor(build_fn=_create_mlp, num_units=num_units, input_dim=input_dim, epochs=epochs, batch_size=batchsize, verbose=verbose)
    
    evoc_est_pipeline = Pipeline([("extract features", evoc_feat_ext),
                                  ("estimator", estimator)
                                  ])
    evoc_est_pipeline.fit(X,y) #train the feature extractor
    return evoc_est_pipeline
    

def main(dataset_to_test):
    parser = argparse.ArgumentParser(description='Evocation Estimation')
    parser.add_argument('--batchsize', '-b', type=int, default=5000,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=204, #our data has 409 features, 204 is half the number of features
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    
    pickle_dir = "features/{}".format(dataset_to_test)

    print("{}\tStarting test".format(strftime("%y-%m-%d_%H:%M:%S")))

    lda_loc="c:/vectors/lda_prep_no_lemma/no_lemma.101.lda"
    wordids_loc="c:/vectors/lda_prep_no_lemma/lda_no_lemma_wordids.txt.bz2"
    tfidf_loc="c:/vectors/lda_prep_no_lemma/lda_no_lemma.tfidf_model"
    w2v_loc="c:/vectors/GoogleNews-vectors-negative300.bin"
    glove_loc="c:/vectors/glove.840B.300d.withheader.bin"
    # w2g_model_loc="c:/vectors/wiki.biggervocab.w2g"
    # w2g_vocab_loc="c:/vectors/wiki.biggersize.gz"
    w2g_vocab_loc="c:/vectors/wiki.moreselective.gz"
    w2g_model_loc="c:/vectors/wiki.hyperparam.selectivevocab.w2g"
    autoex_loc = "c:/vectors/autoextend.word2vecformat.bin"
    lesk_loc = "d:/git/PyWordNetSimilarity/PyWordNetSimilarity/src/lesk-relation.dat"
     
#         lda_loc="/mnt/c/vectors/lda_prep_no_lemma/no_lemma.101.lda"
#         wordids_loc="/mnt/c/vectors/lda_prep_no_lemma/lda_no_lemma_wordids.txt.bz2"
#         tfidf_loc="/mnt/c/vectors/lda_prep_no_lemma/lda_no_lemma.tfidf_model"
#         w2v_loc="/mnt/c/vectors/GoogleNews-vectors-negative300.bin"
#         glove_loc="/mnt/c/vectors/glove.840B.300d.withheader.bin"
#         # w2g_model_loc="/mnt/c/vectors/wiki.biggervocab.w2g"
#         # w2g_vocab_loc="/mnt/c/vectors/wiki.biggersize.gz"
#         w2g_vocab_loc="/mnt/c/vectors/wiki.moreselective.gz"
#         w2g_model_loc="/mnt/c/vectors/wiki.hyperparam.selectivevocab.w2g"
#         autoex_loc = "/mnt/c/vectors/autoextend.word2vecformat.bin"
#         lesk_loc = "/mnt/d/git/PyWordNetSimilarity/PyWordNetSimilarity/src/lesk-relation.dat"
    
    betweenness_pkl="wordnet_betweenness.pkl"
    load_pkl="wordnet_load.pkl"
    
#         evoc_feat_ext = EvocationFeatureExtractor(lda_loc=lda_loc,
#                                                   wordids_loc=wordids_loc,
#                                                   tfidf_loc=tfidf_loc,
#                                                   w2v_loc=w2v_loc,
#                                                   autoex_loc=autoex_loc,
#                                                   betweenness_loc=betweenness_pkl,
#                                                   load_loc=load_pkl,
#                                                   glove_loc=glove_loc,
#                                                   w2g_model_loc=w2g_model_loc,
#                                                   w2g_vocab_loc=w2g_vocab_loc,
#                                                   lesk_relations=lesk_loc)
    
    with open(join(pickle_dir, "word_pairs.pkl"), "rb") as words_file:
        stimuli_response = pickle.load(words_file, encoding="latin1")
    with open(join(pickle_dir, "strengths.pkl"), "rb") as strength_file:
        targets = pickle.load(strength_file, encoding="latin1")#https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    targets = targets*100 #seems to help
    test_size = int(targets.shape[0] * 0.2) #hold out 20% for testing
    
    test_X = stimuli_response[:test_size]
    train_X= stimuli_response[test_size:]
    
    test_y = targets[:test_size]
    train_y = targets[test_size:]

    model = train_evocation_estimation_pipeline(train_X, train_y, epochs=args.epoch, batchsize=args.batchsize,
                                                lda_loc=lda_loc,
                                                wordids_loc=wordids_loc,
                                                tfidf_loc=tfidf_loc,
                                                w2v_loc=w2v_loc,
                                                autoex_loc=autoex_loc,
                                                betweenness_loc=betweenness_pkl,
                                                load_loc=load_pkl,
                                                glove_loc=glove_loc,
                                                w2g_model_loc=w2g_model_loc,
                                                w2g_vocab_loc=w2g_vocab_loc,
                                                lesk_relations=lesk_loc)
#     model = load_keras_pipeline("models/{}".format(dataset_to_test))
    
    print("{}\tTraining finished".format(strftime("%y-%m-%d_%H:%M:%S")))
    
    print("{}\tSaving model".format(strftime("%y-%m-%d_%H:%M:%S")))
#     from util.gensim_wrappers.gensim_vector_models import purge_all_gensim_vector_models
#     from util.word2gauss_wrapper import purge_all_word2gauss_vector_models
#     from util.gensim_wrappers.gensim_topicsum_models import purge_all_gensim_topicsum_models
#     purge_all_gensim_vector_models()
#     purge_all_word2gauss_vector_models()
#     purge_all_gensim_topicsum_models()
    
    save_keras_pipeline("models/{}".format(dataset_to_test), model)
    
    print("{}\tModel saved".format(strftime("%y-%m-%d_%H:%M:%S")))
    
    print("{}\tStarting test".format(strftime("%y-%m-%d_%H:%M:%S")))
    test_pred=model.predict(test_X)
#         test_y = test_y.reshape((-1,1))
    print()
    print(spearmanr(test_y, test_pred))
#         test_pred=test_pred.flatten()
    test_y = test_y.flatten()
    print(pearsonr(test_y,test_pred))
    print("{}\tTest finished".format(strftime("%y-%m-%d_%H:%M:%S")))
    
if __name__ == '__main__':
    #import model wrappers to keep them from being garbage collected
    from util.gensim_wrappers import gensim_vector_models
    from util import word2gauss_wrapper
    for d in ["evoc", "usf", "eat"]:
        main(d)
