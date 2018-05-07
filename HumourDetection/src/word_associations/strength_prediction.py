'''
Created on Jan 9, 2017

@author: Andrew Cattle <acattle@cse.ust.hk>
'''
from time import strftime
from scipy.stats.stats import spearmanr, pearsonr as sk_pearsonr #TODO: normal
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from word_associations.association_feature_extractor import AssociationFeatureExtractor, DEFAULT_FEATS
import numpy as np
from util.keras_metrics import r2_score, pearsonr
from sklearn.pipeline import Pipeline


def _create_mlp(num_units=None, input_dim=None):
def _create_mlp(num_units=None, input_dim=None, metrics=[],optimizer="adam", dropout=0.5,initializer='glorot_uniform'): #r2_score,pearsonr
    if num_units == None:
        raise ValueError("num_units cannot be None. Please specify a value.")
    if input_dim == None:
        raise ValueError("input_dim cannot be None. Please specify a value.")
    model = Sequential()
    model.add(Dense(num_units, input_dim=input_dim, kernel_initializer=initializer))
    model.add(Dropout(dropout))
    model.add(Dense(num_units, input_dim=num_units, kernel_initializer=initializer))
    model.add(Dropout(dropout))
    model.add(Dense(1, input_dim=num_units, activation="sigmoid")) #sigmoid
#     optimizer = Adam(lr=0.005, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics) #binary_crossentropy
    return model

def train_cattle_ma_2017_association_pipeline(X, y, num_units = None, epochs=50, batchsize=5000, features=DEFAULT_FEATS, lda_model_getter=None, w2v_model_getter=None, autoex_model_getter=None, betweenness_loc=None, load_loc=None,  glove_model_getter=None, w2g_model_getter=None, lesk_relations=None, verbose=False, low_memory=True):

    """
        Create an association prediction pipeline including feature extraction
        and neural network regressor as described in "Predicting Word
        Association Strengths" by Cattle and Ma (2017).
        
            Cattle, A., & Ma, X. (2017). Predicting Word Association Strengths.
            In Proceedings of the 2017 Conference on Empirical Methods in
            Natural Language Processing (pp. 1283-1288) 
            
        :param X: Association word pairs
        :type X: Iterable[Tuple[str, str]]
        :param y: Associations strengths
        :type y: Iterable[float]
        :param num_units: Number of units per layer in the neural net
        :type num_units: int
        :param epochs: Number of epochs to train neural network
        :type epochs: int
        :param batchsize: size of minibatch used for training
        :type batchsize: int
        :param features: list of association features to extract
        :type features: Iterable[str]
        :param lda_model_getter: function for retrieving LDA model to use for feature extraction
        :type lda_model_getter: Callable[[], GensimTopicSumModel]
        :param w2v_model_getter: function for retrieving Word2Vec model to use for feature extraction
        :type w2v_model_getter: Callable[[], GensimVectorModel]
        :param autoex_model_getter: function for retrieving AutoExtend model to use for feature extraction
        :type autoex_model_getter: Callable[[], GensimVectorModel]
        :param betweenness_loc: location of betweenness centrality pkl
        :type betweenness_loc: str
        :param load_loc: location of load centrality pkl
        :type load_loc: str
        :param glove_model_getter: function for retrieving GloVe model to use for feature extraction
        :type glove_model_getter: Callable[[], GensimVectorModel]
        :param w2g_model_getter: function for retrieving Word2Gauss model to use for feature extraction
        :type w2g_model_getter: Callable[[], Word2GaussModel]
        :param lesk_relations: Location of relations.dat for use with ExtendedLesk
        :type lesk_relations: str
        :param verbose: whether verbose mode should be used or not
        :type verbose: bool
        :param low_memory: specifies whether models should be purged from memory after use. This reduces memory usage but increases disk I/O as models will need to be automatically read back from disk before next use
        :type low_memory: bool
    """
    assoc_feat_ext = AssociationFeatureExtractor(features=features,
                                                lda_model_getter=lda_model_getter,
                                                w2v_model_getter=w2v_model_getter,
                                                autoex_model_getter=autoex_model_getter,
                                                betweenness_loc=betweenness_loc,
                                                load_loc=load_loc,
                                                glove_model_getter=glove_model_getter,
                                                w2g_model_getter=w2g_model_getter,
                                                lesk_relations=lesk_relations,
                                                verbose=verbose,
                                                low_memory=low_memory)
       
    input_dim = assoc_feat_ext.get_num_dimensions()
    if num_units == None:
        #if num_units not specified, default to half of the input (minimum 5 units)
        num_units = max(int(input_dim/2), 5)
    estimator = KerasRegressor(build_fn=_create_mlp, num_units=num_units, input_dim=input_dim, epochs=epochs, batch_size=batchsize, verbose=verbose)
    
    evoc_est_pipeline = Pipeline([("extract features", assoc_feat_ext),
                                  ("estimator", estimator)
                                  ])
    evoc_est_pipeline.fit(X,y) #train the feature extractor
    return evoc_est_pipeline
    

def main(dataset):
    """
    :param dataset: (<name>, <word_pairs>, <strengths>)
    :type dataset: Tuple[str, Iterable[Tuple[str, str], float]]
    """
    import pickle
    from util.keras_pipeline_persistance import save_keras_pipeline,\
        load_keras_pipeline
    from util.model_wrappers.common_models import get_wikipedia_lda, get_google_word2vec,\
    get_stanford_glove, get_wikipedia_word2gauss, get_google_autoextend
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description='Association Strength Prediction')
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

    print("{}\tStarting test".format(strftime("%y-%m-%d_%H:%M:%S")))
    
    lesk_loc = "d:/git/PyWordNetSimilarity/PyWordNetSimilarity/src/lesk-relation.dat"
#     lesk_loc = "/mnt/d/git/PyWordNetSimilarity/PyWordNetSimilarity/src/lesk-relation.dat"
    betweenness_pkl="d:/git/HumourDetection/HumourDetection/src/word_associations/wordnet_betweenness.pkl"
    load_pkl="d:/git/HumourDetection/HumourDetection/src/word_associations/wordnet_load.pkl"
    
    name, data = dataset
    random.seed(10)
    random.shuffle(data)
    word_pairs, strengths = zip(*data)
        
    strengths = np.array(strengths)
#     targets = targets*100 #seems to help
    test_size = int(strengths.shape[0] * 0.2) #hold out 20% for testing
         
    test_X = word_pairs[:test_size]
    train_X= word_pairs[test_size:]
          
    test_y = strengths[:test_size]
    train_y = strengths[test_size:]
    
    model = train_cattle_ma_2017_association_pipeline(train_X, train_y, epochs=args.epoch, batchsize=args.batchsize,
                                                      lda_model_getter=get_wikipedia_lda,
                                                      w2v_model_getter=get_google_word2vec,
                                                      autoex_model_getter=get_google_autoextend,
                                                      betweenness_loc=betweenness_pkl,
                                                      load_loc=load_pkl,
                                                      glove_model_getter=get_stanford_glove,
                                                      w2g_model_getter=get_wikipedia_word2gauss,
                                                      lesk_relations=lesk_loc,
                                                      verbose=True)
#     model = load_keras_pipeline("models/{}".format(dataset_to_test))
    
    
    print("{}\tSaving model".format(strftime("%y-%m-%d_%H:%M:%S")))
#     from util.gensim_wrappers.gensim_vector_models import purge_all_gensim_vector_models
#     from util.word2gauss_wrapper import purge_all_word2gauss_vector_models
#     from util.gensim_wrappers.gensim_topicsum_models import purge_all_gensim_topicsum_models
#     purge_all_gensim_vector_models()
#     purge_all_word2gauss_vector_models()
#     purge_all_gensim_topicsum_models()
     
    save_keras_pipeline("models/{}".format(dataset_to_test), model)
    
    model.fit(train_X, train_y, validation_data=(test_X, test_y))
      
    print("{}\tTraining finished".format(strftime("%y-%m-%d_%H:%M:%S")))
     
    print("{}\tSaving model".format(strftime("%y-%m-%d_%H:%M:%S"))) 
    save_keras_pipeline("models/{}".format(name), p)
    print("{}\tModel saved".format(strftime("%y-%m-%d_%H:%M:%S")))
    
#     model = load_keras_pipeline("models/{}-all".format(name))
         
    print("{}\tStarting test".format(strftime("%y-%m-%d_%H:%M:%S")))
    test_pred=model.predict(test_X)
    
#         test_y = test_y.reshape((-1,1))
    print(f"{name} pred: {test_pred}")
    print(f"{name} spearman {spearmanr(test_y, test_pred)}")
#         test_pred=test_pred.flatten()
#     test_y = test_y.flatten()
    print(f"{name} pearson {sk_pearsonr(test_y, test_pred)}") #TODO: sk_
       
    print("{}\tTest finished".format(strftime("%y-%m-%d_%H:%M:%S")))
    
if __name__ == '__main__':
    #import model wrappers to keep them from being garbage collected
    from util.model_wrappers.gensim_wrappers import gensim_vector_models
    from util.model_wrappers import word2gauss_wrapper
    #TODO: are these even needed?
    
    from word_associations.association_readers.xml_readers import EAT_XML_Reader, USF_XML_Reader, EvocationDataset, SWoW_Dataset
    eat = EAT_XML_Reader("../Data/eat/eat-stimulus-response.xml").get_all_associations()
    usf = USF_XML_Reader("../Data/usf/cue-target.xml").get_all_associations()
    evoc = EvocationDataset("../Data/evocation").get_all_associations()
    evoc = [((wp[0].split(".")[0], wp[1].split(".")[0]), stren) for wp, stren in evoc] #remove synset information
    swow = SWoW_Dataset("D:/datasets/SWoW/SWOW-EN.complete.csv").get_all_associations()
     
    for dataset in [("swow", swow)]:#("eat", eat), ("usf", usf), ("evoc", evoc), ("swow", swow)]:
        main(dataset)
