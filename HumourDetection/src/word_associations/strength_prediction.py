'''
Created on Jan 9, 2017

@author: Andrew Cattle <acattle@cse.ust.hk>
'''
import os

# #force cpu for parallelism
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['KERAS_BACKEND'] = 'theano'

from time import strftime
from scipy.stats.stats import spearmanr, pearsonr as sk_pearsonr #TODO: normal
from word_associations.association_feature_extractor import AssociationFeatureExtractor, DEFAULT_FEATS,\
    ALL_FEATS, FEAT_W2V_SIM, FEAT_W2V_OFFSET, FEAT_GLOVE_SIM, FEAT_GLOVE_OFFSET,\
    FEAT_W2G_SIM, FEAT_W2G_ENERGY, FEAT_W2G_OFFSET, FEAT_MAX_AUTOEX_SIM,\
    FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM,\
    FEAT_AVG_WUP_SIM, FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM, FEAT_MAX_PATH_SIM,\
    FEAT_AVG_PATH_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD,\
    FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS,\
    FEAT_DIR_REL, FEAT_LESK, FEAT_W2V_VECTORS, FEAT_GLOVE_VECTORS,\
    FEAT_W2G_VECTORS
import numpy as np
from sklearn.pipeline import Pipeline

def _create_mlp(num_units=None, input_dim=None, metrics=None,optimizer="adam", dropout=0.5,initializer='glorot_uniform'): #r2_score,pearsonr
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import Adam, SGD
    from util.keras_metrics import r2_score, pearsonr
    
    metrics=[r2_score,pearsonr]
    initializer="uniform"
#     optimizer = SGD(momentum=0.9, nesterov=True)
    if num_units == None:
        raise ValueError("num_units cannot be None. Please specify a value.")
    if input_dim == None:
        raise ValueError("input_dim cannot be None. Please specify a value.")
    model = Sequential()
    model.add(Dense(num_units, input_dim=input_dim, kernel_initializer=initializer))
    model.add(Dropout(dropout))
    model.add(Dense(num_units, input_dim=num_units, kernel_initializer=initializer))
    model.add(Dropout(dropout))
    model.add(Dense(1, input_dim=num_units, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics)
#     model.add(Dense(1, input_dim=num_units, activation="relu"))
#     model.compile(loss="mse", optimizer=optimizer, metrics=metrics)
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
    
    input_dim = 415 #TODO: remove
#     input_dim = assoc_feat_ext.get_num_dimensions()
    if num_units == None:
        #if num_units not specified, default to half of the input (minimum 5 units)
        num_units = max(int(input_dim/2), 5)

#     from sklearn.ensemble import RandomForestRegressor
#     estimator = RandomForestRegressor(n_estimators=100)
#     from sklearn.linear_model import LinearRegression
#     estimator = LinearRegression()
    from keras.wrappers.scikit_learn import KerasRegressor
    estimator = KerasRegressor(build_fn=_create_mlp, num_units=num_units, input_dim=input_dim, epochs=epochs, batch_size=batchsize, verbose=2)#TODO: verbose=verbose
    
#     return assoc_feat_ext, estimator #TODO: undo
    evoc_est_pipeline = Pipeline([("extract features", assoc_feat_ext),
                                  ("estimator", estimator)
                                  ])
    evoc_est_pipeline.fit(X,y) #train the feature extractor
    return evoc_est_pipeline
    

FEAT_W2V_SIM_SCALED = f"{FEAT_W2V_SIM} scaled"
FEAT_GLOVE_SIM_SCALED = f"{FEAT_GLOVE_SIM} scaled"
FEAT_W2G_SIM_SCALED = f"{FEAT_W2G_SIM} scaled"
FEAT_W2G_ENERGY_SCALED = f"{FEAT_W2G_ENERGY} scaled"
FEAT_MAX_AUTOEX_SIM_SCALED = f"{FEAT_MAX_AUTOEX_SIM} scaled"
FEAT_AVG_AUTOEX_SIM_SCALED = f"{FEAT_AVG_AUTOEX_SIM} scaled"
FEAT_LDA_SIM_SCALED = f"{FEAT_LDA_SIM} scaled"
FEAT_LEXVECTORS_SCALED = f"{FEAT_LEXVECTORS} scaled"
FEAT_MAX_WUP_SIM_SCALED = f"{FEAT_MAX_WUP_SIM} scaled"
FEAT_AVG_WUP_SIM_SCALED = f"{FEAT_AVG_WUP_SIM} scaled"
FEAT_MAX_LCH_SIM_SCALED = f"{FEAT_MAX_LCH_SIM} scaled"
FEAT_AVG_LCH_SIM_SCALED = f"{FEAT_AVG_LCH_SIM} scaled"
FEAT_MAX_PATH_SIM_SCALED = f"{FEAT_MAX_PATH_SIM} scaled"
FEAT_AVG_PATH_SIM_SCALED = f"{FEAT_AVG_PATH_SIM} scaled"
FEAT_MAX_LOAD_SCALED = f"{FEAT_MAX_LOAD} scaled"
FEAT_AVG_LOAD_SCALED = f"{FEAT_AVG_LOAD} scaled"
FEAT_TOTAL_LOAD_SCALED = f"{FEAT_TOTAL_LOAD} scaled"
FEAT_MAX_BETWEENNESS_SCALED = f"{FEAT_MAX_BETWEENNESS} scaled"
FEAT_AVG_BETWEENNESS_SCALED = f"{FEAT_AVG_BETWEENNESS} scaled"
FEAT_TOTAL_BETWEENNESS_SCALED = f"{FEAT_TOTAL_BETWEENNESS} scaled"
FEAT_DIR_REL_SCALED = f"{FEAT_DIR_REL} scaled"
FEAT_LESK_SCALED = f"{FEAT_LESK} scaled"

FEAT_TO_SCALED_FEAT = {FEAT_W2V_SIM : FEAT_W2V_SIM_SCALED,
                       FEAT_GLOVE_SIM : FEAT_GLOVE_SIM_SCALED,
                       FEAT_W2G_SIM : FEAT_W2G_SIM_SCALED,
                       FEAT_W2G_ENERGY : FEAT_W2G_ENERGY_SCALED,
                       FEAT_MAX_AUTOEX_SIM : FEAT_MAX_AUTOEX_SIM_SCALED,
                       FEAT_AVG_AUTOEX_SIM : FEAT_AVG_AUTOEX_SIM_SCALED,
                       FEAT_LDA_SIM : FEAT_LDA_SIM_SCALED,
                       FEAT_LEXVECTORS : FEAT_LEXVECTORS_SCALED,
                       FEAT_MAX_WUP_SIM : FEAT_MAX_WUP_SIM_SCALED,
                       FEAT_AVG_WUP_SIM : FEAT_AVG_WUP_SIM_SCALED,
                       FEAT_MAX_LCH_SIM : FEAT_MAX_LCH_SIM_SCALED,
                       FEAT_AVG_LCH_SIM : FEAT_AVG_LCH_SIM_SCALED,
                       FEAT_MAX_PATH_SIM : FEAT_MAX_PATH_SIM_SCALED,
                       FEAT_AVG_PATH_SIM : FEAT_AVG_PATH_SIM_SCALED,
                       FEAT_MAX_LOAD : FEAT_MAX_LOAD_SCALED,
                       FEAT_AVG_LOAD : FEAT_AVG_LOAD_SCALED,
                       FEAT_TOTAL_LOAD : FEAT_TOTAL_LOAD_SCALED,
                       FEAT_MAX_BETWEENNESS : FEAT_MAX_BETWEENNESS_SCALED,
                       FEAT_AVG_BETWEENNESS : FEAT_AVG_BETWEENNESS_SCALED,
                       FEAT_TOTAL_BETWEENNESS : FEAT_TOTAL_BETWEENNESS_SCALED,
                       FEAT_DIR_REL : FEAT_DIR_REL_SCALED,
                       FEAT_LESK : FEAT_LESK_SCALED
                       }

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
    from sklearn.model_selection import cross_val_predict
    import csv
    
    from keras.wrappers.scikit_learn import KerasRegressor

    print(f"{strftime('%y-%m-%d_%H:%M:%S')}\tStarting test")

    epochs=100
    batchsize=250
    lesk_loc = "d:/git/PyWordNetSimilarity/PyWordNetSimilarity/src/lesk-relation.dat"
    betweenness_pkl="d:/git/HumourDetection/HumourDetection/src/word_associations/wordnet_betweenness.pkl"
    load_pkl="d:/git/HumourDetection/HumourDetection/src/word_associations/wordnet_load.pkl"
    
    name, data = dataset
#     random.seed(10)
#     random.shuffle(data)
#     word_pairs, strengths = zip(*data)
#          
#     strengths = np.array(strengths)
#      
# #     with open(f"{name}/word pairs.pkl", "rb") as wp_f:
# #         word_pairs=pickle.load(wp_f)
#         
#     assoc_feat_ext = AssociationFeatureExtractor(#features=ALL_FEATS,
#                                                  features=list(FEAT_TO_SCALED_FEAT.keys()),
#                                                  lda_model_getter=get_wikipedia_lda,
#                                                  w2v_model_getter=get_google_word2vec,
#                                                  autoex_model_getter=get_google_autoextend,
#                                                  betweenness_loc=betweenness_pkl,
#                                                  load_loc=load_pkl,
#                                                  glove_model_getter=get_stanford_glove,
#                                                  w2g_model_getter=get_wikipedia_word2gauss,
#                                                  lesk_relations=lesk_loc,
#                                                  verbose=2,
#                                                  low_memory=True)
#     feature_dict = assoc_feat_ext.extract_features(word_pairs)
#     
#     if not os.path.exists(f"{name}"):
#         os.mkdir(f"{name}")
#     with open(f"{name}/feature names.pkl", "wb") as fn_f:
#         pickle.dump(list(feature_dict.keys()), fn_f)
#     for feat, mat in feature_dict.items():
#         np.save(f"{name}/{feat}.npy", mat)
#     np.save(f"{name}/strengths.npy", strengths)
#     with open(f"{name}/word pairs.pkl", "wb") as wp_f:
#         pickle.dump(word_pairs,wp_f)
        
# #     feature_dict = np.load(f"{name} feature dict.npy")[()]
#     with open(f"{name}/feature names.pkl", "rb") as fn_f:
#         feature_names = pickle.load(fn_f)
#     feature_dict = {}
#     for feat in feature_names:
#         feature_dict[feat] = np.load(f"{name}/{feat}.npy")
    strengths = np.load(f"{name}/strengths.npy")
#     with open(f"{name}/word pairs.pkl", "rb") as wp_f:
#         word_pairs=pickle.load(wp_f)
    
#     #scale
#     from sklearn.preprocessing import StandardScaler
#     for feat, scaled_feat in FEAT_TO_SCALED_FEAT.items():
#         feat_mat = np.load(f"{name}/{feat}.npy")
#         s = StandardScaler()
#         np.save(f"{name}/{scaled_feat}.npy", s.fit_transform(feat_mat))
        
    
    #individual feature performance
    configs = [(f"{feat} only ", [feat]) for feat in ALL_FEATS]
    configs.extend([("all autoex only", [FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM]),
                    ("all betweenness only", [FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS]),
                    ("all load only", [FEAT_MAX_LOAD, FEAT_AVG_LOAD]),
                    ("all wup only", [FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM]),
                    ("all path only", [FEAT_MAX_PATH_SIM, FEAT_AVG_PATH_SIM]),
                    ("all lch only", [FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM]),
                    ])
#     #scaled individual features
#     configs.extend([(f"{feat} only", [feat]) for feat in FEAT_TO_SCALED_FEAT.values()])
#     configs.extend([("all scaled autoex only", [FEAT_MAX_AUTOEX_SIM_SCALED, FEAT_AVG_AUTOEX_SIM_SCALED]),
#                     ("all scaled betweenness only", [FEAT_MAX_BETWEENNESS_SCALED, FEAT_AVG_BETWEENNESS_SCALED]),
#                     ("all scaled load only", [FEAT_MAX_LOAD_SCALED, FEAT_AVG_LOAD_SCALED]),
#                     ("all scaled wup only", [FEAT_MAX_WUP_SIM_SCALED, FEAT_AVG_WUP_SIM_SCALED]),
#                     ("all scaled path only", [FEAT_MAX_PATH_SIM_SCALED, FEAT_AVG_PATH_SIM_SCALED]),
#                     ("all scaled lch only", [FEAT_MAX_LCH_SIM_SCALED, FEAT_AVG_LCH_SIM_SCALED]),
#                     ])
    #full feature suites
    configs.extend([#("all feats, all embeddings", ALL_FEATS),
#                     ("all features, all embeddings (offset only)", [FEAT_W2V_SIM, FEAT_W2V_OFFSET, FEAT_GLOVE_SIM, FEAT_GLOVE_OFFSET, FEAT_W2G_SIM, FEAT_W2G_ENERGY, FEAT_W2G_OFFSET, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM, FEAT_MAX_PATH_SIM, FEAT_AVG_PATH_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS, FEAT_DIR_REL, FEAT_LESK]),
#                     ("all features, all embeddings (vectors only)", [FEAT_W2V_SIM, FEAT_W2V_VECTORS, FEAT_GLOVE_SIM, FEAT_GLOVE_VECTORS, FEAT_W2G_SIM, FEAT_W2G_ENERGY, FEAT_W2G_VECTORS, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM, FEAT_MAX_PATH_SIM, FEAT_AVG_PATH_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS, FEAT_DIR_REL, FEAT_LESK]),
                    ("all features (w2v only, offset only)", [FEAT_W2V_SIM, FEAT_W2V_OFFSET, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM, FEAT_MAX_PATH_SIM, FEAT_AVG_PATH_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS, FEAT_DIR_REL, FEAT_LESK]),
                    ("all features (w2v only, vectors only)", [FEAT_W2V_SIM, FEAT_W2V_VECTORS, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM, FEAT_MAX_PATH_SIM, FEAT_AVG_PATH_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS, FEAT_DIR_REL, FEAT_LESK]),
                    ("all features (glove only, offset only)", [FEAT_GLOVE_SIM, FEAT_GLOVE_OFFSET, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM, FEAT_MAX_PATH_SIM, FEAT_AVG_PATH_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS, FEAT_DIR_REL, FEAT_LESK]),
                    ("all features (glove only, vectors only)", [FEAT_GLOVE_SIM, FEAT_GLOVE_VECTORS, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM, FEAT_MAX_PATH_SIM, FEAT_AVG_PATH_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS, FEAT_DIR_REL, FEAT_LESK]),
                    ("all features (w2g only, offset only)", [FEAT_W2G_SIM, FEAT_W2G_ENERGY, FEAT_W2G_OFFSET, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM, FEAT_MAX_PATH_SIM, FEAT_AVG_PATH_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS, FEAT_DIR_REL, FEAT_LESK]),
                    ("all features (w2g only, vectors only)", [FEAT_W2G_SIM, FEAT_W2G_ENERGY, FEAT_W2G_VECTORS, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM, FEAT_MAX_PATH_SIM, FEAT_AVG_PATH_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS, FEAT_DIR_REL, FEAT_LESK]),
                    ("all features, all sims (w2v offset only)", [FEAT_W2V_SIM, FEAT_GLOVE_SIM, FEAT_W2G_ENERGY, FEAT_W2G_SIM, FEAT_W2V_OFFSET, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM, FEAT_MAX_PATH_SIM, FEAT_AVG_PATH_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS, FEAT_DIR_REL, FEAT_LESK]),
                    ("all features, all sims (w2v vectors only)", [FEAT_W2V_SIM, FEAT_GLOVE_SIM, FEAT_W2G_ENERGY, FEAT_W2G_SIM, FEAT_W2V_VECTORS, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM, FEAT_MAX_PATH_SIM, FEAT_AVG_PATH_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS, FEAT_DIR_REL, FEAT_LESK]),
                    ("all features, all sims (glove offset only)", [FEAT_W2V_SIM, FEAT_GLOVE_SIM, FEAT_W2G_ENERGY, FEAT_W2G_SIM, FEAT_GLOVE_OFFSET, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM, FEAT_MAX_PATH_SIM, FEAT_AVG_PATH_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS, FEAT_DIR_REL, FEAT_LESK]),
                    ("all features, all sims (glove vectors only)", [FEAT_W2V_SIM, FEAT_GLOVE_SIM, FEAT_W2G_ENERGY, FEAT_W2G_SIM, FEAT_GLOVE_VECTORS, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM, FEAT_MAX_PATH_SIM, FEAT_AVG_PATH_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS, FEAT_DIR_REL, FEAT_LESK]),
                    ("all features, all sims (w2g offset only)", [FEAT_W2V_SIM, FEAT_GLOVE_SIM, FEAT_W2G_ENERGY, FEAT_W2G_SIM, FEAT_W2G_OFFSET, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM, FEAT_MAX_PATH_SIM, FEAT_AVG_PATH_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS, FEAT_DIR_REL, FEAT_LESK]),
                    ("all features, all sims (w2g vectors only)", [FEAT_W2V_SIM, FEAT_GLOVE_SIM, FEAT_W2G_ENERGY, FEAT_W2G_SIM, FEAT_W2G_VECTORS, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LCH_SIM, FEAT_AVG_LCH_SIM, FEAT_MAX_PATH_SIM, FEAT_AVG_PATH_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_TOTAL_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_TOTAL_BETWEENNESS, FEAT_DIR_REL, FEAT_LESK]),
                    ("default (w2v offset)", [FEAT_W2V_SIM, FEAT_W2V_OFFSET, FEAT_GLOVE_SIM, FEAT_W2G_SIM, FEAT_W2G_ENERGY, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_DIR_REL]),
                    ("default (glove offset)", [FEAT_W2V_SIM, FEAT_GLOVE_OFFSET, FEAT_GLOVE_SIM, FEAT_W2G_SIM, FEAT_W2G_ENERGY, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_DIR_REL]),
                    ("default (w2g offset)", [FEAT_W2V_SIM, FEAT_W2G_OFFSET, FEAT_GLOVE_SIM, FEAT_W2G_SIM, FEAT_W2G_ENERGY, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_DIR_REL]),
                    ("default (word2vec only)", DEFAULT_FEATS),
                    ("default (glove only)", [FEAT_GLOVE_SIM, FEAT_GLOVE_OFFSET, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_DIR_REL]),
                    ("default (w2g only)", [FEAT_W2G_SIM, FEAT_W2G_ENERGY, FEAT_W2G_OFFSET, FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM, FEAT_LDA_SIM, FEAT_LEXVECTORS, FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM, FEAT_MAX_LOAD, FEAT_AVG_LOAD, FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS, FEAT_DIR_REL]),
                   ])
    #ablation
    configs.extend([(f"default (w2v only, offset only) minus {feat}", [f for f in DEFAULT_FEATS if f != feat]) for feat in DEFAULT_FEATS])
    configs.extend([("default( w2v only, offset only) minus all autoex", [f for f in DEFAULT_FEATS if f not in (FEAT_MAX_AUTOEX_SIM, FEAT_AVG_AUTOEX_SIM)]),
                    ("default( w2v only, offset only) minus all betweenness", [f for f in DEFAULT_FEATS if f not in (FEAT_MAX_BETWEENNESS, FEAT_AVG_BETWEENNESS)]),
                    ("default( w2v only, offset only) minus all load", [f for f in DEFAULT_FEATS if f not in (FEAT_MAX_LOAD, FEAT_AVG_LOAD)]),
                    ("default( w2v only, offset only) minus all wup", [f for f in DEFAULT_FEATS if f not in (FEAT_MAX_WUP_SIM, FEAT_AVG_WUP_SIM)]),
                    ])
   
    start_time = strftime('%y-%m-%d_%H%M%S')
    os.mkdir(f"{name}/{start_time}")
    with open(f"{name}/{start_time}/prediction results.csv", "w", newline="") as f:
        #newline arg prevents extra whitespace after each row
        #https://stackoverflow.com/questions/3348460/csv-file-written-with-python-has-blank-lines-between-each-row#3348664
        w = csv.writer(f)
        w.writerow(["", "","pearson r", "p", "","spearman rho", "p"])
           
        for experiment, feature_set in configs:
            print(f"{start_time}\tStarting {experiment}")
               
#             feature_matrix = np.hstack([feature_dict[feat] for feat in feature_set])
            feature_matrix = np.hstack([np.load(f"{name}/{feat}.npy") for feat in feature_set])
#             feature_matrix = np.hstack([np.load(f"{name}/{FEAT_TO_SCALED_FEAT.get(feat, feat)}.npy") for feat in feature_set])
               
            input_dim = feature_matrix.shape[1]
            num_units = max(int(input_dim/2), 5)
            estimator = KerasRegressor(build_fn=_create_mlp, num_units=num_units, input_dim=input_dim, epochs=epochs, batch_size=batchsize, verbose=2)#TODO: verbose=verbose
            pred_y = cross_val_predict(estimator, feature_matrix, strengths, cv=5, n_jobs=1)
#             test_size = int(0.5*feature_matrix.shape[0])
#             test_X, test_y = feature_matrix[:test_size], strengths[:test_size]
#             train_X, train_y = feature_matrix[test_size:], strengths[test_size:]
#             history = estimator.fit(train_X,train_y, validation_data=(test_X, test_y))
#             pred_y = estimator.predict(test_X)
   
#             history = estimator.fit(feature_matrix,strengths, validation_split=0.1)
#             import matplotlib.pyplot as plt
#             # summarize history for accuracy
#             plt.plot(history.history['r2_score'])
#             plt.plot(history.history['val_r2_score'])
#             plt.title('model r2')
#             plt.ylabel('r2')
#             plt.xlabel('epoch')
#             plt.legend(['train', 'test'], loc='upper left')
#             plt.show()
#             # summarize history for loss
#             plt.plot(history.history['loss'])
#             plt.plot(history.history['val_loss'])
#             plt.title('model loss')
#             plt.ylabel('loss')
#             plt.xlabel('epoch')
#             plt.legend(['train', 'test'], loc='upper left')
#             plt.show()
#             #plot pearson r
#             plt.plot(history.history['pearsonr'])
#             plt.plot(history.history['val_pearsonr'])
#             plt.title('model pearsonr')
#             plt.ylabel('pearsonr')
#             plt.xlabel('epoch')
#             plt.legend(['train', 'test'], loc='upper left')
#             plt.show()
               
            r = sk_pearsonr(strengths, pred_y)
            rho = spearmanr(strengths, pred_y)
               
               
               
            w.writerow([experiment, "", r[0], r[1], "", rho[0], rho[1]])
            print(f"{strftime('%y-%m-%d_%H:%M:%S')}\tFinished {experiment}")
               
            np.save(f"{name}/{start_time}/{experiment} predictions.npy",pred_y)
                
    print(f"{strftime('%y-%m-%d_%H:%M:%S')}\tTest finished")
    
if __name__ == '__main__':
    #import model wrappers to keep them from being garbage collected
#     from util.model_wrappers.gensim_wrappers import gensim_vector_models
#     from util.model_wrappers import word2gauss_wrapper
    #TODO: are these even needed?
    
#     from util.dataset_readers..association_readers.xml_readers import EAT_XML_Reader, USF_XML_Reader, EvocationDataset, SWoW_Dataset, SWoW_Strengths_Dataset
#     
#     usf = USF_XML_Reader("../Data/usf/cue-target.xml").get_all_associations()
#     main(("usf", usf))
#     del usf
#       
#     eat = EAT_XML_Reader("../Data/eat/eat-stimulus-response.xml").get_all_associations()
#     main(("eat", eat))
#     del eat
#        
#     evoc = EvocationDataset("../Data/evocation").get_all_associations()
#     evoc = [((wp[0].split(".")[0], wp[1].split(".")[0]), stren) for wp, stren in evoc] #remove synset information
#     main(("evoc", evoc))
#     del evoc
#       
#     swow_stren = SWoW_Strengths_Dataset("D:/datasets/SWoW/strength.SWOW-EN.R123.csv").get_all_associations()
#     main(("swow stren", swow_stren))
#     del swow_stren
#      
#     swow_100 = SWoW_Dataset("D:/datasets/SWoW/SWOW-EN.R100.csv",complete=False).get_all_associations()
#     main(("swow 100", swow_100))
#     del swow_100 
#       
#     swow_all = SWoW_Dataset("D:/datasets/SWoW/SWOW-EN.complete.csv").get_all_associations()
#     main(("swow all", swow_all))
#     del swow_all

    import sys
    
    print(sys.argv)
    
    if len(sys.argv) < 2:
#         main(("usf", None))
        main(("eat", None))
        main(("evoc", None))
        main(("swow stren", None))
        main(("swow 100", None))
        main(("swow all", None))
    
    else:
        main((sys.argv[1], None))
    
#     for dataset in [("eat", eat), ("usf", usf), ("evoc", evoc), ("swow all", swow_all), ("swow 100", swow_100), ("swow stren", swow_stren)]:
#         print(len(dataset[1]))
#         main(dataset)
