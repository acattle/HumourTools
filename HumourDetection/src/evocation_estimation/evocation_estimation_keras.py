'''
Created on Jan 9, 2017

@author: Andrew
'''
import argparse


# from evocation_reader import EATGraph, USFGraph, EvocationDataset
# from evocation_feature_extractor import FeatureExtractor
from numpy import array, float32, vstack,hstack, nan_to_num#, float64
from time import strftime
from scipy.stats.stats import spearmanr, pearsonr
import pickle
# import warnings
# warnings.filterwarnings("error")
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow import Session
#needed for unpickling to work
# from autoextend import AutoExtendEmbeddings
# from wordnet_graph import WordNetGraph
# from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
import random
from math import ceil
from os.path import join, exists
from numpy import inf, empty
# from functools import partial
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.feature_selection.rfe import RFECV
from sklearn.model_selection import StratifiedKFold #NOT from cross_validation
from sklearn.feature_selection.univariate_selection import SelectKBest
from sklearn.feature_selection.mutual_info_ import mutual_info_regression


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
    
    dtype = float32
    

#     feat_labels = [#"lda sim", "w2v sim", "dirrel", "lexvector", "avg stimuli betweenness","avg response betweenness", "wup avg",
#                    "lda sim","w2v sim", "glove sim","w2g energy","w2g sim",
#                     "max autoex sim", "avg autoex sim",
# #                     "max stimuli betweenness", "avg stimuli betweenness",# "total stimuli betweenness",
# #                     "max response betweenness", "avg response betweenness",# "total response betweenness",
# #                     "max stimuli load", "avg stimuli load",# "total stimuli load",
# #                     "max response load", "avg response load",# "total response load",
#                     "dirrel",# "stimuli lexvector", "response lexvector",
#                     "wup max", "wup avg", #"path max", "path avg", "lch max", "lch avg",
#                     "max load", "avg load", "max betweenness", "avg betweenness", "lexvector",
# #                    "lda sim", "w2v sim", "autoex sim mapped", "wup", "stimuli betweenness", "response betweenness", "stimuli load", "response load", "dirrel", "stimuli lexvector", "response lexvector",
#                     "extended lesk"
#                     ]
#     offset_labels = ["glove offset"]#"w2v offset", "glove offset", "w2g offset"]
# #     embedding_labels = ["w2v all", "glove all", "w2g all sim", "w2g all energy", "w2g all both"]
#     
# #     feature_sets = [("max load", ["max response load", "max stimuli load"]),
# #                     ("avg load", ["avg response load", "avg stimuli load"]),
# #                     ("max betweenness", ["max response betweenness", "max stimuli betweenness"]),
# #                     ("avg betweenness", ["avg response betweenness", "avg stimuli betweenness"]),
# #                     ("lexvector", ["stimuli lexvector", "response lexvector"])]
    output_dir = "results/{}".format(dataset_to_test)
    pickle_dir = "features/{}".format(dataset_to_test)
    
    
    
    feat_labels_to_scale = [#"lda sim", "w2v sim", "dirrel", "lexvector", "avg stimuli betweenness","avg response betweenness", "wup avg",
                   "lda sim","w2v sim", "glove sim","w2g energy","w2g sim",
                    "max autoex sim", "avg autoex sim",
#                     "dirrel",# "stimuli lexvector", "response lexvector",
                    "wup max", "wup avg", #"path max", "path avg", "lch max", "lch avg",
                    "max load", "avg load", "max betweenness", "avg betweenness",
                    "lexvector","dirrel",
#                     "extended lesk"
                    ]
    feat_labels_no_scale = ["w2v offset",]
    
    with open(join(pickle_dir, "strengths.pkl"), "rb") as strength_file:
        targets = pickle.load(strength_file, encoding="latin1")#https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    targets = targets*100 #seems to help
    test_size = int(targets.shape[0] * 0.2) #hold out 20% for testing
#     targets = targets[test_size:]
#     train_size = int(targets.shape[0] * 0.8)
    
#     experiments = [("ranking all feats + lesk", feat_labels + ["glove offset"])]
    experiments = [("all +lexvector scaling -lesk", (feat_labels_to_scale, feat_labels_no_scale))]
#     experiments.append(("testing best features (w2v)", ["w2v sim", "w2v offset"]))
#     for feature in feat_labels :#+ offset_labels:#embedding_labels:
#         experiments.append(("{} only on test".format(feature), [feature]))
#     for feature in ["w2v sim", "glove sim", "w2g sim", "w2g energy", "w2v offset", "glove offset", "w2g offset"]:
#         experiments.append(("{} only on test".format(feature), [feature]))
#     for feature in feat_labels + ["w2v offset"]:
#         experiments.append(("minus {} (with w2v) on test".format(feature), [f for f in feat_labels + ["w2v offset"] if f != feature]))
#     for feature in ["w2v sim", "w2v offset"]:
#         experiments.append(("minus {} (with w2v) on test".format(feature), feat_labels + [f for f in ["w2v sim", "w2v offset"] if f != feature]))
#     experiments.append(("all feats with w2v offset on test", feat_labels + ["w2v offset"]))
#     for offset in offset_labels:
#         experiments.append(("all feats with {} on test".format(offset), feat_labels + [offset]))
#     for embedding in embedding_labels:
#         experiments.append(("all feats with {} on test".format(embedding), feat_labels + [embedding]))
#     experiments.append(("all embeddings and sims on test", feat_labels + offset_labels))
#     for feature_set_label, feature_set in feature_sets:
#     experiments.append(("minus w2v and glove sims (with w2v) on test".format(feature), [f for f in feat_labels + ["w2v offset"] if f not in ["w2v sim", "glove sim"]]))
    for label, feats in experiments:#, "w2v offset"] + feat_labels:
#     def run_experiment(experiment):
#         label, feats = experiment
#         output_loc = join(output_dir, label)
#         if exists(output_loc):
#             print("Found previous {} result. Skipping.".format(label))
# #             return
#             continue
        print("{}\tStarting test for {}".format(strftime("%y-%m-%d_%H:%M:%S"), label))
        
        
#         feature_subset_vects = []
#         for feat in feats:
# #             if feat == feature:
# #                 continue
#             with open(join(pickle_dir, "{}.pkl".format(feat)), "rb") as feature_file:
#                 vects = pickle.load(feature_file, encoding="latin1")#https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
#             feature_subset_vects.append(vects)
#             
#         feature_subset_vects = hstack(feature_subset_vects)
#         
#         test_X = feature_subset_vects[:test_size]
#         train_X = feature_subset_vects[test_size:]
#         
#         scaler = StandardScaler()
#         train_X = scaler.fit_transform(train_X)
#         test_X = scaler.transform(test_X)
        
        
        feats_to_scale, feats_no_scale = feats
        
        
#         feats_to_scale = feats_to_scale + feats_no_scale
#         feats_no_scale=[]
#         
        feature_vects_to_scale = []
        for feat in feats_to_scale:
            with open(join(pickle_dir, "{}.pkl".format(feat)), "rb") as feature_file:
                vects = pickle.load(feature_file, encoding="latin1")#https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
            feature_vects_to_scale.append(vects)
        
        feature_vects_to_scale = hstack(feature_vects_to_scale)
        train_X_to_scale = feature_vects_to_scale[test_size:]
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X_to_scale)
        feature_subset_vects = scaler.transform(feature_vects_to_scale)
#         feature_subset_vects = feature_vects_to_scale
        
        if len(feats_no_scale) > 0:
            feature_vects_no_scale = []
            for feat in feats_no_scale:
                with open(join(pickle_dir, "{}.pkl".format(feat)), "rb") as feature_file:
                    vects = pickle.load(feature_file, encoding="latin1")#https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
                feature_vects_no_scale.append(vects)
            
            feature_vects_no_scale = hstack(feature_vects_no_scale)
            feature_subset_vects = hstack((feature_subset_vects,feature_vects_no_scale))
            
        
        test_X = feature_subset_vects[:test_size]
        train_X = feature_subset_vects[test_size:]

        
        test_Y = targets[:test_size]
        train_Y = targets[test_size:]
        print("{}\tVectorization complete".format(strftime("%y-%m-%d_%H:%M:%S")))


        num_features = feature_subset_vects.shape[1]
        print(num_features)
        num_units = int(ceil(float(num_features)/2)) #2?
        
        if num_units < 5:
            num_units=5
        
        
        
#         def create_model():
        model = Sequential()
        model.add(Dense(num_units, input_dim=num_features))
        model.add(Dropout(0.5))
        model.add(Dense(num_units, input_dim=num_units))
        model.add(Dropout(0.5))
        model.add(Dense(1, input_dim=num_units, activation="relu"))
        model.compile(loss="mse", optimizer="adam")
#             return model
        
#         reg  = KerasRegressor(build_fn=create_model, epochs=args.epoch, batch_size=args.batchsize, verbose=1)
#         kselect = SelectKBest(mutual_info_regression, "all")
#         kselect.fit(train_X, train_Y)
        
#         print("Optimal number of features: {}\n".format(rfecv.n_features_))
#         for feat, mask, rank in zip(feat_labels, kselect.get_support(), kselect.scores_):
#             print("{}\t{}\t{}".format(feat, mask, rank))
#         
#         reg.fit(train_X,train_Y, epochs=args.epoch, batch_size=args.batchsize, verbose=1)
        model.fit(train_X,train_Y, epochs=args.epoch, batch_size=args.batchsize, verbose=1)
        test_P=model.predict(test_X)
        print(spearmanr(test_Y, test_P))
        print(pearsonr(test_Y,test_P))
        
        
    #     chainer.serializers.save_npz("eat.model", model)
        print("{}\tTraining {} finished".format(strftime("%y-%m-%d_%H:%M:%S"), label))
    
#    p = Pool(11)
#    p.map(run_experiment, experiments)
    
if __name__ == '__main__':
    for d in ["evoc", "usf", "eat"]:
        main(d)
