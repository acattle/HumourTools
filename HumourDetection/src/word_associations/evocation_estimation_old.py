'''
Created on Jan 9, 2017

@author: Andrew
'''
from __future__ import print_function #for Python 2.7 compatibility
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
# from evocation_reader import EATGraph, USFGraph, EvocationDataset
# from evocation_feature_extractor import FeatureExtractor
from numpy import array, float32, vstack,hstack, nan_to_num#, float64
from chainer.datasets.tuple_dataset import TupleDataset
from chainer.datasets.sub_dataset import split_dataset#, split_dataset_random
from chainer.reporter import report
from time import strftime
from scipy.stats.stats import spearmanr, pearsonr
import pickle
# import warnings
# warnings.filterwarnings("error")

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

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, 1),  # n_units -> n_out
        )

    def __call__(self, x, train=True):
#         h1 = F.dropout(F.relu(self.l1(x)), train=train)
#         h2 = F.dropout(F.relu(self.l2(h1)), train=train)
        with chainer.using_config('train', train):
            h1 = F.dropout(self.l1(x))
            h2 = F.dropout(self.l2(h1))
            return F.relu(self.l3(h2))

class MeanSquaredRegression(chainer.Chain):
    def __init__(self, predictor):
        super(MeanSquaredRegression, self).__init__(predictor=predictor)
        
    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        r2 = F.r2_score(y, t)
        
        #reshape into 1D arrays instead of Nx1 matrices
        #needed for compatibility with pearsonr
        y_1d = y.data.reshape((y.data.shape[0],))
        t_1d = t.reshape((t.shape[0],))
        pearson_cor, pearson_pvalue = pearsonr(y_1d, t_1d)
        spearman_cor, spearman_pvalue = spearmanr(y_1d, t_1d)
        report({'loss': loss, 'r2': r2, "pearson_cor":pearson_cor, "pearson_pvalue":pearson_pvalue, "spearman_cor":spearman_cor, "spearman_pvalue":spearman_pvalue}, self)
        return loss


def main(dataset_to_test):
    parser = argparse.ArgumentParser(description='Evocation Estimation')
    parser.add_argument('--batchsize', '-b', type=int, default=200,
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
    
#     # Load EAT data
#     print("{}\tloading Associations".format(strftime("%y-%m-%d_%H:%M:%S")))
# #     eat = EATGraph("../Data/eat/pajek/EATnew2.net")
# #     eat = EATGraph("../shortest_paths/EATnew2.net")
#     usf = USFGraph("../Data/PairsFSG2.net")
#     associations=usf.get_all_associations()
# #     usf = USFGraph("../shortest_paths/PairsFSG2.net")
# #     evocation = EvocationDataset("./evocation/", "mt_all")
# # #     evocation = EvocationDataset("C:/Users/Andrew/git/HumourDetection/HumourDetection/src/Data/evocation/", "mt_all")
# #     associations = evocation.get_all_associations()
#     print("{}\tAssociations loaded".format(strftime("%y-%m-%d_%H:%M:%S")))
#     
#     #load feature extractors
#     print("{}\tLoading feature extractor".format(strftime("%y-%m-%d_%H:%M:%S")))
#     lda_loc="c:/Users/Andrew/Desktop/vectors/lda_prep_no_lemma/no_lemma.lda"
#     wordids_loc="c:/Users/Andrew/Desktop/vectors/lda_prep_no_lemma/lda_no_lemma_wordids.txt.bz2"
#     tfidf_loc="c:/Users/Andrew/Desktop/vectors/lda_prep_no_lemma/lda_no_lemma.tfidf_model"
#     w2v_loc="c:/Users/Andrew/Desktop/vectors/GoogleNews-vectors-negative300.bin"
#     glove_loc="c:/Users/Andrew/Desktop/vectors/glove.840B.300d.withheader.bin"
#     w2g_model_loc="c:/Users/Andrew/Desktop/vectors/wiki.biggervocab.w2g"
#     w2g_vocab_loc="c:/Users/Andrew/Desktop/vectors/wiki.biggersize.gz"
# #     lda_loc="/home/acattle/lda/no_lemma.lda"
# #     wordids_loc="/home/acattle/lda/lda_no_lemma_wordids.txt.bz2"
# #     tfidf_loc="/home/acattle/lda/lda_no_lemma.tfidf_model"
# #     w2v_loc="/home/acattle/vectors/word2vec_main_GoogleNews-vectors-negative300.bin"
# #     glove_loc="/home/acattle/vectors/glove.840B.300d.withheader.txt"
# #     w2g_model_loc="/home/acattle/vectors/wiki.biggervocab.w2g"
# #     w2g_vocab_loc="/home/acattle/vectors/wiki.biggersize.gz"
#     autoex_loc = "c:/Users/Andrew/Desktop/vectors/autoextend.word2vecformat.bin"
# #     autoex_pkl="autoextend.pkl"
#     betweenness_pkl="wordnet_betweenness.pkl"
#     load_pkl="wordnet_load.pkl"
#     wordnetgraph_pkl="wordnet_graph.pkl"
#     feature_extractor = FeatureExtractor(lda_loc=lda_loc,
#                                          wordids_loc=wordids_loc,
#                                          tfidf_loc=tfidf_loc,
#                                          w2v_loc=w2v_loc,
# #                                          autoex_loc=autoex_pkl,
#                                          autoex_loc=autoex_loc,
#                                          betweenness_loc=betweenness_pkl,
#                                          load_loc=load_pkl,
#                                          wordnetgraph_loc=wordnetgraph_pkl,
#                                          glove_loc=glove_loc,
#                                          w2g_model_loc=w2g_model_loc,
#                                          w2g_vocab_loc=w2g_vocab_loc,
#                                          dtype=float32)
# #     print("{}\tFeature extractor loaded".format(strftime("%y-%m-%d_%H:%M:%S")))
#      
#     print("{}\tExtracting features".format(strftime("%y-%m-%d_%H:%M:%S")))
#     
# #     p=Pool(12)
# #     results = p.map(partial(getVector, feature_extractor=feature_extractor), associations)
# #     p.close()
# #     feature_vects, targets = map(*results)
#     
#     
#     feature_vects = feature_extractor.get_feature_vectors(associations[:20])
#     targets = zip(*associations)[2] #since the strength are at index 2 of the association tuples
#     feature_vects = array(feature_vects).astype(float32)
#     feature_vects = nan_to_num(feature_vects)
#     targets = array(targets, dtype=feature_vects.dtype)
#     targets = targets.reshape((targets.shape[0], 1))
#     data = TupleDataset(feature_vects, targets)
#     print("{}\tFeatures extracted".format(strftime("%y-%m-%d_%H:%M:%S")))
# 
#     print("{}\tLoading preextracted features".format(strftime("%y-%m-%d_%H:%M:%S")))
#     features_loc = "c:/Users/Andrew/git/HumourDetection/HumourDetection/src/word_associations/usf_feats_all3.pkl"
# #     features_loc = "c:/Users/Andrew/git/HumourDetection/HumourDetection/src/word_associations/eat_feats_all3.pkl"
#     with open(features_loc, "rb") as features_f:
#         association_feats = pickle.load(features_f)
#     
#     print("{}\tShuffling pairs".format(strftime("%y-%m-%d_%H:%M:%S")))
#     random.seed(10)
#     random.shuffle(association_feats)
#     
#     print("{}\tExcluding test set".format(strftime("%y-%m-%d_%H:%M:%S")))
#     test_size = int(len(association_feats) * 0.2) #hold out 20% for testing
#     association_feats = association_feats[test_size:]
#     
#     train_size = int(len(association_feats) * 0.8)  
#     
#     _, _, feature_dicts, targets = zip(*association_feats)
#     targets = array(targets, dtype=dtype)
#     targets = targets.reshape((targets.shape[0], 1))
#     targets = targets*100 #needed? Seems to help
    
    
    feat_labels = ["lda sim", "glove sim", "dirrel", "lexvector", "avg stimuli betweenness","avg response betweenness", "wup avg",
        #"lda sim","w2v sim", "glove sim","w2g energy","w2g sim",
#                    "max autoex sim", "avg autoex sim",
#                    "max stimuli betweenness", "avg stimuli betweenness",# "total stimuli betweenness",
#                    "max response betweenness", "avg response betweenness",# "total response betweenness",
#                    "max stimuli load", "avg stimuli load",# "total stimuli load",
#                    "max response load", "avg response load",# "total response load",
#                    "dirrel",# "stimuli lexvector", "response lexvector",
#                    "wup max", "wup avg", "path max", "path avg", "lch max", "lch avg",
#                     "max load", "avg load", "max betweenness", "avg betweenness", "lexvector"
#                    "lda sim", "w2v sim", "autoex sim mapped", "wup", "stimuli betweenness", "response betweenness", "stimuli load", "response load", "dirrel", "stimuli lexvector", "response lexvector",
#                     "extended lesk"
                    ]
    offset_labels = ["glove offset"]#"w2v offset", "glove offset", "w2g offset"]
#     embedding_labels = ["w2v all", "glove all", "w2g all sim", "w2g all energy", "w2g all both"]
    
#     feature_sets = [("max load", ["max response load", "max stimuli load"]),
#                     ("avg load", ["avg response load", "avg stimuli load"]),
#                     ("max betweenness", ["max response betweenness", "max stimuli betweenness"]),
#                     ("avg betweenness", ["avg response betweenness", "avg stimuli betweenness"]),
#                     ("lexvector", ["stimuli lexvector", "response lexvector"])]
    output_dir = "results/{}".format(dataset_to_test)
    pickle_dir = "features/{}".format(dataset_to_test)
    
#     #test each feature one at a time.
#     for feature in feature_dicts[0].keys():
#         print("{}\tStarting test for {}".format(strftime("%y-%m-%d_%H:%M:%S"), feature))
#         output_loc = join(output_dir, feature)
#         
#         feature_vects = []
#         for f_dict in feature_dicts:
#             feature_vects.append(f_dict[feature])
#         feature_vects = vstack(feature_vects).astype(dtype)
# #         feature_vects = nan_to_num(feature_vects)


    #will this cause problems when loading the large eat feature set?
#     feature_vects_no_offset = []
#     offset_vects = {offset:[] for offset in offset_labels}
#     for f_dict in feature_dicts:
#         association_vect = []
#         for feature in feat_labels:
#             val = f_dict[feature]
#             if ("lexvector" not in feature) and ((val == inf) or (val ==-inf)):
#                 val = 0
#             association_vect.append(val)
#         feature_vects_no_offset.append(hstack(association_vect))
#         for offset in offset_labels:
#             offset_vects[offset].append(f_dict[offset])            
#     feature_vects_no_offset = vstack(feature_vects_no_offset).astype(dtype)
#     feature_vects_no_offset=nan_to_num(feature_vects_no_offset)
#     all_offsets = []
#     for offset in offset_labels:
#         offsets_vect = nan_to_num(vstack(offset_vects[offset]).astype(dtype))
#         offset_vects[offset] = offsets_vect
#         all_offsets.append(offsets_vect)
#     offset_vects["no offset"] = empty((feature_vects_no_offset.shape[0], 0), dtype=dtype)
#     offset_vects["all offsets"] = hstack(all_offsets)
# 
#     all_labels = feat_labels# + offset_labels
#     del feat_labels
#     del offset_labels
#     
#     feature_vects = {feature:[] for feature in all_labels}
#     feature_vects["w2v offset"] = []
#     for feature in feature_vects:
#         for f_dict in feature_dicts:
#             val = f_dict[feature]
#             if ("lexvector" not in feature) and ("offset" not in feature) and ((val == inf) or (val ==-inf)):
#                 val = 0
#             feature_vects[feature].append(val)
#         feature_vects[feature] = nan_to_num(vstack(feature_vects[feature]).astype(dtype))
#     del feature_dicts
#     
#     all_labels = ["all"] + all_labels
#     #test one offset at a time
#     for offset in offset_vects:
    #test one feature at a time
    with open(join(pickle_dir, "strengths.pkl"), "rb") as strength_file:
        targets = pickle.load(strength_file, encoding="latin1")#https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    targets = targets*100 #seems to help
    test_size = int(targets.shape[0] * 0.2) #hold out 20% for testing
#     targets = targets[test_size:]
#     train_size = int(targets.shape[0] * 0.8)
    
    experiments = [("hand picked + glove", feat_labels + ["glove offset"])]
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
        output_loc = join(output_dir, label)
#         if exists(output_loc):
#             print("Found previous {} result. Skipping.".format(label))
# #             return
#             continue
        print("{}\tStarting test for {}".format(strftime("%y-%m-%d_%H:%M:%S"), label))
        
        feature_subset_vects = []
        for feat in feats:
#             if feat == feature:
#                 continue
            with open(join(pickle_dir, "{}.pkl".format(feat)), "rb") as feature_file:
                vects = pickle.load(feature_file,encoding="latin1")
            feature_subset_vects.append(vects)
#         with open(join(pickle_dir, "{}.pkl".format(feat)), "rb") as feature_file:
#             offfset_vects = pickle.load(feature_file)
#         feature_subset_vects.append(offfset_vects)
#         feature_subset_vects = hstack(feature_subset_vects)
#         feature_subset_vects=feature_subset_vects[test_size:] #hold out 20% for testing
#         feature_subset_vects = []
#         for feat in feature_vects:
#             if feat == feature:
#                 continue
#             feature_subset_vects.append(feature_vects[feat])
        feature_subset_vects = hstack(feature_subset_vects)
#         feature_subset_vects = feature_vects[feature]
        
        data = TupleDataset(feature_subset_vects, targets)
#         train, validation = split_dataset(data, train_size)
        test, train = split_dataset(data, test_size)
        print("{}\tVectorization complete".format(strftime("%y-%m-%d_%H:%M:%S")))


        num_features = feature_subset_vects.shape[1]
        print( num_features)
        num_units = int(ceil(float(num_features)/2)) #2?
        
        if num_units < 5:
            num_units=5

        # Set up a neural network to train
        # MeanSquaredRegression reports mean squared error and accuracy at every
        # iteration, which will be used by the PrintReport extension below.
        model = MeanSquaredRegression(MLP(num_units))
#         model = MeanSquaredRegression(MLP(args.unit))
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
            model.to_gpu()  # Copy the model to the GPU
    
        # Setup an optimizer
        optimizer = chainer.optimizers.Adam()
        optimizer.use_cleargrads(True) #needed?
        optimizer.setup(model)
    
     
         
    #     print("{}\tWriting data".format(strftime("%y-%m-%d_%H:%M:%S")))
    #     with open("evocation_feature_dataset.pkl", "wb") as data_pkl:
    #         pickle.dump(data, data_pkl)
    # #     print("{}\tLoading data".format(strftime("%y-%m-%d_%H:%M:%S")))
    # # #     with open("usf_dataset.pkl", "rb") as data_pkl:
    # #     with open("c:/users/andrew/desktop/vectors/usf_dataset.pkl", "rb") as data_pkl:
    # #         data=pickle.load(data_pkl)
            
#         train_size = int(len(data) * 0.8) #use 80% for training. Cast to int to avoid invalid indexes
#         train, test = split_dataset_random(data, train_size)   
        
        print("{}\tStarting training {}".format(strftime("%y-%m-%d_%H:%M:%S"), label))
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
        test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                     repeat=False, shuffle=False)
#         test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
#                                                      repeat=False, shuffle=False)
    
        # Set up a trainer
        updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
        trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=output_loc)
#         trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    
        # Evaluate the model with the test dataset for each epoch
        trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    
        # Dump a computational graph from 'loss' variable at the first iteration
        # The "main" refers to the target link of the "main" optimizer.
        trainer.extend(extensions.dump_graph('main/loss'))
    #     trainer.extend(extensions.dump_graph('main/r2'))
    #     trainer.extend(extensions.dump_graph('main/spearman_cor'))
    
        # Take a snapshot at each epoch
        trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    
        # Write a log of evaluation statistics for each epoch
        trainer.extend(extensions.LogReport())
    
        # Save two plot images to the result dir
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch',
                                  file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(['main/r2', 'validation/main/r2'],
                                  'epoch', file_name='r2.png'))
        trainer.extend(
            extensions.PlotReport(['main/spearman_cor', 'validation/main/spearman_cor'],
                                  'epoch', file_name='spearman.png'))
        trainer.extend(
            extensions.PlotReport(['main/pearson_cor', 'validation/main/pearson_cor'],
                                  'epoch', file_name='pearson.png'))
    
        # Print selected entries of the log to stdout
        # Here "main" refers to the target link of the "main" optimizer again, and
        # "validation" refers to the default name of the Evaluator extension.
        # Entries other than 'epoch' are reported by the Classifier link, called by
        # either the updater or the evaluator.
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/r2', 'validation/main/r2',
             'main/spearman_cor', 'validation/main/spearman_cor',
             'main/pearson_cor', 'validation/main/pearson_cor',
             'elapsed_time']))
    
        # Print a progress bar to stdout
        trainer.extend(extensions.ProgressBar())
    
        if args.resume:
            # Resume from a snapshot
            chainer.serializers.load_npz(args.resume, trainer)
    
        # Run the training
        trainer.run()
        
    #     chainer.serializers.save_npz("eat.model", model)
        print("{}\tTraining {} finished".format(strftime("%y-%m-%d_%H:%M:%S"), label))
    
#    p = Pool(11)
#    p.map(run_experiment, experiments)
    
if __name__ == '__main__':
    for d in ["evoc", "usf", "eat"]:
        main(d)
