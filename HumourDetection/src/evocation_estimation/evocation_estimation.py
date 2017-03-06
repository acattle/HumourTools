'''
Created on Jan 9, 2017

@author: Andrew
'''
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from evocation_reader import EATGraph, USFGraph, EvocationDataset
from evocation_feature_extractor import FeatureExtractor
from numpy import array, float32, nan_to_num
from chainer.datasets.tuple_dataset import TupleDataset
from chainer.datasets.sub_dataset import split_dataset_random
from chainer.reporter import report
from time import strftime
from scipy.stats.stats import spearmanr
import pickle

#needed for unpickling to work
from autoextend import AutoExtendEmbeddings
from wordnet_graph import WordNetGraph
# from functools import partial
# from multiprocessing import Pool

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
        h1 = F.dropout(F.relu(self.l1(x)), train=train)
        h2 = F.dropout(F.relu(self.l2(h1)), train=train)
#         h1 = F.dropout(self.l1(x), train=train)
#         h2 = F.dropout(self.l2(h1), train=train)
        return F.relu(self.l3(h2))

class MeanSquaredRegression(chainer.Chain):
    def __init__(self, predictor):
        super(MeanSquaredRegression, self).__init__(predictor=predictor)
        
    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        r2 = F.r2_score(y, t)
        spearman_cor, spearman_pvalue = spearmanr(y.data, t.data)
        report({'loss': loss, 'r2': r2, "spearman_cor":spearman_cor, "spearman_pvalue":spearman_pvalue}, self)
        return loss


def main():
    parser = argparse.ArgumentParser(description='Evocation Estimation')
    parser.add_argument('--batchsize', '-b', type=int, default=250,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
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

    # Set up a neural network to train
    # MeanSquaredRegression reports mean squared error and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = MeanSquaredRegression(MLP(args.unit))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load EAT data
    print("{}\tloading Evocation".format(strftime("%y-%m-%d_%H:%M:%S")))
#     eat = EATGraph("../Data/eat/pajek/EATnew2.net")
#     eat = EATGraph("../shortest_paths/EATnew2.net")
#     usf = USFGraph("../Data/PairsFSG2.net")
#     usf = USFGraph("../shortest_paths/PairsFSG2.net")
    evocation = EvocationDataset("./evocation/", "mt_all")
#     evocation = EvocationDataset("C:/Users/Andrew/git/HumourDetection/HumourDetection/src/Data/evocation/", "mt_all")
    associations = evocation.get_all_associations()
    print("{}\tEvocation loaded".format(strftime("%y-%m-%d_%H:%M:%S")))
 
    #load feature extractors
    print("{}\tLoading feature extractor".format(strftime("%y-%m-%d_%H:%M:%S")))
#     lda_loc="c:/Users/Andrew/Desktop/vectors/lda_prep_no_lemma/no_lemma.lda"
#     wordids_loc="c:/Users/Andrew/Desktop/vectors/lda_prep_no_lemma/lda_no_lemma_wordids.txt.bz2"
#     tfidf_loc="c:/Users/Andrew/Desktop/vectors/lda_prep_no_lemma/lda_no_lemma.tfidf_model"
#     w2v_loc="c:/Users/Andrew/Desktop/vectors/GoogleNews-vectors-negative300.bin"
    lda_loc="/home/acattle/lda/no_lemma.lda"
    wordids_loc="/home/acattle/lda/lda_no_lemma_wordids.txt.bz2"
    tfidf_loc="/home/acattle/lda/lda_no_lemma.tfidf_model"
    w2v_loc="/home/acattle/vectors/word2vec_main_GoogleNews-vectors-negative300.bin"
    autoex_pkl="autoextend.pkl"
    betweenness_pkl="wordnet_betweenness.pkl"
    load_pkl="wordnet_load.pkl"
    wordnetgraph_pkl="wordnet_graph.pkl"
    feature_extractor = FeatureExtractor(lda_loc, wordids_loc, tfidf_loc, w2v_loc, autoex_pkl, betweenness_pkl, load_pkl, wordnetgraph_pkl, dtype=float32)
    print("{}\tFeature extractor loaded".format(strftime("%y-%m-%d_%H:%M:%S")))
     
    print("{}\tExtracting features".format(strftime("%y-%m-%d_%H:%M:%S")))
    feature_vects = []
    targets = []
    
#     p=Pool(12)
#     results = p.map(partial(getVector, feature_extractor=feature_extractor), associations)
#     p.close()
#     feature_vects, targets = map(*results)
    
    
    for stimuli, response, strength in associations:
        feature_vects.append(feature_extractor.get_feature_vector(stimuli, response))
        targets.append(strength)
    feature_vects = array(feature_vects).astype(float32)
    feature_vects = nan_to_num(feature_vects)
    targets = array(targets, dtype=feature_vects.dtype)
    targets = targets.reshape((targets.shape[0], 1))
    data = TupleDataset(feature_vects, targets)
    print("{}\tFeatures extracted".format(strftime("%y-%m-%d_%H:%M:%S")))
     
    print("{}\tWriting data".format(strftime("%y-%m-%d_%H:%M:%S")))
    with open("evocation_feature_dataset.pkl", "wb") as data_pkl:
        pickle.dump(data, data_pkl)
#     print("{}\tLoading data".format(strftime("%y-%m-%d_%H:%M:%S")))
# #     with open("usf_dataset.pkl", "rb") as data_pkl:
#     with open("c:/users/andrew/desktop/vectors/usf_dataset.pkl", "rb") as data_pkl:
#         data=pickle.load(data_pkl)
        
    train_size = int(len(data) * 0.8) #use 80% for training. Cast to int to avoid invalid indexes
    train, test = split_dataset_random(data, train_size)   
    
    print("{}\tStarting training".format(strftime("%y-%m-%d_%H:%M:%S")))
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

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

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/r2', 'validation/main/r2', 'main/spearman_cor', 'validation/main/spearman_cor', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()
    
#     chainer.serializers.save_npz("eat.model", model)
    print("{}\tTraining finished".format(strftime("%y-%m-%d_%H:%M:%S")))

if __name__ == '__main__':
#     def getVector(association, feature_extractor):
#         stimuli, response, strength = association
#          
#         return (feature_extractor.get_feature_vector(stimuli, response), strength)
    main()