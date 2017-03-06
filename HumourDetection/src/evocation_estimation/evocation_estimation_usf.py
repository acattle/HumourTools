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
from evocation_reader import USFGraph
from evocation_feature_extractor import FeatureExtractor
from numpy import array, vstack
from chainer import Variable
from chainer.datasets.tuple_dataset import TupleDataset
from chainer.datasets.sub_dataset import split_dataset_random
from chainer.reporter import report
import pickle
from time import strftime

#needed for unpickling to work
from autoextend import AutoExtendEmbeddings
from wordnet_graph import WordNetGraph
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
        return self.l3(h2)

class MeanSquaredRegression(chainer.Chain):
    def __init__(self, predictor):
        super(MeanSquaredRegression, self).__init__(predictor=predictor)
        
    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss


def main():
    parser = argparse.ArgumentParser(description='Evocation Estimation')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    
    try:
    
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
        print("{}\tloading USF".format(strftime("%y-%m-%d_%H:%M:%S")))
        eat = USFGraph("PairsFSG.net")
        associations = eat.get_all_associations()
        print("{}\tUSF loaded".format(strftime("%y-%m-%d_%H:%M:%S")))
    
        #load feature extractors
        print("{}\tLoading feature extractor".format(strftime("%y-%m-%d_%H:%M:%S")))
        lda_loc="c:/Users/Andrew/Desktop/vectors/lda_prep_no_lemma/no_lemma.lda"
        wordids_loc="c:/Users/Andrew/Desktop/vectors/lda_prep_no_lemma/lda_no_lemma_wordids.txt.bz2"
        tfidf_loc="c:/Users/Andrew/Desktop/vectors/lda_prep_no_lemma/lda_no_lemma.tfidf_model"
        w2v_loc="c:/Users/Andrew/Desktop/vectors/GoogleNews-vectors-negative300.bin"
#         lda_loc="/home/acattle/lda/no_lemma.lda"
#         wordids_loc="/home/acattle/lda/lda_no_lemma_wordids.txt.bz2"
#         tfidf_loc="/home/acattle/lda/lda_no_lemma.tfidf_model"
#         w2v_loc="/home/acattle/vectors/word2vec_main_GoogleNews-vectors-negative300.bin"
        autoex_pkl="autoextend.pkl"
        betweenness_pkl="wordnet_betweenness.pkl"
        load_pkl="wordnet_load.pkl"
        wordnetgraph_pkl="wordnet_graph.pkl"
        feature_extractor = FeatureExtractor(lda_loc, wordids_loc, tfidf_loc, w2v_loc, autoex_pkl, betweenness_pkl, load_pkl, wordnetgraph_pkl)
        print("{}\tFeature extractor loaded".format(strftime("%y-%m-%d_%H:%M:%S")))
        
        print("{}\tExtracting features".format(strftime("%y-%m-%d_%H:%M:%S")))
        
        partial_extractor = partial(get_feature_vectors, feature_extractor=feature_extractor)
        association_chunks = [[associations[i] for i in xrange(len(associations)) if (i % 3) == r] for r in range(3)]
        p=Pool(3)
        data = []
        target = []
        for data_p, target_p in p.map(partial_extractor, association_chunks):
            data.append(data_p)
            target.append(target_p)
        p.close()
        
        data = vstack(data)
        target = vstack(target)
        print("{}\tFeatures extracted".format(strftime("%y-%m-%d_%H:%M:%S")))
        
        with open("usf_feature_matrix.pkl", "wb") as data_pkl:
            pickle.dump(data, data_pkl)
        with open("usf_target_vector.pkl", "wb") as target_pkl:
            pickle.dump(target, target_pkl)
        
        print("{}\tStarting training".format(strftime("%y-%m-%d_%H:%M:%S")))
        data = Variable(array(data))
        target = Variable(array(target))
        
        dataset=TupleDataset(data,target)
        
        train, test = split_dataset_random(dataset, len(dataset)*0.8)
    
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
    
        # Take a snapshot at each epoch
        trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    
        # Write a log of evaluation statistics for each epoch
        trainer.extend(extensions.LogReport())
    
        # Save two plot images to the result dir
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch',
                                  file_name='loss_usf.png'))
        trainer.extend(
            extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                                  'epoch', file_name='accuracy_usf.png'))
    
        # Print selected entries of the log to stdout
        # Here "main" refers to the target link of the "main" optimizer again, and
        # "validation" refers to the default name of the Evaluator extension.
        # Entries other than 'epoch' are reported by the Classifier link, called by
        # either the updater or the evaluator.
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    
        # Print a progress bar to stdout
        trainer.extend(extensions.ProgressBar())
    
        if args.resume:
            # Resume from a snapshot
            chainer.serializers.load_npz(args.resume, trainer)
    
        # Run the training
        trainer.run()
        
        chainer.serializers.save_npz("usf.model", model)
        print("{}\tTraining finished".format(strftime("%y-%m-%d_%H:%M:%S")))
    except Exception,e:
        print e
        raise 

if __name__ == '__main__':
    main()