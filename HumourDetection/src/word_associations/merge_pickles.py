'''
Created on Apr 8, 2017

@author: Andrew
'''
from __future__ import print_function #for Python 2.7 compatibility
import pickle
import os
from numpy import float32, nan_to_num, vstack, inf, hstack
import random

feats_folder = "features"
for dataset in ["evoc","eat","usf"]:
# dataset="eat"
    if not os.path.exists(feats_folder):
        os.mkdir(feats_folder)
    full_dir = os.path.join(feats_folder, dataset)
    if not os.path.exists(full_dir):
        os.mkdir(full_dir)
    
    
    feature_sets = [("max load", ["max response load", "max stimuli load"]),
                        ("avg load", ["avg response load", "avg stimuli load"]),
                        ("max betweenness", ["max response betweenness", "max stimuli betweenness"]),
                        ("avg betweenness", ["avg response betweenness", "avg stimuli betweenness"]),
                        ("lexvector", ["stimuli lexvector", "response lexvector"])]
    for label, feats in feature_sets:
        vects = []
        for feat in feats:
            with open(os.path.join(full_dir, "{}.pkl".format(feat)), "rb") as feats_file:
                vects.append(pickle.load(feats_file, encoding='latin1'))
        vects = hstack(vects)
        
        with open(os.path.join(full_dir, "{}.pkl".format(label)), "wb") as feat_file:
            pickle.dump(vects, feat_file, protocol=2)
        print("{} done: {}".format(label, vects.shape))
    print("done")