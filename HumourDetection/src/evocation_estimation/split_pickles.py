'''
Created on Apr 8, 2017

@author: Andrew
'''
from __future__ import print_function #for Python 2.7 compatibility
import pickle
import os
from numpy import float32, nan_to_num, vstack, inf
import random

feats_folder = "features"
# dataset="usf"
dataset="usf2"
if not os.path.exists(feats_folder):
    os.mkdir(feats_folder)
full_dir = os.path.join(feats_folder, dataset)
if not os.path.exists(full_dir):
    os.mkdir(full_dir)

# with open("usf_feats_all3.pkl", "rb") as feats_file:
with open("usf_feats_betterw2g.pkl", "rb") as feats_file:
    dataset_feats = pickle.load(feats_file)
print("usf loaded")
# print("eat loaded")

random.seed(10)
random.shuffle(dataset_feats)

feat_dicts=[]
strengths=[]
for _,_,feat_dict,strength in dataset_feats:
    feat_dicts.append(feat_dict)
    strengths.append(strength)
del dataset_feats

strengths=vstack(strengths).astype(float32)
strengths=strengths.reshape((strengths.shape[0],1))

with open(os.path.join(full_dir, "strengths.pkl"), "wb") as strength_file:
    pickle.dump(strengths, strength_file)
print("strenghts saved")
del strengths

feats = feat_dicts[0].keys()
print("{} feautres".format(len(feats)))
for feat in feats:
    feat_vect = []
    for feat_dict in feat_dicts:
        val = feat_dict[feat]
        if ("lexvector" not in feat) and ("offset" not in feat) and ((val == inf) or (val ==-inf)):
            val = 0
        feat_vect.append(val)
        del feat_dict[feat]
    feat_vect =nan_to_num(vstack(feat_vect).astype(float32))
    
    with open(os.path.join(full_dir, "{}.pkl".format(feat)), "wb") as feat_file:
        pickle.dump(feat_vect, feat_file)
    print("{} done: {}".format(feat, feat_vect.shape))
print("done")