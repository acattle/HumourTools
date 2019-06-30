'''
Created on Jun 4, 2019

:author: Andrew Cattle <acattle@connect.ust.hk>

Code uased to perform HAHA2019 experiment
'''

from util.dataset_readers.haha_2019_reader import read_haha2019_file
from humour_features.centrality.tensor_decomp import decompose_tensors
from sklearn.semi_supervised import LabelSpreading
from scipy.spatial.distance import cdist
from sklearn.svm.classes import LinearSVC
import csv
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor, ExtraTreesRegressor,\
    RandomForestClassifier, ExtraTreesClassifier

training_loc = "d:/datasets/HAHA 2019/haha_2019_train.csv"
test_loc = "d:/datasets/HAHA 2019/haha_2019_test.csv"

training = read_haha2019_file(training_loc)
test = read_haha2019_file(test_loc, test=True)

training_ids, training_docs, training_labels, training_score = zip(*training)
test_ids, test_docs, _, _ = zip(*test)

all_docs = training_docs + test_docs

tensors, _ = decompose_tensors(all_docs, win_size=5, cp_rank=50)

# tensors_loc = "d:/datasets/HAHA 2019/tensors.pkl"
# import joblib
# with open(tensors_loc, "rb") as t:
#     tensors = joblib.load(t)
 
training_tensors = tensors[:len(training_docs)]
test_tensors = tensors[len(training_docs):]
 
 
# label_prop_model = LabelSpreading()
# label_prop_model.fit(training_tensors, training_labels)
# training_label_prop_pred = label_prop_model.predict(training_tensors) #predicting the same data the model was trained on. This might cause issues, as noted below
# test_label_prop_pred = label_prop_model.predict(test_tensors)
#   
# center = np.mean(training_tensors, axis=0)
# training_dist = cdist(training_tensors, [center])
# test_dist=cdist(test_tensors, [center])
#  
# training = np.hstack([training_tensors, training_label_prop_pred.reshape((len(training_label_prop_pred),1)), training_dist])
# test = np.hstack([test_tensors, test_label_prop_pred.reshape((len(test_label_prop_pred),1)), test_dist])
training = training_tensors
test=test_tensors

# cls = LinearSVC()
cls=RandomForestClassifier(n_estimators=100)
# cls=ExtraTreesClassifier(n_estimators=100)
cls.fit(training, training_labels)
label_pred = cls.predict(test)

rgr = RandomForestRegressor(n_estimators=100)
# rgr = ExtraTreesRegressor(n_estimators=100)
rgr.fit(training, training_score)
score_pred = rgr.predict(test)

 
with open("d:/datasets/HAHA 2019/sparse.csv", "w", newline='') as o_f:
    writer = csv.writer(o_f)
     
    #write headerz
    writer.writerow(["id","is_humor","funniness_average"])
     
    for test_id, label_prediction, score_prediction in zip(test_ids, label_pred, score_pred):
        writer.writerow([test_id, label_prediction, score_prediction])