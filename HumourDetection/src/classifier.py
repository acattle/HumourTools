'''
Created on Oct 17, 2016

@author: Andrew
'''
import re
import codecs
import os
import itertools
import glob
from random import shuffle, seed
from multiprocessing import Pool
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from time import strftime, time, mktime, localtime
from nltk.corpus import stopwords
from _collections import defaultdict
from numpy import mean
from math import exp
import pickle
from sklearn.metrics.classification import accuracy_score,\
    precision_recall_fscore_support
from sklearn.model_selection import KFold,cross_val_predict
from random import random

w2v = "w2v_sim"
usf = "usf"
eat="eat"
perplexity="perplexity"

english_stopwords = list(stopwords.words("english"))
pos_to_ignore = ["D",  "~", ",", "!", "U", "E",     "T","&","P","X","Y"]

def make_pairs(tweets_features):
    #seperate the tweets by their label
    pair_dict = {}
    if tweets_features[tweets_features.keys()[0]][2] != None:
        win_list = []
        top10_list = []
        nont10_list = []
        tweets_lists = [nont10_list, top10_list, win_list]
        for tweet_id in tweets_features:
            ngram_features, other_features, tweet_label = tweets_features[tweet_id]
            tweets_lists[int(tweet_label)].append(tweet_id)
        
        for list1, list2 in itertools.combinations(tweets_lists, 2):
            #for ever 2-way paring of the 3 lists
            for t1, t2 in itertools.product(list1, list2):
                #get every pairwise combination
                pair_dict[(t1, t2)] = 0
                pair_dict[(t2, t1)] = 1
    else:
        for t1, t2 in itertools.combinations(tweets_features.keys(), 2):
            #get every pairwise combination
            print("none label")
            pair_dict[(t1, t2)] = None
            pair_dict[(t2, t1)] = None
    
    return pair_dict

# def make_test_pairs(tweets_features):
#     pair_dict = {}
#     for t1, t2 in itertools.combinations(tweets_features.keys(), 2):
#         #get every pairwise combination
#         if random()>0.5:
#             pair_dict[(t1, t2)] = 0
#         else:
#             pair_dict[(t2, t1)] = 1
#     return pair_dict

def get_pair_features(tweets_features):
    diff_str = "__DIFF__"
    left_str = "__LEFT__"
    right_str = "__RIGHT__"
    
    pairs = make_pairs(tweets_features)
    
    pair_features = []
    labels = []
    for pair, label in pairs.items():
        labels.append(label)
        pair_feats = {}
        
        l_tweet_id, r_tweet_id = pair
        l_ngram_features, l_other_features, _ = tweets_features[l_tweet_id] #we don't care about tweet labels anymore, only pair labels
        r_ngram_features, r_other_features, _ = tweets_features[r_tweet_id]
        
        ngram_keys = set(l_ngram_features.keys()) | set(r_ngram_features.keys())
        for key in ngram_keys:
            pair_feats[u"{}{}".format(diff_str,key)] = l_ngram_features.get(key,0) - r_ngram_features.get(key,0) #ngram_features should be defaultdicts but use get() anyway to avoid implicitly creating dictionary entries
            pair_feats[u"{}{}".format(left_str,key)] = l_ngram_features.get(key,0)
            pair_feats[u"{}{}".format(right_str,key)] = r_ngram_features.get(key,0)
            #TODO: calling defaultdict[key] creates that key if it doesn't alreay exist. Is this causing my memory explosion?
#             pair_feats[u"{}{}".format(diff_str,key)] = l_ngram_features[key] - r_ngram_features[key] #ngrams should be defaultdicts
#             pair_feats[u"{}{}".format(left_str,key)] = l_ngram_features[key]
#             pair_feats[u"{}{}".format(right_str,key)] = r_ngram_features[key]
        
        #TODO: should I also change to get() calls to avoid possibiliy of missing keys (even though other)features uses a fixed set of keys)
        other_keys = set(l_other_features.keys()) | set(r_other_features.keys())
        for key in other_keys:
#             if (l_other_features[key] != 0):
            pair_feats[u"{}{}".format(left_str,key)] = l_other_features[key]
#             if (r_other_features[key] != 0):
            pair_feats[u"{}{}".format(right_str,key)] = r_other_features[key]
            if (l_other_features[key] != 0) and (r_other_features[key] != 0):
                pair_feats[u"{}{}".format(diff_str,key)] = l_other_features[key] - r_other_features[key]
        
    
        pair_features.append((pair, label, pair_feats))
    
    return pair_features
        
        
# def prefix_dict_keys(prefix, dictionary):
#     renamed_items = []
#     for key, value in dictionary.items():
#         renamed_items.append((u"{}{}".format(prefix, key),value))
#     
#     return dict(renamed_items)

def get_ngram_features(tokens, ignore=[], pos=None, ignore_pos=[], ngrams=[1,2]):
    ngram_counts = defaultdict(int)
    
    tokens = [token.lower() for token in tokens]
    if pos:
        tokens = [u"_".join(tp) for tp in zip(tokens, pos) if (tp[0] not in ignore) and (tp[1] not in ignore_pos)]
    else:
        tokens = [token for token in tokens if token not in ignore]
        
    length = len(tokens)
    for i in range(length):
        for n in ngrams:
            j = i+n
            if j <= length: #if we can extract a valid ngram
                ngram = u" ".join(tokens[i:j])
                ngram_counts[ngram] += 1
    
    return ngram_counts

def get_evocation_features(evoc_loc, evoc_name):
    
    feature_labels = ["__MIN_{}_F__", "__AVG_{}_F__", "__MAX_{}_F__",
                      "__MIN_{}_B__", "__AVG_{}_B__", "__MAX_{}_B__",
                      "__OVERALL_MIN_{}_D__", "__OVERALL_AVG_{}_D__", "__OVERALL_MAX_{}_D__",
                      "__MIN_WORD_{}_D__","__AVG_WORD_{}_D__", "__MAX_WORD_{}_D__"]
    feature_labels = [label.format(evoc_name.upper()) for label in feature_labels]
    
    evoc_by_id = {}
    with codecs.open(evoc_loc, "r", encoding="utf-8") as evoc_file:
        for line in evoc_file:
            tweet_id, min_avg_max_evoc_f, min_avg_max_evoc_b, per_word_evoc_fs, per_word_evoc_bs  = line.strip().split("\t")
            
            #forward
            #TODO: I called min() and max() on negative log probs. I flipped their ordetr here, but I really should have fixed it in the files themselv
            max_evoc_f_prob, avg_evoc_f_prob, min_evoc_f_prob = [exp(-float(val)) for val in min_avg_max_evoc_f.split()]
            min_avg_max_per_word_evoc_f = [token_min_av_max_f.split(":") for token_min_av_max_f in per_word_evoc_fs.split()]
            per_word_evoc_f_prob = [exp(-float(v)) for min_avg_max in min_avg_max_per_word_evoc_f for v in min_avg_max]
            if (min_evoc_f_prob == 0.0): #if we had some OOV word
                if (max_evoc_f_prob > 0.0): #there's at least 1 valid reading
                    non_zero_evoc_f_prob = [v for v in per_word_evoc_f_prob if v > 0.0]
                    avg_evoc_f_prob = mean(non_zero_evoc_f_prob)
                    min_evoc_f_prob = min(non_zero_evoc_f_prob)
            
            #backward
            #TODO: I called min() and max() on negative log probs. I flipped their ordetr here, but I really should have fixed it in the files themselv
            max_evoc_b_prob, avg_evoc_b_prob, min_evoc_b_prob = [exp(-float(val)) for val in min_avg_max_evoc_b.split()]
            min_avg_max_per_word_evoc_b = [token_min_av_max_b.split(":") for token_min_av_max_b in per_word_evoc_bs.split()]
            per_word_evoc_b_prob = [exp(-float(v)) for min_avg_max in min_avg_max_per_word_evoc_b for v in min_avg_max]
            if min_evoc_b_prob == 0.0: #if we had some OOV words
                if (max_evoc_b_prob > 0.0): #there's at least 1 valid reading
                    non_zero_evoc_b_prob = [v for v in per_word_evoc_b_prob if v > 0.0]
                    avg_evoc_b_prob = mean(non_zero_evoc_b_prob)
                    min_evoc_b_prob = min(non_zero_evoc_b_prob)
 
            #difference
            valid_evoc_d = [f-b for f,b in zip(per_word_evoc_f_prob, per_word_evoc_b_prob) if (f != 0.0) and (b != 0.0)]
            
            min_word_evoc_d = 0
            avg_word_evoc_d = 0
            max_word_evoc_d = 0
            if len(valid_evoc_d) > 0:
                #ignore min since we're assuming a small difference won't affect results either way
                min_word_evoc_d = min(valid_evoc_d)
                avg_word_evoc_d = mean(valid_evoc_d)
                max_word_evoc_d = max(valid_evoc_d)
                             
            overall_min_evoc_d = 0
            overall_avg_evoc_d = 0
            overall_max_evoc_d = 0
            #TODO: fixed error where I was taking exp(-prob)
            if (min_evoc_f_prob != 0) and (min_evoc_b_prob != 0):
                overall_min_evoc_d = min_evoc_f_prob-min_evoc_b_prob
            if (avg_evoc_f_prob != 0) and (avg_evoc_b_prob != 0):
                overall_avg_evoc_d = avg_evoc_f_prob-avg_evoc_b_prob
            if (max_evoc_f_prob != 0) and (max_evoc_b_prob != 0):
                overall_max_evoc_d = max_evoc_f_prob-max_evoc_b_prob
                
            features = (min_evoc_f_prob, avg_evoc_f_prob, max_evoc_f_prob,
                        min_evoc_b_prob, avg_evoc_b_prob, max_evoc_b_prob,
                        overall_min_evoc_d, overall_avg_evoc_d, overall_max_evoc_d,
                        min_word_evoc_d, avg_word_evoc_d, max_word_evoc_d)
            evoc_by_id[tweet_id] = dict(zip(feature_labels,features))
    
    return evoc_by_id            

def process_file(file_loc):
    name = os.path.splitext(os.path.basename(file_loc))[0]
    print("{}\tprocessing {}".format(strftime("%y-%m-%d_%H:%M:%S"),name))
    
    #get tweets
    tweet_ids = []
    tweet_tokens = []
    tweet_pos = []
    tweet_labels = []
    with codecs.open(file_loc, "r", encoding="utf-8") as tweet_file:
        for line in tweet_file:
            line=line.strip()
            if line == "":
                continue
            line_split = line.split("\t")
            tweet_tokens.append(line_split[0].split())
            tweet_pos.append(line_split[1].split())
            tweet_ids.append(line_split[3])
            if len(line_split) >5:
                tweet_labels.append(int(line_split[5]))
            else:
                tweet_labels.append(None)

#     get perplexity features    
    perp_loc = u"{}.{}".format(file_loc, perplexity)
    perplexity_by_id = {}
    with codecs.open(perp_loc, "r", encoding="utf-8") as perp_file:
        perp_file.readline() #get rid of the header
        for line in perp_file:
            tweet_id, unigram_perplexity, bigram_perplexity, tokens, per_token_perplexity = line.split("\t") #note that per_token_perplexity still has a \n
            if unigram_perplexity != "None":
                unigram_perplexity = float(unigram_perplexity)
                bigram_perplexity = float(bigram_perplexity)
            else:
                unigram_perplexity = 0
                bigram_perplexity = 0
            perplexity_by_id[tweet_id] = {"__UNIGRAM_PERPLEXITY__" : unigram_perplexity,
                                          "__BIGRAM_PERPLEXITY__" : bigram_perplexity}
     
    #get w2v features    
    w2v_loc = u"{}.{}".format(file_loc, w2v)
    w2v_by_id = {}
    with codecs.open(w2v_loc, "r", encoding="utf-8") as w2v_file:
        w2v_file.readline() #get rid of the header
        for line in w2v_file:
#             if line.strip() != "":
            tweet_id, min_avg_max_w2v, per_token_w2v = line.split("\t", 2)
            min_w2v, avg_w2v, max_w2v = [float(val) for val in min_avg_max_w2v.split()]
            if min_w2v == 0.0: #if we had some OOV words
                non_zero_w2v = [float(v) for v in per_token_w2v.split() if float(v) != 0.0]
                if len(non_zero_w2v) > 0: #if there's at least one non-zero reading
                    min_w2v = min(non_zero_w2v)
                    avg_w2v = mean(non_zero_w2v)
                  
                #otherwise, 0 values are fine
                  
            w2v_by_id[tweet_id] = {"__MIN_W2V__" : min_w2v,
                                   "__AVG_W2V__" : avg_w2v,
                                   "__MAX_W2V__" : max_w2v}
     
    #get usf features    
    usf_loc = u"{}.{}".format(file_loc, usf)
    usf_by_id = get_evocation_features(usf_loc, usf)
     
    #get eat features
    eat_loc = u"{}.{}".format(file_loc, eat)
    eat_by_id = get_evocation_features(eat_loc, eat)
    
    
    
    tweets = zip(tweet_ids, tweet_tokens, tweet_pos, tweet_labels)
    
    #get ngram features
    hashtag = "#{}".format(re.sub("_", "", name.lower()))
    ignore = ["@midnight", hashtag]
    ignore.extend(english_stopwords)
    tweets_features = {}
    for tweet_id, tokens, pos, label in tweets:
#         ngram_features = get_ngram_features(tokens, ignore)
        ngram_features = {}
        
        #it's useful to differentiate ngram features from other features since
        #they are treated differently when it comes time to get the pairwaise different
        other_features = dict()
        #add perplexity features
        other_features.update(perplexity_by_id[tweet_id])
         
        #add w2v features
        other_features.update(w2v_by_id[tweet_id])
          
        #add usf features
        other_features.update(usf_by_id[tweet_id])
         
        #add eat features
        other_features.update(eat_by_id[tweet_id])
        
        tweets_features[tweet_id] = (ngram_features, other_features, label)
    
    return tweets_features
        
        

def train_and_test():
    start_time = localtime()
    print("{} Starting test".format(strftime("%y-%m-%d_%H%M")))
#     print("1gram perplexity, 2gram perplexity, w2v sim, USF, EAT")
#     print("1gram, 2gram, 1gram perplexity, 2gram perplexity, w2v sim, USF, EAT")
    semeval_dir = "D:/datasets/SemEval Data"
#     semeval_dir = "/home/acattle/SemEval"
    dirs = ["trial_dir/trial_data",
            "train_dir/train_data"]#,
            #"evaluation_dir/evaluation_data"]
#     train_dir = "train_dir/train_data"
#     trial_dir = "trial_dir/trial_data"
    trial_dir= "evaluation_dir/evaluation_data"
    tagged_dir = "tagged"
    experiment_dir = "{} top 5 feats on eval".format(strftime("%y-%m-%d_%H%M", start_time))
    prediction_dir = "predictions"
    experiment_dir_full_path = os.path.join(semeval_dir,experiment_dir)
    if not os.path.exists(experiment_dir_full_path):
        os.makedirs(experiment_dir_full_path)
     
    train_filenames = []
    for d in dirs:
        os.chdir(os.path.join(semeval_dir, d, tagged_dir))
        for f in glob.glob("*.tsv"):
            train_filenames.append(os.path.join(semeval_dir, d, tagged_dir,f))
    
    test_filenames= []
    os.chdir(os.path.join(semeval_dir, trial_dir, tagged_dir))
    for f in glob.glob("*.tsv"):
        test_filenames.append(os.path.join(semeval_dir, trial_dir, tagged_dir,f))
     
    p = Pool(3)
     
    print("{} extracting training features".format(strftime("%y-%m-%d_%H%M")))
    train_tweets_by_file = p.map(process_file, train_filenames)
#     train_tweets_by_file=[]
#     for f in train_filenames:
#         train_tweets_by_file.append(process_file(f))
#     print("{} making training pairs".format(strftime("%y-%m-%d_%H%M")))
#     train_pairs_labels_features_by_file = p.map(get_pair_features, train_tweets_by_file)
#     train_pairs_labels_features_by_file=[]
#     for train_tweets in train_tweets_by_file:
#         s = time()
#         train_pairs_labels_features_by_file.append(get_pair_features(train_tweets))
#         print("{}s".format(time()-s))
#     print("{} making training pairs finished".format(strftime("%y-%m-%d_%H%M")))
#     del train_tweets_by_file #free up memory
#     print("{} writing training pair features to disk".format(strftime("%y-%m-%d_%H%M")))
#     with open(os.path.join(experiment_dir_full_path, "train_pairs.pkl"), "wb") as train_pairs_file:
#         pickle.dump(train_pairs_labels_features_by_file, train_pairs_file)
#     print("{} training feature extraction finished".format(strftime("%y-%m-%d_%H%M")))
# 
#     train_pairs_labels_features =[]
#     for file_pairs_labels_features in train_pairs_labels_features_by_file:
#         train_pairs_labels_features.extend(file_pairs_labels_features)
#     
#     #randomize the order
#     seed(10) #set a seed for repeatability
#     shuffle(train_pairs_labels_features)
#     
#     train_pairs, train_labels, train_pair_dicts = zip(*train_pairs_labels_features)
#     
#     dict_vect = DictVectorizer()
#     train_vectors = dict_vect.fit_transform(train_pair_dicts)
#     del train_pair_dicts #free up RAM
#     
#     #save vectorizer
#     vectorizer_loc = os.path.join(experiment_dir_full_path,"vectorizer.pkl")
#     print("{} Writing vectorizer to {}".format(strftime("%y-%m-%d_%H%M"), vectorizer_loc))
#     #save the model
#     #http://scikit-learn.org/stable/modules/model_persistence.html
#     joblib.dump(dict_vect, vectorizer_loc)
    
    
    print("{} extracting test features".format(strftime("%y-%m-%d_%H%M")))
    test_tweets_by_file = p.map(process_file, test_filenames)
#     test_tweets_by_file=[]
#     for f in test_filenames:
#         test_tweets_by_file.append(process_file(f))
    print("{} making test pairs".format(strftime("%y-%m-%d_%H%M")))
    test_pairs_labels_features_by_file = p.map(get_pair_features, test_tweets_by_file)
# #     test_pairs_labels_features_by_file=[]
# #     for test_tweets in test_tweets_by_file:
# #         s = time()
# #         test_pairs_labels_features_by_file.append(get_pair_features(test_tweets))
# #         print("{}s".format(time()-s))
    print("{} making test pairs finished".format(strftime("%y-%m-%d_%H%M")))
    del test_tweets_by_file #free up memory


#     print("{} writing test pair features to disk".format(strftime("%y-%m-%d_%H%M")))
#     with open(os.path.join(experiment_dir_full_path, "test_pairs.pkl"), "wb") as test_pairs_file:
#         pickle.dump(test_pairs_labels_features_by_file, test_pairs_file)
#     print("{} test feature extraction finished".format(strftime("%y-%m-%d_%H%M")))
    
#     p.close()
#     with open(os.path.join(semeval_dir, "train_pairs_no_ngram.pkl"), "rb") as train_pairs_file:
#         train_pairs_labels_features_by_file=pickle.load(train_pairs_file)
#     with open(os.path.join(semeval_dir, "trial_pairs_no_ngram.pkl"), "rb") as trial_pairs_file:
#         test_pairs_labels_features_by_file = pickle.load(trial_pairs_file)
#     print("loaded")

#TODO:for debugging evocation reading
#     usf_f=0
#     usf_b=0
#     usf_d=0
#     neg_usf_d=0
#     eat_f=0
#     eat_b=0
#     eat_d=0
#     total=0
#     for t_f in train_tweets_by_file:
#         for tid in t_f:
#             total+=1
#             if t_f[tid][1]["__MIN_USF_F__"] > 0:
#                 usf_f+=1
#             if t_f[tid][1]["__MIN_USF_B__"] > 0:
#                 usf_b+=1
#             if (t_f[tid][1]["__MIN_USF_F__"] > 0) and (t_f[tid][1]["__MIN_USF_B__"] > 0):
#                 usf_d+=1
#             if t_f[tid][1]["__MIN_EAT_F__"] > 0:
#                 eat_f+=1
#             if t_f[tid][1]["__MIN_EAT_B__"] > 0:
#                 eat_b+=1
#             if (t_f[tid][1]["__MIN_EAT_F__"] > 0) and (t_f[tid][1]["__MIN_EAT_B__"] > 0):
#                 eat_d+=1
#      
#     print("{}/{}".format(usf_f, total))
#     print("{}/{}".format(usf_b, total))
#     print("{}/{}".format(usf_d, total))
#     print("{}/{}".format(eat_f, total))
#     print("{}/{}".format(eat_b, total))
#     print("{}/{}".format(eat_d, total))
    
    to_test= [#("best w/ ngram", True, ["__UNIGRAM_PERPLEXITY__","__BIGRAM_PERPLEXITY__","__MAX_USF_F__","__OVERALL_MAX_USF_D__","__AVG_WORD_USF_D__","__MIN_EAT_F__","__MIN_EAT_B__","__OVERALL_AVG_EAT_D__","__AVG_WORD_EAT_D__"]),
              #("best", False, ["__UNIGRAM_PERPLEXITY__","__BIGRAM_PERPLEXITY__","__MAX_USF_F__","__OVERALL_MAX_USF_D__","__AVG_WORD_USF_D__","__MIN_EAT_F__","__MIN_EAT_B__","__OVERALL_AVG_EAT_D__","__AVG_WORD_EAT_D__"]), 
              #("best a lot", False, ["__OVERALL_AVG_EAT_D__","__AVG_WORD_EAT_D__","__MIN_EAT_F__", "__AVG_EAT_F__", "__MAX_EAT_F__","__MIN_USF_F__", "__MAX_USF_F__","__BIGRAM_PERPLEXITY__","__AVG_USF_F__","__MIN_EAT_B__","__AVG_EAT_B__","__MAX_EAT_B__"]),
              #("best less", False, ["__OVERALL_AVG_EAT_D__","__AVG_WORD_EAT_D__","__MIN_USF_F__",__MAX_USF_F__","__BIGRAM_PERPLEXITY__","__AVG_USF_F__"])]
              
              
              ("1 from each", False, ["__MIN_EAT_B__","__MIN_EAT_F__","__OVERALL_MIN_EAT_D__","__AVG_USF_F__","__OVERALL_MIN_USF_D__","__MIN_USF_B__","__AVG_WORD_EAT_D__","__MIN_WORD_USF_D__"]),
              ("top 5", False, ["__MIN_EAT_B__","__MIN_EAT_F__","__OVERALL_MIN_EAT_D__","__AVG_USF_F__","__OVERALL_MIN_USF_D__"]),
              ("top 10", False, ["__MIN_EAT_B__","__MIN_EAT_F__","__OVERALL_MIN_EAT_D__","__AVG_USF_F__","__OVERALL_MIN_USF_D__","__OVERALL_AVG_USF_D__","__MIN_USF_F__","__MAX_EAT_B__","__OVERALL_AVG_EAT_D__","__AVG_EAT_F__"]),
              ("perps", False, ["__UNIGRAM_PERPLEXITY__","__BIGRAM_PERPLEXITY__"]),
              ("1 from each+ (no ngram)", False, ["__UNIGRAM_PERPLEXITY__","__BIGRAM_PERPLEXITY__","__MIN_EAT_B__","__MIN_EAT_F__","__OVERALL_MIN_EAT_D__","__AVG_USF_F__","__OVERALL_MIN_USF_D__","__MIN_USF_B__","__AVG_WORD_EAT_D__","__MIN_WORD_USF_D__"]),
              ("top 5+ (no ngram)", False, ["__UNIGRAM_PERPLEXITY__","__BIGRAM_PERPLEXITY__","__MIN_EAT_B__","__MIN_EAT_F__","__OVERALL_MIN_EAT_D__","__AVG_USF_F__","__OVERALL_MIN_USF_D__"]),
              ("top 10+ (no ngram)", False, ["__UNIGRAM_PERPLEXITY__","__BIGRAM_PERPLEXITY__","__MIN_EAT_B__","__MIN_EAT_F__","__OVERALL_MIN_EAT_D__","__AVG_USF_F__","__OVERALL_MIN_USF_D__","__OVERALL_AVG_USF_D__","__MIN_USF_F__","__MAX_EAT_B__","__OVERALL_AVG_EAT_D__","__AVG_EAT_F__"])]#,
              
              
    """
              ("1 from each+", True, ["__MIN_EAT_B__","__MIN_EAT_F__","__OVERALL_MIN_EAT_D__","__AVG_USF_F__","__OVERALL_MIN_USF_D__","__MIN_USF_B__","__AVG_WORD_EAT_D__","__MIN_WORD_USF_D__"]),
              ("top 5+", True, ["__MIN_EAT_B__","__MIN_EAT_F__","__OVERALL_MIN_EAT_D__","__AVG_USF_F__","__OVERALL_MIN_USF_D__"]),
              ("top 10+", True, ["__MIN_EAT_B__","__MIN_EAT_F__","__OVERALL_MIN_EAT_D__","__AVG_USF_F__","__OVERALL_MIN_USF_D__","__OVERALL_AVG_USF_D__","__MIN_USF_F__","__MAX_EAT_B__","__OVERALL_AVG_EAT_D__","__AVG_EAT_F__"]),
              ("perps+", True, ["__UNIGRAM_PERPLEXITY__","__BIGRAM_PERPLEXITY__"]),
              ("1 from each+", True, ["__UNIGRAM_PERPLEXITY__","__BIGRAM_PERPLEXITY__","__MIN_EAT_B__","__MIN_EAT_F__","__OVERALL_MIN_EAT_D__","__AVG_USF_F__","__OVERALL_MIN_USF_D__","__MIN_USF_B__","__AVG_WORD_EAT_D__","__MIN_WORD_USF_D__"]),
              ("top 5+", True, ["__UNIGRAM_PERPLEXITY__","__BIGRAM_PERPLEXITY__","__MIN_EAT_B__","__MIN_EAT_F__","__OVERALL_MIN_EAT_D__","__AVG_USF_F__","__OVERALL_MIN_USF_D__"]),
              ("top 10+", True, ["__UNIGRAM_PERPLEXITY__","__BIGRAM_PERPLEXITY__","__MIN_EAT_B__","__MIN_EAT_F__","__OVERALL_MIN_EAT_D__","__AVG_USF_F__","__OVERALL_MIN_USF_D__","__OVERALL_AVG_USF_D__","__MIN_USF_F__","__MAX_EAT_B__","__OVERALL_AVG_EAT_D__","__AVG_EAT_F__"])]#,
              
    
              ("all", True, ["__UNIGRAM_PERPLEXITY__","__BIGRAM_PERPLEXITY__","__MIN_W2V__", "__AVG_W2V__", "__MAX_W2V__", "__MIN_USF_F__","__AVG_USF_B__","__AVG_USF_F__","__MAX_USF_F__","__MIN_USF_B__", "__AVG_USF_B__", "__MAX_USF_B__","__OVERALL_MIN_USF_D__", "__OVERALL_AVG_USF_D__", "__OVERALL_MAX_USF_D__","__AVG_WORD_USF_D__", "__MAX_WORD_USF_D__","__MIN_EAT_F__", "__AVG_EAT_F__", "__MAX_EAT_F__","__MIN_EAT_B__", "__AVG_EAT_B__", "__MAX_EAT_B__","__OVERALL_MIN_EAT_D__", "__OVERALL_AVG_EAT_D__", "__OVERALL_MAX_EAT_D__","__AVG_WORD_EAT_D__", "__MAX_WORD_EAT_D__"]),
              ("ngram only", True, []),
              ("no ngram", False, ["__UNIGRAM_PERPLEXITY__","__BIGRAM_PERPLEXITY__","__MIN_W2V__", "__AVG_W2V__", "__MAX_W2V__", "__MIN_USF_F__","__AVG_USF_F__","__MAX_USF_F__","__MIN_USF_B__", "__AVG_USF_B__", "__MAX_USF_B__","__OVERALL_MIN_USF_D__", "__OVERALL_AVG_USF_D__", "__OVERALL_MAX_USF_D__","__MIN_WORD_USF_D__","__AVG_WORD_USF_D__", "__MAX_WORD_USF_D__","__MIN_EAT_F__", "__AVG_EAT_F__", "__MAX_EAT_F__","__MIN_EAT_B__", "__AVG_EAT_B__", "__MAX_EAT_B__","__OVERALL_MIN_EAT_D__","__MIN_WORD_EAT_D__", "__OVERALL_AVG_EAT_D__", "__OVERALL_MAX_EAT_D__","__AVG_WORD_EAT_D__", "__MAX_WORD_EAT_D__"]),
              ("w2v",False, ["__MIN_W2V__", "__AVG_W2V__", "__MAX_W2V__"]),
              ("usf all", False, ["__MIN_USF_F__","__AVG_USF_F__","__MAX_USF_F__","__MIN_USF_B__", "__AVG_USF_B__", "__MAX_USF_B__","__OVERALL_MIN_USF_D__", "__OVERALL_AVG_USF_D__", "__OVERALL_MAX_USF_D__","__MIN_WORD_USF_D__","__AVG_WORD_USF_D__", "__MAX_WORD_USF_D__"]),
              ("usf forward", False, ["__MIN_USF_F__", "__AVG_USF_F__", "__MAX_USF_F__"]),
              ("usf forward min", False, ["__MIN_USF_F__"]),
              ("usf forward avg", False, ["__AVG_USF_F__"]),
              ("usf forward max", False, ["__MAX_USF_F__"]),
              ("usf backward", False, ["__MIN_USF_B__", "__AVG_USF_B__", "__MAX_USF_B__"]),
              ("usf backward min", False, ["__MIN_USF_B__"]),
              ("usf backward avg", False, ["__AVG_USF_B__"]),
              ("usf backward max", False, ["__MAX_USF_B__"]),
              ("usf difference", False, ["__OVERALL_MIN_USF_D__", "__OVERALL_AVG_USF_D__", "__OVERALL_MAX_USF_D__","__MIN_WORD_USF_D__","__AVG_WORD_USF_D__", "__MAX_WORD_USF_D__"]),
              ("usf difference overall", False, ["__OVERALL_MIN_USF_D__", "__OVERALL_AVG_USF_D__", "__OVERALL_MAX_USF_D__"]),
              ("usf difference overall min", False, ["__OVERALL_MIN_USF_D__"]),
              ("usf difference overall avg", False, ["__OVERALL_AVG_USF_D__"]),
              ("usf difference overall max", False, ["__OVERALL_MAX_USF_D__"]),
              ("usf difference word", False, ["__MIN_WORD_USF_D__","__AVG_WORD_USF_D__", "__MAX_WORD_USF_D__"]),
              ("usf difference word min", False, ["__MIN_WORD_USF_D__"]),
              ("usf difference word avg", False, ["__AVG_WORD_USF_D__"]),
              ("usf difference word max", False, ["__MAX_WORD_USF_D__"]),
              ("eat all", False, ["__MIN_EAT_F__", "__AVG_EAT_F__", "__MAX_EAT_F__","__MIN_EAT_B__", "__AVG_EAT_B__", "__MAX_EAT_B__","__OVERALL_MIN_EAT_D__", "__OVERALL_AVG_EAT_D__", "__OVERALL_MAX_EAT_D__","__MIN_WORD_EAT_D__","__AVG_WORD_EAT_D__", "__MAX_WORD_EAT_D__"]),
              ("eat forward", False, ["__MIN_EAT_F__", "__AVG_EAT_F__", "__MAX_EAT_F__"]),
              ("eat forward min", False, ["__MIN_EAT_F__"]),
              ("eat forward avg", False, ["__AVG_EAT_F__"]),
              ("eat forward max", False, ["__MAX_EAT_F__"]),
              ("eat backward", False, ["__MIN_EAT_B__", "__AVG_EAT_B__", "__MAX_EAT_B__"]),
              ("eat backward min", False, ["__MIN_EAT_B__"]),
              ("eat backward avg", False, ["__AVG_EAT_B__"]),
              ("eat backward max", False, ["__MAX_EAT_B__"]),
              ("eat difference", False, ["__OVERALL_MIN_EAT_D__", "__OVERALL_AVG_EAT_D__", "__OVERALL_MAX_EAT_D__","__MIN_WORD_EAT_D__","__AVG_WORD_EAT_D__", "__MAX_WORD_EAT_D__"]),
              ("eat difference overall", False, ["__OVERALL_MIN_EAT_D__", "__OVERALL_AVG_EAT_D__", "__OVERALL_MAX_EAT_D__"]),
              ("eat difference overall min", False, ["__OVERALL_MIN_EAT_D__"]),
              ("eat difference overall avg", False, ["__OVERALL_AVG_EAT_D__"]),
              ("eat difference overall max", False, ["__OVERALL_MAX_EAT_D__"]),
              ("eat difference word", False, ["__MIN_WORD_EAT_D__","__AVG_WORD_EAT_D__", "__MAX_WORD_EAT_D__"]),
              ("eat difference word min", False, ["__MIN_WORD_EAT_D__"]),
              ("eat difference word avg", False, ["__AVG_WORD_EAT_D__"]),
              ("eat difference word max", False, ["__MAX_WORD_EAT_D__"])]
              
              """
    
    overall_output_file = os.path.join(semeval_dir,experiment_dir, "overall_accuracy.txt")
    with open(overall_output_file, "w") as output_file:
        for exp_label, ngram, features_subset in to_test:
            exp_start = time()
            experiment_dir_full_path = os.path.join(semeval_dir,experiment_dir, exp_label)
            if not os.path.exists(experiment_dir_full_path):
                os.makedirs(experiment_dir_full_path)
            
            train_tweets_by_file_key_subset = []
            for tweet_file in train_tweets_by_file:
                tweet_file_key_subset = {}
                for tweet_id in tweet_file:
                    ngram_features, other_features, label = tweet_file[tweet_id]
                    if not ngram:
                        ngram_features = {}
                    other_features_subset = dict([(key, other_features[key]) for key in other_features if key in features_subset])
                    tweet_file_key_subset[tweet_id] = (ngram_features, other_features_subset, label)
                train_tweets_by_file_key_subset.append(tweet_file_key_subset)
            print("{} making training pairs".format(strftime("%y-%m-%d_%H%M")))
            train_pairs_labels_features_by_file = p.map(get_pair_features, train_tweets_by_file_key_subset)
            
            #turns out SVC(kernel="linear") and LinearSVC() use different libraries. LinearSVC finishes way way fasterS
            clfs = [RandomForestClassifier(n_estimators=100, n_jobs=3)]#, LinearSVC(), SVC(kernel="rbf")]
            for clf, clf_type in zip(clfs, ["rf"]):#, "linsvm","rbfsvm"]):
                print("{} Starting {} classifier training for {}".format(strftime("%y-%m-%d_%H%M"), clf_type, exp_label))
#                 train_start = time()
                
                all_labels = []
                all_predictions = []
                precisions = []
                recalls = []
                f1s = []
                accuracies = []
#                 kf = KFold(n_splits=10)
#                 for train, test in kf.split(train_pairs_labels_features_by_file):
#                                        
#                     train_pairs_labels_features =[]
#                     for i in train:
#                         train_pairs_labels_features.extend(train_pairs_labels_features_by_file[i])
                train_pairs_labels_features =[]
                for file_pairs_labels_features in train_pairs_labels_features_by_file:
                    train_pairs_labels_features.extend(file_pairs_labels_features)
                         
                #randomize the order
                seed(10) #set a seed for repeatability
                shuffle(train_pairs_labels_features)
                 
                train_pairs, train_labels, train_pair_dicts = zip(*train_pairs_labels_features)
        
                dict_vect = DictVectorizer()
                train_vectors = dict_vect.fit_transform(train_pair_dicts)
                
                test_pairs_labels_vectors_by_file = []
                for test_pairs_labels_features in test_pairs_labels_features_by_file:
                    test_pairs, test_labels, test_pair_dicts = zip(*test_pairs_labels_features)
#                     test_filenames = []
#                     for i in test:
#                         test_pairs, test_labels, test_pair_dicts = zip(*train_pairs_labels_features_by_file[i])
#                         test_filenames.append(train_filenames[i])
                    test_vectors = dict_vect.transform(test_pair_dicts)
                    test_pairs_labels_vectors_by_file.append((test_pairs, test_labels, test_vectors))
#                 #     del test_pairs_labels_features_by_file #free RAM
                
                    
#                 cv_predictions = cross_val_predict(clf, train_vectors, train_labels, cv=10, n_jobs=4, pre_dispatch=12)
                
                #TODO: Uncomment all of this
                clf.fit(train_vectors, train_labels)
#                     print("{} {} classifier trained".format(strftime("%y-%m-%d_%H%M"), clf_type))
#                     training_time = time() - train_start
#                     m,s=divmod(training_time, 60)
#                     h,m=divmod(m, 60)
#                     h = str(int(h)).zfill(2)
#                     m = str(int(m)).zfill(2)
#                     s = str(int(s)).zfill(2)
#                     print("Training time: {}:{}:{}".format(h,m,s))
                
                #TODO: Uncomment all of this
#                     print("{} starting testing".format(strftime("%y-%m-%d_%H%M")))
                #TODO: uncomment after leave-one-file-out cross validation
#                     all_labels = []
#                     all_predictions = []
                for test_filename, test_pairs_labels_vectors in zip(test_filenames, test_pairs_labels_vectors_by_file):
#                 for test_pairs_labels_vectors in test_pairs_labels_vectors_by_file:
                    hashtag, extension = os.path.splitext(os.path.basename(test_filename))
                    print("{} Testing {}".format(strftime("%y-%m-%d_%H%M"), hashtag))
                    predict_dir_full_path = os.path.join(experiment_dir_full_path,prediction_dir,clf_type)
                    if not os.path.exists(predict_dir_full_path):
                        os.makedirs(predict_dir_full_path)
                    prediction_filename = os.path.join(predict_dir_full_path, hashtag+"_PREDICT"+extension)
                     
                    test_pairs, test_labels, test_vectors = test_pairs_labels_vectors
                    all_labels.extend(test_labels)
                     
                    test_predictions = clf.predict(test_vectors)
                    all_predictions.extend(test_predictions)
                    
                    precision, recall, f1, support = precision_recall_fscore_support(test_labels, test_predictions,average="micro")
                    precisions.append(precision)
                    recalls.append(recall)
                    f1s.append(f1)
                    accuracies.append(accuracy_score(test_labels, test_predictions))
                 
                    print("{} Writing predictions to {}".format(strftime("%y-%m-%d_%H%M"), prediction_filename))
                    prediction_strs = []
                    for test_pair, test_prediction in zip(test_pairs, test_predictions):
                        left_tweet_id, right_tweet_id = test_pair
                        prediction_strs.append("\t".join([left_tweet_id, right_tweet_id, str(test_prediction)]))
                        
                    prediction_str = "\n".join(prediction_strs)
                    with open(prediction_filename, "w") as prediction_file:
                        prediction_file.write(prediction_str)
                      
                  
                      
#                     model_loc = os.path.join(experiment_dir_full_path,"{}_{}_classifier.pkl".format(strftime("%y-%m-%d_%H%M", start_time),clf_type))
#                     print("{} Writing model to {}".format(strftime("%y-%m-%d_%H%M"), model_loc))
#                     #save the model
#                     #http://scikit-learn.org/stable/modules/model_persistence.html
#                     joblib.dump(clf, model_loc)
                
                print("{} experiment {} finished".format(strftime("%y-%m-%d_%H%M"), exp_label))
                exp_time = time() - exp_start
                m,s=divmod(exp_time, 60)
                h,m=divmod(m, 60)
                h = str(int(h)).zfill(2)
                m = str(int(m)).zfill(2)
                s = str(int(s)).zfill(2)
                print("Training time: {}:{}:{}".format(h,m,s))
                
                if (len(all_labels) > 0) and (all_labels[0] != None):
    #                     output_loc = os.path.join(predict_dir_full_path, "output.txt")
                    test_label = "{} - {}".format(exp_label, clf_type)
                    report_str = classification_report(all_labels, all_predictions, digits=6)
                    accuracy_str = "Accuracy: {}".format(accuracy_score(all_labels, all_predictions))
                    accuracies_str = "\t".join([str(acc) for acc in accuracies])
    #                     report_str = classification_report(train_labels, cv_predictions, digits=6)
    #                     accuracy_str = "Accuracy: {}".format(accuracy_score(train_labels, cv_predictions))
    #                     with open(output_loc, "w") as output_file:
                    output_file.write(test_label)
                    output_file.write("\n")
                    output_file.write(report_str)
                    output_file.write("\n")
                    output_file.write(accuracy_str)
                    output_file.write("\n")
                    output_file.write(accuracies_str)
                    output_file.write("\n")
                    print(test_label)
                    print(report_str)
                    print(accuracy_str)
                    print("\n")
    
    print("{} Finished".format(strftime("%y-%m-%d_%H%M")))
    running_time = time() - mktime(start_time)
    m,s=divmod(running_time, 60)
    h,m=divmod(m, 60)
    h = str(int(h)).zfill(2)
    m = str(int(m)).zfill(2)
    s = str(int(s)).zfill(2)
    print("Total running time {}:{}:{}".format(h,m,s))


if __name__ == '__main__':
    train_and_test()
