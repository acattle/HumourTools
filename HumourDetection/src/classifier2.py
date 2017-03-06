'''
Created on Oct 17, 2016

@author: Andrew
'''
from random import shuffle, seed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import re
# from pymongo import MongoClient
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from scipy.sparse import hstack, vstack, csr_matrix, lil_matrix
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import codecs
from sklearn.externals import joblib
import time
import os
# from multiprocessing import Pool
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.pipeline import Pipeline
import numpy as np

likes="total likes"
punch_words = "punch words"
text = "text"
tweet_id = "tweet id"
text_vect = "text vector"
left_str = "left"
right_str = "right"
perplexity_2 = "perplexity 2"
perplexity_3 = "perplexity 3"
perplexity_4 = "perplexity 4"
pos_perplexity_2 = "pos perplexity 2"
pos_perplexity_3 = "pos perplexity 3"
pos_perplexity_4 = "pos perplexity 4"
fwa_forward_highest = "usf fwa forward most"
fwa_forward_lowest = "usf fwa forward least"
fwa_forward_average = "usf fwa forward average"
fwa_backward_highest = "usf fwa backward most"
fwa_backward_lowest = "usf fwa backward least"
fwa_backward_average = "usf fwa backward average"
fwa_difference_highest = "usf fwa difference most"
fwa_difference_lowest = "usf fwa difference least"
fwa_difference_average = "usf fwa difference average"
ngd_highest = "ngd furthest"
ngd_lowest = "ngd closest"
ngd_average = "ngd average"
w2v_highest = "w2v most"
w2v_lowest = "w2v least"
w2v_average = "w2v average"
all_features = [perplexity_2, perplexity_3, perplexity_4,\
                pos_perplexity_2, pos_perplexity_3, pos_perplexity_4,\
                fwa_forward_highest, fwa_forward_lowest, fwa_forward_average,\
                fwa_backward_highest, fwa_backward_lowest, fwa_backward_average,\
                fwa_difference_highest, fwa_difference_lowest, fwa_difference_average,\
                ngd_highest, ngd_lowest, ngd_average,\
                w2v_highest, w2v_lowest, w2v_average]
atMentions = re.compile(ur"@\w+", flags=re.I|re.U)
atMidnight = re.compile(u"@midnight", flags=re.I|re.U)
hashtag = re.compile(ur"#\w+", flags=re.I|re.U)

class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        return [{'length': len(text),
                 'num_sentences': text.count('.')}
                for text in posts]


class LeftRightExtractor(BaseEstimator, TransformerMixin):
    """Extract the subject & body from a usenet post in a single pass.

    Takes a sequence of strings and produces a dict of sequences.  Keys are
    `subject` and `body`.
    """
    def __init__(self,tweets):
        self.tweets = tweets
        super(LeftRightExtractor, self).__init__()
        
    def fit(self, x, y=None):
        return self

    def transform(self, pairs):
        features = np.recarray(shape=(len(pairs),),
                               dtype=[('left', object), ('right', object)])
        for i, pairs in enumerate(pairs):
            left_id, right_id = pair
            left=tweets
            features['body'][i] = bod

            prefix = 'Subject:'
            sub = ''
            for line in headers.split('\n'):
                if line.startswith(prefix):
                    sub = line[len(prefix):]
                    break
            features['subject'][i] = sub

        return features

class TweetVectSelector(BaseEstimator, TransformerMixin):
    def __init__(self, tweet_vects, tweet_map, index):
        self.tweet_vects = tweet_vects
        self.tweet_map = tweet_map
        self.index = index

    def fit(self, x, y=None):
        return self

    def transform(self, pairs):
        vectors = []
        for pair in pairs:
            vectors.append(self.tweet_vects[self.tweet_map[pair[self.index]]])
        
        return vstack(vectors)
    
class CalcDifference(BaseEstimator, TransformerMixin):
    def __init__(self, tweet_vects, tweet_map):
        self.tweet_vects = tweet_vects
        self.tweet_map = tweet_map

    def fit(self, x, y=None):
        return self

    def transform(self, pairs):
        diff_vectors = []
        for pair in pairs:
            left_vect = self.tweet_vects[self.tweet_map[pair[0]]]
            right_vect = self.tweet_vects[self.tweet_map[pair[1]]]
            diff_vectors.append(left_vect - right_vect)
            
        return vstack(diff_vectors)

def select_and_santize_tweets(client, col, db="tweets", minLikes=7):
    """
    Method for selecting all tweets from client.db.col with total likes > minLikes
    and without any extra #hashtags, @mentions, links, etc.
    """
    tweets = []
    for tweet in client[db][col].find({likes: {"$gte" : minLikes}}):
        #only looking at annotated tweets
        if punch_words not in tweet:
            continue
        if (tweet[punch_words] == None) or (tweet[punch_words] == []):
            continue
        for word in tweet[punch_words]:
            if word == "None":
                continue
            if not word:
                continue
        
        tweet_text = tweet[text]
        #filter out extra mentions or extra hashtags
        mentions = atMentions.findall(tweet_text)
        if len(mentions) > 1: #if more than 1 person is mentioned
            continue
        elif len(mentions) == 1:
            if not atMidnight.match(mentions[0]): #if the mention someone other than @midnight
                continue
        if len(hashtag.findall(tweet_text)) > 1: #if there's more than 1 hashtag
            continue
        
        #sanitize tweet text
        tweet_text = atMentions.sub("", tweet_text)
        tweet_text = hashtag.sub("", tweet_text)
        
        tweet[text] = tweet_text.strip()
        
        tweets.append(tweet)
    
    return tweets

def randomize_pairs(pairs, labels):
    random_indexes =range(len(pairs)) #get all possible indexes
    shuffle(random_indexes) #radnomize them
    pairs_randomized = []
    labels_randomized = []
    for i in random_indexes: #for each index
        labels_randomized.append(labels[i])
        pairs_randomized.append(pairs[i])
            
    return pairs_randomized, labels_randomized

def balance_pairs(pairs):
    pairs_balanced = []
    #create a list of half 0s and half 1s
    num_pairs = len(pairs)
    mid_point = num_pairs/2
    #create a balanced label set for each collection
    labels = [0] * mid_point + [1] * (num_pairs - mid_point)
    shuffle(labels) #shuffle the labels
    
    for i in range(num_pairs):
        if labels[i]  == 0: #if left should by funnier
            pairs_balanced.append(pairs[i]) #we don't need to chanfe the order
        else: #if right should be funnier
            left, right = pairs[i]
            pairs_balanced.append((right, left)) #swap the order
    
    return pairs_balanced, labels

def make_pairs(tweets, minDiff=1):
    """
    Generate balanced, shuffled tweet pairs for training and test data
    
    @param tweets: list of tweets for a single hashtag game
    @type tweets: [dict]
    @param minDiff: the smallest difference we accept between pairs
    @returns tuple containing list of pairs by tweet_id (stored as tuples) and list of labels (0 = left is funnier, 1 = right is funnier)
    @rtype ([(int, int)], [int])
    """
    pairs_unbalanced = [] #holds all pairs. index [0] is always funnier
    for i in xrange(len(tweets)): #for each tweet id
        tweet_i = tweets[i]
        tweet_i_likes = tweet_i[likes] #get the number of likes for tweet i
        tweet_i_id = tweet_i[tweet_id]
        for j in xrange(i+1, len(tweets)): #for each tweet id we haven't made pairs for yet
            tweet_j = tweets[j]
            tweet_j_id = tweet_j[tweet_id]
            likeDiff = tweet_i_likes - tweet_j[likes]
            if abs(likeDiff) < minDiff: #if they are not different enough
                continue #skip
            elif likeDiff < 0: #if j is funnier
                pairs_unbalanced.append((tweet_j_id, tweet_i_id))
            elif likeDiff > 0:  #i must be funnier than j
                pairs_unbalanced.append((tweet_i_id,  tweet_j_id))
            else: #this should never happen
                raise Exception("Threshold is less than 1: {}".format(minDiff))
    
    pairs_balanced, labels = balance_pairs(pairs_unbalanced)
    pairs_randomized, labels_randomized = randomize_pairs(pairs_balanced, labels)
    
    return pairs_randomized, labels_randomized

def extract_raw_feature_sets (tweets, dict_keys=all_features):
    id_to_index_map = {}
    texts = []
    dicts = []
    for i in xrange(len(tweets)):
        tweet = tweets[i]
        id_to_index_map[tweet[tweet_id]] = i
        texts.append(tweet[text])
        dicts.append({key:tweet[key] for key in all_features if key in tweet})
    
    return texts, dicts, id_to_index_map

def tuples_to_matrix(tweet_vects, tweet_map, pairs):
    left_vects = []
    right_vects = []
    #initialize empty matrices with len(pairs) rows and approrpriate number of columns
#     left_matrix = lil_matrix((len(pairs), tweet_vects[0].shape[1]))
#     right_matrix = lil_matrix((len(pairs), tweet_vects[0].shape[1]))
#     for i, pair in enumerate(pairs):
#         left_id, right_id = pair
#         left_matrix[i] = tweet_vects[tweet_map[left_id]]
#         right_matrix[i] = tweet_vects[tweet_map[right_id]]
    for left_id, right_id in pairs:
        left_vects.append(tweet_vects[tweet_map[left_id]])
        right_vects.append(tweet_vects[tweet_map[right_id]])
#     left_matrix = left_matrix.tocsr()
#     right_matrix = right_matrix.tocsr()
    left_matrix = vstack(left_vects,format="csr")
    right_matrix = vstack(right_vects,format="csr")
    diff_matrix = left_matrix - right_matrix
    return hstack([left_matrix, right_matrix, diff_matrix],format="csr")


def read_and_zip_tweets(file_loc):
    feature_names = [tweet_id]
    feature_names.extend(all_features)
    feature_names.append(text)
    tweets=[]
    with codecs.open(file_loc, "r", "utf-8") as tweet_file:
        for line in tweet_file:
            features = line.split("\t")
            
            for i in xrange(len(all_features)):
                if features[1+i] == "": #offset by 1 5o avoid tweet id
                    features[1+i] = None
                else:
                    features[1+i] = float(features[1+i])
                    
            
            tweet_dict = dict([item for item in zip(feature_names, features) if item[1] != None])
            
            features[-1] = features[-1].strip() #remove final \n
            
            tweets.append(tweet_dict)
    
    return tweets

def read_pairs(file_loc):
    pairs = []
    labels = []
    
    with open(file_loc, "r") as pairs_file:
        for line in pairs_file:
            id1, id2, label = line.split("\t")
            
            pairs.append((id1, id2))
            labels.append(int(label))
    
    return pairs, labels

def train_and_test(test_col, tweets_by_col, tweet_dicts, tweet_texts, tweet_id_map, results_dir="results", model_dir="models"):
    #get test tweets and pairs
#c        test_tweets = tweets_by_col[test_col]
#         test_pairs, test_labels = make_pairs(test_tweets)
    test_pairs, test_labels = read_pairs("tweet_data/{}_pairs.txt".format(test_col))
     
    #get training tweets and pairs
    train_tweets = []
    train_pairs = []
    train_labels = []
    for col in tweets_by_col:
        if col == test_col:
            continue #skip the testing col
        #we want all collections except the test collection
        col_tweets = tweets_by_col[col]
         
        #add to list of training tweets
        train_tweets.extend(col_tweets)
         
            #make pairs on a per collection basis to prevent left and right from being from different hashtag games
#             train_col_pairs, train_col_labels = make_pairs(col_tweets)
        train_col_pairs, train_col_labels = read_pairs("tweet_data/{}_pairs.txt".format(col))
        train_pairs.extend(train_col_pairs)
        train_labels.extend(train_col_labels)
    
    #mix the pairs and labels so they aren't separated by collection
    train_pairs, train_labels = randomize_pairs(train_pairs, train_labels) 

    pipeline = Pipeline([
        # Extract the subject & body
        ('leftright', leftRightExtractor()),
    
        # Use FeatureUnion to combine the features from subject and body
        ('union', FeatureUnion(
            transformer_list=[
    
                # Pipeline for pulling features from the post's subject line
                ('subject', Pipeline([
                    ('selector', ItemSelector(key='subject')),
                    ('tfidf', TfidfVectorizer(min_df=50)),
                ])),
    
                # Pipeline for standard bag-of-words model for body
                ('body_bow', Pipeline([
                    ('selector', ItemSelector(key='body')),
                    ('tfidf', TfidfVectorizer()),
                    ('best', TruncatedSVD(n_components=50)),
                ])),
    
                # Pipeline for pulling ad hoc features from post's body
                ('body_stats', Pipeline([
                    ('selector', ItemSelector(key='body')),
                    ('stats', TextStats()),  # returns a list of dicts
                    ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                ])),
    
            ],
    
            # weight components in FeatureUnion
            transformer_weights={
                'subject': 0.8,
                'body_bow': 0.5,
                'body_stats': 1.0,
            },
        )),
    
        # Use a SVC classifier on the combined features
        ('svc', SVC(kernel='linear')),
    ])
    
    # limit the list of categories to make running this example faster.
    categories = ['alt.atheism', 'talk.religion.misc']
    train = fetch_20newsgroups(random_state=1,
                               subset='train',
                               categories=categories,
                               )
    test = fetch_20newsgroups(random_state=1,
                              subset='test',
                              categories=categories,
                              )
    
    pipeline.fit(train.data, train.target)
    y = pipeline.predict(test.data)
    print(classification_report(y, test.target))
        
#     result_loc = os.path.join(results_dir,"{}_{}_rf.txt".format(time.strftime("%y-%m-%d_%H%M"), test_col))
#     with open(result_loc, "w") as outfile:
#         outfile.write(classification_report(test_labels, test_predict))
       
    #save the model
    #http://scikit-learn.org/stable/modules/model_persistence.html
#     model_loc = os.path.join(model_dir,"{}_{}_rf.pkl".format(time.strftime("%y-%m-%d_%H%M"), test_col))
#     joblib.dump(clf, model_loc)


if __name__ == '__main__':
    seed(10) #set a seed for repeatability
#     client = MongoClient()
    
    collections = ["GentlerSongs", "OlympicSongs", "BoringBlockbusters", "OceanMovies"]
    tweets = []
    tweets_by_col = {}
    
#     #TODO REMOVE
#     whitespace = re.compile(r"\s+", re.I|re.U)
#     def write_tweet(tweet, o_file):
#         features = [tweet[tweet_id]]
#         for feat in all_features:
#             feat_value = u"" #start as blank
#             if feat in tweet:
#                 feat_value = unicode(tweet[feat])
#             features.append(feat_value)
#         
#         features.append(whitespace.sub(ur" ", tweet[text]))
#         o_file.write(u"\t".join(features) + u"\n")
        
        
        
    for col in collections: #for each collection
#         tweets_col = select_and_santize_tweets(client, col) #get the tweets
        tweets_col = read_and_zip_tweets("tweet_data/{}_tweets.txt".format(col))
        tweets_by_col[col] = tweets_col #save them by collection so we can exclude the test collection from the vectorizer fitting
        tweets.extend(tweets_col) #add to the list of all tweets

#         #TODO REMOVE
#         with codecs.open("{}_tweets.txt".format(col), "w", "utf-8") as tweet_file:       
#             for tweet in tweets_col:
#                 write_tweet(tweet, tweet_file)
# 
#         pairs_col, labels_col = make_pairs(tweets_col)
#         
#         with open("{}_pairs.txt".format(col), "w") as pair_file:
#             for i in xrange(len(pairs_col)):
#                 pair_file.write("{}\t{}\t{}\n".format(pairs_col[i][0], pairs_col[i][1], labels_col[i]))
        
        
    
    tweet_texts, tweet_dicts, tweet_id_map = extract_raw_feature_sets(tweets)
    
#     pool = Pool(processes=4)
    
    for test_col in collections:
#         pool.apply_async(train_and_test, [test_col, tweets_by_col, tweet_dicts, tweet_texts, tweet_id_map])
#     #wait for jobs to finish
#     #https://docs.python.org/2/library/multiprocessing.html#multiprocessing.pool.multiprocessing.Pool.join
#     pool.close()
#     pool.join()
        train_and_test(test_col, tweets_by_col, tweet_dicts, tweet_texts, tweet_id_map)
         
    """
    What about features like length, # of pronouns, etc?
     
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    stop words?
    nltk tokenizing?
    stemming?
    min_df (document frequency)? Requires precomputed vocabulary
    higher rank ngrams? But it would create a larger vocabulary, slower training
     
    Other types of classifier?
    """