'''
Created on Oct 17, 2016

@author: Andrew
'''
from random import shuffle, seed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import re
from pymongo import MongoClient
from sklearn.feature_extraction.dict_vectorizer import DictVectorizer
from scipy.sparse import hstack, vstack
from sklearn.svm import SVC
from sklearn.metrics import classification_report

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
    for left_id, right_id in pairs:
        left_vects.append(tweet_vects[tweet_map[left_id]])
        right_vects.append(tweet_vects[tweet_map[right_id]])
    
    left_matrix = vstack(left_vects)
    right_matrix = vstack(right_vects)
    diff_matrix = left_matrix - right_matrix
    return hstack([left_matrix, right_matrix, diff_matrix])








if __name__ == '__main__':
    seed(10) #set a seed for repeatability
    client = MongoClient()
    
    collections = ["GentlerSongs", "OlympicSongs", "BoringBlockbusters", "OceanMovies"]
    tweets = []
    tweets_by_col = {}
    for col in collections: #for each collection
        tweets_col = select_and_santize_tweets(client, col) #get the tweets
        tweets_by_col[col] = tweets_col #save them by collection so we can exclude the test collection from the vectorizer fitting
        tweets.extend(tweets_col) #add to the list of all tweets
    
    tweet_texts, tweet_dicts, tweet_id_map = extract_raw_feature_sets(tweets)
    for test_col in collections:
        #get test tweets and pairs
        test_tweets = tweets_by_col[test_col]
        test_pairs, test_labels = make_pairs(test_tweets)
        
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
            train_col_pairs, train_col_labels = make_pairs(col_tweets)
            train_pairs.extend(train_col_pairs)
            train_labels.extend(train_col_labels)
        
        #mix the pairs and labels so they aren't separated by collection
        train_pairs, train_labels = randomize_pairs(train_pairs, train_labels) 
        
        #we want to extra the texts for training purposes but we don't care about the map
        train_texts, train_dicts, _ = extract_raw_feature_sets(train_tweets)

        #pre-fit count vectorizer so we can avoid vectorizing the same text multiple times
        count_vect = CountVectorizer()
        count_vect.fit(train_texts) #vectorizer should fit all training texts (no testing)
        tweet_text_vects = count_vect.transform(tweet_texts) #get vectors for all tweets, including testing
        
        #pre-fit count vectorizer so we can avoid vectorizing the same text multiple times
        dict_vect = DictVectorizer()
        #Unlike count_vect, since we manually select the feature keys, it doesn't matter if dict_vect is fit to the test data
        tweet_dict_vects = dict_vect.fit_transform(tweet_dicts)
        
        #combine text and dictionary features into a single vector
        tweet_vects = hstack([tweet_text_vects, tweet_dict_vects])
        tweet_vects = tweet_vects.tocsr() #convert to csr to allow indexing
        
        train_matrix = tuples_to_matrix(tweet_vects, tweet_id_map, train_pairs)
        test_matrix = tuples_to_matrix(tweet_vects, tweet_id_map, test_pairs)
        
        clf = SVC(kernel='linear')
        print "SVM fit start"
        clf.fit(train_matrix, train_labels)
        print "SVM fitted"
        test_predict = clf.predict(test_matrix)
        
        print "{}".format(test_col)
        print classification_report(test_labels, test_predict)
        print "\n\n"
        
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