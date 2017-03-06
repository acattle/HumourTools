'''
Created on Jan 28, 2017

@author: Andrew
'''
from nltk.corpus import stopwords

english_stopwords = stopwords.words("english")

#https://www.cs.cmu.edu/~ark/TweetNLP/gimpel+etal.acl11.pdf
pos_to_ignore = ["D","P","X","Y", "T", "&", "~", "'"]

def get_features(tweet_token_pos, hashtag_words):
    hash_words = [word for word in hashtag_words if word not in english_stopwords]
    for token, pos in tweet_token_pos:
        if pos in pos_to_ignore:
            continue
        if token in english_stopwords:
            continue
        
        
if __name__ == "__main__":
    