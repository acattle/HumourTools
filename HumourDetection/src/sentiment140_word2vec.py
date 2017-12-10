'''
Created on Oct 29, 2016

@author: Andrew
'''

from gensim.models import Word2Vec
import multiprocessing
from gensim.models.word2vec import LineSentence

token_loc = "C:\\Users\\Andrew\\Desktop\\Senti140\\tokens.txt"
model_loc = "C:\\Users\\Andrew\\Desktop\\Senti140\\tokens_100.model"
vector_loc = "C:\\Users\\Andrew\\Desktop\\Senti140\\tokens_100.vector"

# tweets_str = ""
# with open(token_loc, "r") as tweetFile:
#     tweets_str = tweetFile.read()
#     
# tweets = []
# for tweet in tweets_str.split("\n"):
#     tweets.append(tweet.split())

tweets = LineSentence(token_loc)

model = Word2Vec(tweets, size=100, window=5, min_count=5,
            workers=multiprocessing.cpu_count())

model.init_sims(replace=True)
model.save(model_loc)
model.save_word2vec_format(vector_loc, binary=True)