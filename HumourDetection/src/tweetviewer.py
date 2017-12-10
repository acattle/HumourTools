'''
Created on Nov 22, 2016

@author: Andrew
'''

from pymongo import MongoClient

c = MongoClient()

for tweet in c.tweets.OlympicSongs.find({"$and" : [{"total likes" : {"$gte" : 7}}, {"punch words" : {"$exists" : True}}]}):
    if len(tweet["punch words"]) > 1:
        print tweet["text"]