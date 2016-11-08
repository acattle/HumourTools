'''
Created on Sep 23, 2016

@author: Andrew
'''
from pymongo import MongoClient
import re

client = MongoClient()

atMentions = re.compile(ur"@\w+", flags=re.I|re.U)
atMidnight = re.compile(u"@midnight", flags=re.I|re.U)
hashtag = re.compile(ur"#\w+", flags=re.I|re.U)
features = [("usf fwa forward most", "usf fwa backward most", "usf fwa difference most", "usf fwa difference most sign"), ("usf fwa forward least", "usf fwa backward least", "usf fwa difference least", "usf fwa difference least sign"), ("usf fwa forward average",  "usf fwa backward average", "usf fwa difference average", "usf fwa difference average sign")]
cols = ["GentlerSongs", "OlympicSongs", "OceanMovies", "BoringBlockbusters"]
p_values = []
for featureF, featureB, featureD, featureS in features:
    print "Testing {} vs {}".format(featureF, featureB)
    lessMoreDiff = [] #holds difference in feature value for less funny - more funny
    for col in cols:
        tweets = []
        for tweet in client.tweets[col].find({"$and" : [{"total likes" : {"$gte" : 7}}, {featureF : {"$exists" : True}}, {featureB : {"$exists" : True}}]}):
            if "punch words" not in tweet:
                continue
            if (tweet["punch words"] == None) or (tweet["punch words"] == []):
                continue
            for word in tweet["punch words"]:
                if word == "None":
                    continue
                if not word:
                    continue
            mentions = atMentions.findall(tweet["text"])
            if len(mentions) > 1: #if more than 1 person is mentione
                continue
            elif len(mentions) == 1:
                if not atMidnight.match(mentions[0]): #if the mention someone other than @midngiht
                    continue
            if len(hashtag.findall(tweet["text"])) > 1: #if there's more than 1 hashtag
                continue
            
            if (tweet[featureF] > 0) and (tweet[featureB] > 0):
                tweet[featureD] = tweet[featureF] - tweet[featureB]
                sign = 0 #assume forward and back are equal
                if (tweet[featureF] - tweet[featureB]) > 0:
                    sign = 1
                elif ((tweet[featureF] - tweet[featureB])) < 0:
                    sign = -1
                tweet[featureS] = sign
                client.tweets[col].update({"_id" : tweet["_id"]}, tweet)
        
        