'''
Created on Sep 20, 2016

@author: Andrew
'''
'''
Created on Sep 20, 2016

@author: Andrew
'''
from pymongo import MongoClient
import re

client = MongoClient()

atMentions = re.compile(ur"@\w+", flags=re.I|re.U)
atMidnight = re.compile(u"@midnight", flags=re.I|re.U)
hashtag = re.compile(ur"#\w+", flags=re.I|re.U)
    
likes = "total likes"
features = ["usf fwa difference most", "usf fwa difference least", "usf fwa difference average"] 
cols = ["GentlerSongs", "OlympicSongs", "OceanMovies", "BoringBlockbusters"]

threshold = 1 #minimum difference in likes/retweets to be counted as a pair
print "For a threshold of {}".format(threshold)
for feature in features:
    print "for {}".format(feature)
    pairs = 0
    tweetCount = 0
    negCount = 0
    for col in cols:
        tweets = []
        for tweet in client.tweets[col].find({"$and" : [{"total likes" : {"$gte" : 7}}, {feature : {"$exists" : True}}]}):
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
            tweets.append(tweet)
        
        tweetCountCol = len(tweets)
        print "\t{} has {} tweets".format(col, tweetCountCol)
        
        pairsCol = 0
        negCountCol = 0
        for i in range(len(tweets)):
            for j in range(i+1, len(tweets)):
                if abs(tweets[i][likes] - tweets[j][likes]) < threshold: #if they are equally funny
                    continue #skip
                else:  #one is funnier than the other
                    pairsCol = pairsCol + 1
                    if tweets[j][likes] > tweets[i][likes]:
                        if tweets[j][feature] < 0:
                            negCountCol = negCountCol + 1
                    else:
                        if tweets[i][feature] < 0:
                            negCountCol = negCountCol + 1
                    
        print "\t\tand {} pairs (excluding ties)".format(pairsCol)
        print "\t\t{} pairs had a funnier tweet with neg difference, that's {}%".format(negCountCol, float(negCountCol)/pairsCol*100)
        
        pairs = pairs + pairsCol
        negCount = negCount + negCountCol
        tweetCount = tweetCount + tweetCountCol
    
    print "Total"
    print "\t{} tweets".format(tweetCount)
    print "\t{} pairs (excluding ties)".format(pairs)
    print "\t{} pairs had a funnier tweet with neg difference, that's {}%".format(negCount, float(negCount)/pairs*100)
        
            
            
            