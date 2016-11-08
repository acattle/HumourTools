'''
Created on Sep 20, 2016

@author: Andrew
'''
from pymongo import MongoClient
from scipy.stats import wilcoxon
from statsmodels.sandbox.stats.multicomp import multipletests
import re

client = MongoClient()

atMentions = re.compile(ur"@\w+", flags=re.I|re.U)
atMidnight = re.compile(u"@midnight", flags=re.I|re.U)
hashtag = re.compile(ur"#\w+", flags=re.I|re.U)

likes = "total likes"
features = ["perplexity 2", "perplexity 3", "perplexity 4", "pos perplexity 2", "pos perplexity 3", "pos perplexity 4", "ark perplexity 2 no user or hash", "ark perplexity 3 no user or hash", "ark perplexity 4 no user or hash", "ark pos perplexity 2", "ark pos perplexity 3", "ark pos perplexity 4", "usf fwa forward most", "usf fwa forward least", "usf fwa forward average", "usf fwa backward most", "usf fwa backward least", "usf fwa backward average", "usf fwa difference most", "usf fwa difference least", "usf fwa difference average", "ngd closest", "ngd furthest", "ngd average", "w2v most", "w2v least", "w2v average", "w2v most tweet 100", "w2v least tweet 100", "w2v average tweet 100", "w2v most tweet 100 2", "w2v least tweet 100 2", "w2v average tweet 100 2", "w2v most tweet 400", "w2v least tweet 400", "w2v average tweet 400"]
cols = ["GentlerSongs", "OlympicSongs", "OceanMovies", "BoringBlockbusters"]
minLikeDiff = 1 #there should be at least this much difference between pairs

print "Only considering pairs with at least {} likes difference".format(minLikeDiff)
p_values = []
for feature in features:
    print "Testing {}".format(feature)
    lessMoreDiff = [] #holds difference in feature value for less funny - more funny
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
            if len(mentions) > 1: #if more than 1 person is mentioned
                continue
            elif len(mentions) == 1:
                if not atMidnight.match(mentions[0]): #if the mention someone other than @midngiht
                    continue
            if len(hashtag.findall(tweet["text"])) > 1: #if there's more than 1 hashtag
                continue
            
            if tweet[feature] > 0:
                tweets.append(tweet)
        
#         print "\t{} has {} valid readings".format(col, len(tweets))
        
        lessMoreDiffCol = []
        for i in range(len(tweets)):
            for j in range(i+1, len(tweets)):
                likeDiff = tweets[i][likes] - tweets[j][likes]
                if abs(likeDiff) < minLikeDiff: #if they are equally funny
                    continue #skip
                elif likeDiff < 0: #if j is funnier
                    lessMoreDiffCol.append(tweets[i][feature] - tweets[j][feature])
                elif likeDiff > 0:  #i must be funnier than j
                    lessMoreDiffCol.append(tweets[j][feature] - tweets[i][feature])
                else: #this should never happen
                    raise Exception("Threshold is less than 1: {}".format(minLikeDiff))
                    
        numPairsCol = len(lessMoreDiffCol)
#         print "\t\tand {} pairs (excluding ties)".format(numPairsCol)
        
        if numPairsCol > 0:
            #TODO: Should I exclude differences of 0 from this test?
            #https://stackoverflow.com/questions/15759827/how-can-i-count-occurrences-of-elements-that-are-bigger-than-a-given-number-in-a
            higherNumCol = sum(i < 0 for i in lessMoreDiffCol) #if the funnier entry has a higher feature value we expect the difference to be negative
#             print "\t\t{} higher is funnier".format(float(higherNumCol)/numPairsCol)
            
            tCol, pCol = wilcoxon(lessMoreDiffCol)
#             print "\t\tT = {}".format(tCol)
#             print "\t\tp = {}".format(pCol)
#             print "\n"
            
            lessMoreDiff.extend(lessMoreDiffCol) #add to the total number
    
#     print "\tTotal"
    numPairs = len(lessMoreDiff)
#     print "\t{} pairs (excluding ties)".format(numPairs)
    
    if numPairs > 0:
        #TODO: Should I exclude differences of 0 from this test?
        #https://stackoverflow.com/questions/15759827/how-can-i-count-occurrences-of-elements-that-are-bigger-than-a-given-number-in-a
        higherNum = sum(i < 0 for i in lessMoreDiff)
        print "\t{} higher is funnier".format(float(higherNum)/numPairs)
        
        t, p = wilcoxon(lessMoreDiff)
#         print "\tT = {}".format(t)
#         print "\tp = {}".format(p)
        print "\n"
        
        p_values.append(p)
        
rej95, p_adjust95, _, _ = multipletests(p_values, alpha=0.05, method="holm")

for i in range(len(rej95)):
    print "{}\t{}\t{}".format(features[i], p_adjust95[i], rej95[i])
    
rej995, p_adjust995, _, _ = multipletests(p_values, alpha=0.005, method="holm")
print "\n"
for i in range(len(rej995)):
    print "{}\t{}\t{}".format(features[i], p_adjust995[i], rej995[i])