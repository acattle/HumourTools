'''
Created on Sep 20, 2016

@author: Andrew
'''
from pymongo import MongoClient
import re
from nltk import word_tokenize, sent_tokenize
from nltk.tag.perceptron import PerceptronTagger

client = MongoClient()

atMentions = re.compile(ur"@\w+", flags=re.I|re.U)
atMidnight = re.compile(u"@midnight", flags=re.I|re.U)
hashtag = re.compile(ur"#\w+", flags=re.I|re.U)
    
likes = "total likes"
cols = ["GentlerSongs", "OlympicSongs", "OceanMovies", "BoringBlockbusters"]
tagger = PerceptronTagger()

for col in cols:
    tweets = []
    for tweet in client.tweets[col].find({likes : {"$gte" : 7}}):
        mentions = atMentions.findall(tweet["text"])
        if len(mentions) > 1: #if more than 1 person is mentione
            continue
        elif len(mentions) == 1:
            if not atMidnight.match(mentions[0]): #if the mention someone other than @midngiht
                continue
        if len(hashtag.findall(tweet["text"])) > 1: #if there's more than 1 hashtag
            continue
        tweets.append(tweet)
    
    for tweet in tweets:
        text = tweet["text"]
        pos_tags = []
        for sent in sent_tokenize(text):
            words = word_tokenize(sent)
            
            for word, pos in tagger.tag(words):
                pos_tags.append(pos)
        
        pos_str = u" ".join(pos_tags)
        tweet["text pos"] = pos_str
        client.tweets[col].update({"_id" : tweet["_id"]}, tweet)
            
            