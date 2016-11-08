'''
Created on Sep 18, 2016

@author: Andrew
'''
import kenlm
from pymongo import MongoClient
import re

class KenLMModel():
    def __init__(self, modelLoc):
        self.model = kenlm.Model(modelLoc)

    def perplexity(self, sentence):
        """
        Compute perplexity of a sentence.
        @param sentence One full sentence to score.  Do not include <s> or </s>.
        """
        words = len(sentence.split()) + 1 # For </s>
        return 10.0**(-self.model.score(sentence) / words)
        
if __name__ == "__main__":
#     km = KenLMModel("/mnt/c/Users/Andrew/Desktop/kenlm models and raw text/wiki_pos_w_punc_4gram_prune.arpa")
    km = KenLMModel("/mnt/c/Users/Andrew/Desktop/Senti140/tokens_2.arpa")
    
    client = MongoClient()
    
    atMentions = re.compile(ur"@\w+", flags=re.I|re.U)
    atMidnight = re.compile(u"@midnight", flags=re.I|re.U)
    hashtag = re.compile(ur"#\w+", flags=re.I|re.U)
        
    likes = "total likes"
    feature = "ark token"
    target = "ark perplexity 2 no user or hash"
    cols = ["GentlerSongs", "OlympicSongs", "OceanMovies", "BoringBlockbusters"]
    
    for col in cols:
        tweets = []
        for tweet in client.tweets[col].find({"$and":[{likes : {"$gte" : 7}}, {feature : {"$exists" : True}}]}):
            mentions = atMentions.findall(tweet["text"])
            if len(mentions) > 1: #if more than 1 person is mentione
                continue
            elif len(mentions) == 1:
                if not atMidnight.match(mentions[0]): #if the mention someone other than @midngiht
                    continue
            if len(hashtag.findall(tweet["text"])) > 1: #if there's more than 1 hashtag
                continue
            
            pos_tags = tweet["ark pos"].split()
            tokens = tweet[feature].split()
            
            for i in xrange(len(pos_tags)-1, -1, -1):
                if pos_tags[i] in ["#", "@", "U"]:
                    del tokens[i]
            
            tweet["text"] = "".join(tokens)
                    
            
            tweets.append(tweet)
            
            
        
        for tweet in tweets:
            perplex = km.perplexity(tweet[feature])
            tweet[target] = perplex
            client.tweets[col].update({"_id" : tweet["_id"]}, {"$set" : {target : tweet[target]}})
            