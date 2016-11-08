'''
Created on Jun 8, 2016

@author: Andrew
'''
from gensim.models import Word2Vec
from pymongo import MongoClient
from numpy import mean

class Word2VecModel():
    def __init__(self, vectorLoc, binary=True):
        self.model = Word2Vec.load_word2vec_format(vectorLoc, binary=binary)
    
    def getSimilarity(self, word1, word2):
        sim = None
        try:
            sim= self.model.similarity(word1,word2)
        except KeyError:
            sim=None #one of the words doesn't exist. Fail silently with undefined
        return sim

if __name__ == '__main__':
#     model = Word2VecModel("wiki.en.text.vector")
    model = Word2VecModel("C:\\Users\\Andrew\\Desktop\\Senti140\\tokens_400.vector")
    client = MongoClient()
    
    cols = [("GentlerSongs", "gentle"), ("OlympicSongs", "olympic"), ("BoringBlockbusters", "boring"), ("OceanMovies", "ocean")]
    for col, setup in cols:
        tweets = client.tweets[col].find({"$and" : [{"punch words" : {"$exists" : True}}, {"w2v" : {"$exists" : False}}]})
        
        count = 0
        total = tweets.count()
        for tweet in tweets:
            most = (-float("inf"), "")
            least = (float("inf"), "")
            allVals = []
            
            if tweet["punch words"]:
                for word in tweet["punch words"]:
                    if word == "None":
                        continue
                    if not word:
                        continue
                    w2vScore = model.getSimilarity(setup, word)
                    if w2vScore == None: #undefined similarity
                        continue
                    if w2vScore > most[0]:
                        most = [w2vScore, word]
                    if w2vScore < least[0]:
                        least = [w2vScore, word] 
                    allVals.append(w2vScore)
                
                if len(allVals) > 0: #if at least one word is valid
                    tweet["w2v most tweet 400"] = most[0]
                    tweet["w2v most word tweet 400"] = most[1]
                    tweet["w2v least tweet 400"] = least[0]
                    tweet["w2v least word tweet 400"] = least[1]
                    tweet["w2v average tweet 400"] = mean(allVals)
                    client.tweets[col].update({"_id" : tweet["_id"]}, tweet)
            
            else:
                pass
            
            count += 1
            print "{} of {} done".format(count, total)