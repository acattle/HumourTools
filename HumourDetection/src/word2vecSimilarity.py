'''
Created on Jun 8, 2016

@author: Andrew
'''
from gensim.models import Word2Vec
from pymongo import MongoClient
from numpy import mean
from util.model_wrappers.common_models import get_google_word2vec
import os
import glob
import re
from nltk.corpus import stopwords
import codecs
from time import strftime

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
#     model = Word2VecModel("C:\\Users\\Andrew\\Desktop\\vectors\\GoogleNews-vectors-negative300.bin")
    model = get_google_word2vec()
#     
#     print "W2V(ROW) = {}".format(model.getSimilarity("olympics", "row"))
#     print "W2V(SAIL) = {}".format(model.getSimilarity("olympics", "sail"))
     
     
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
                    w2vScore = model.get_similarity(setup, word)
                    if w2vScore == None: #undefined similarity
                        continue
                    if w2vScore > most[0]:
                        most = [w2vScore, word]
                    if w2vScore < least[0]:
                        least = [w2vScore, word] 
                    allVals.append(w2vScore)
                 
                if len(allVals) > 0: #if at least one word is valid
                    tweet["w2v most tweet google"] = most[0]
                    tweet["w2v most word tweet google"] = most[1]
                    tweet["w2v least tweet google"] = least[0]
                    tweet["w2v least word tweet google"] = least[1]
                    tweet["w2v average tweet google"] = mean(allVals)
                    client.tweets[col].update({"_id" : tweet["_id"]}, tweet)
             
            else:
                pass
             
            count += 1
            print("{} of {} done".format(count, total))





#     english_stopwords = stopwords.words("english")
#     pos_to_ignore = ["D","P","X","Y", "T", "&", "~", ",", "!", "U", "E"]
#     model = GoogleWord2Vec("C:/Users/Andrew/Desktop/vectors/GoogleNews-vectors-negative300.bin")
#     semeval_dir = r"C:/Users/Andrew/Desktop/SemEval Data"
#     dirs = [r"trial_dir/trial_data",
#             r"train_dir/train_data",
#             r"evaluation_dir/evaluation_data"]
#     tagged_dir = "tagged"
#      
#     for d in dirs:
#         os.chdir(os.path.join(semeval_dir, d, tagged_dir))
#         for f in glob.glob("*.tsv"):
#             name = os.path.splitext(os.path.basename(f))[0]
#             hashtag = "#{}".format(re.sub("_", "", name.lower()))
#             hashtag_words = name.split("_")        
#             #remove swords that don't give you some idea of the domain
#             hashtag_words = [word.lower() for word in hashtag_words if word.lower() not in english_stopwords]
#             #the next 3 are to catch "<blank>In#Words" type hashtags
#             hashtag_words = [word for word in hashtag_words if word != "in"]
#             hashtag_words = [word for word in hashtag_words if not ((len(word) == 1) and (word.isdigit()))]
#             hashtag_words = [word for word in hashtag_words if word != "words"]
#             
#             print "{}\tprocessing {}".format(strftime("%y-%m-%d_%H:%M:%S"),name)
#             tweet_ids = []
#             tweet_tokens = []
#             tweet_pos = []
#             with codecs.open(f, "r", encoding="utf-8") as tweet_file:
#                 for line in tweet_file:
#                     line=line.strip()
#                     if line == "":
#                         continue
#                     line_split = line.split("\t")
#                     tweet_tokens.append(line_split[0].split())
#                     tweet_pos.append(line_split[1].split())
#                     tweet_ids.append(line_split[3])
#             
#             w2v_fileloc = u"{}.w2v_kl"
#             done = 0
#             with codecs.open(w2v_fileloc, "w", encoding="utf-8") as out_file:
#                 for tokens, pos, tweet_id in zip(tweet_tokens,tweet_pos, tweet_ids):
#                     w2v_results_by_word = []
#                     for word in hashtag_words:
#                         w2vs_by_hashtag_word=[]
#                         for token, tag in zip(tokens, pos):
#                             token=token.lower()
#                             if (tag in pos_to_ignore) or (token in english_stopwords):
#                                 continue
#                             if (token == "@midnight") or (token == hashtag): #if it's the @midnight account of the game's hashtag
#                                     continue #we don't want to process it
#                             
#                             w2v = model.get_relative_entropy(word, token)
#                             w2vs_by_hashtag_word.append(w2v)
#                             
#                         if len(w2vs_by_hashtag_word) == 0:
#                             print u"ERRORL no valid tokens\t{}".format(u" ".join(tokens))
#                             w2vs_by_hashtag_word = [0]
#                         
#                         
#                         w2v_results_by_word.append((min(w2vs_by_hashtag_word), mean(w2vs_by_hashtag_word), max(w2vs_by_hashtag_word)))
#                     
#                     mins, avgs, maxes = zip(*w2v_results_by_word) #separate out the columns
#                     
#                     overall = (min(mins), mean(avgs), max(maxes))
#                 
#                     per_word_w2vs = u"\t".join([u"{} {} {}".format(*res) for res in w2v_results_by_word])
#                     overall_w2vs = u"{} {} {}".format(*overall)
#                     line = u"{}\t{}\t{}\n".format(tweet_id, overall_w2vs, per_word_w2vs) 
#                     out_file.write(line)
#                     done+=1
#                     if done % 20 == 0:
#                         print "{}\t{}\t{} of {} completed".format(strftime("%y-%m-%d_%H:%M:%S"), name, done, len(tweet_ids))
#             print "{}\tfinished {}".format(strftime("%y-%m-%d_%H:%M:%S"),name)