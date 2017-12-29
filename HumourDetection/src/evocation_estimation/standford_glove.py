'''
Created on Jan 25, 2017

@author: Andrew
'''

from __future__ import print_function #for Python 2.7 compatibility
from glove import Glove
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
import os
import glob
import re
from time import strftime
import codecs
from numpy import mean

class StanfordGlove(object):
    def __init__(self, vector_loc):
        #https://github.com/maciejkula/glove-python/pull/24/commits/01b3e983f1548aaca7be7bc52062b1566e0df7eb
        self.model = Glove.load_stanford(vector_loc)
    
    def get_vector(self, word):
        return self.model.word_vectors[self.model.dictionary[word]]
    
    def get_similarity(self, word1, word2):
        return 1-cosine(self.get_vector(word1), self.get_vector(word2)) #since it's cosine distance
    
    def get_relative_entropy(self, word1, word2):
        return entropy(self.get_vector(word1), self.get_vector(word2))

if __name__ == "__main__":
    english_stopwords = stopwords.words("english")
    pos_to_ignore = ["D","P","X","Y", "T", "&", "~", ",", "!", "U", "E"]
    model = StanfordGlove("/mnt/c/Users/Andrew/Desktop/vectors/glove.840B.300d.txt")
    print("model loaded")
    semeval_dir = r"/mnt/c/Users/Andrew/Desktop/SemEval Data"
    dirs = [r"trial_dir/trial_data",
            r"train_dir/train_data",
            r"evaluation_dir/evaluation_data"]
    tagged_dir = "tagged"
     
    for d in dirs:
        os.chdir(os.path.join(semeval_dir, d, tagged_dir))
        for f in glob.glob("*.tsv"):
            name = os.path.splitext(os.path.basename(f))[0]
            hashtag = "#{}".format(re.sub("_", "", name.lower()))
            hashtag_words = name.split("_")        
            #remove swords that don't give you some idea of the domain
            hashtag_words = [word.lower() for word in hashtag_words if word.lower() not in english_stopwords]
            #the next 3 are to catch "<blank>In#Words" type hashtags
            hashtag_words = [word for word in hashtag_words if word != "in"]
            hashtag_words = [word for word in hashtag_words if not ((len(word) == 1) and (word.isdigit()))]
            hashtag_words = [word for word in hashtag_words if word != "words"]
            
            print("{}\tprocessing {}".format(strftime("%y-%m-%d_%H:%M:%S"),name))
            tweet_ids = []
            tweet_tokens = []
            tweet_pos = []
            with codecs.open(f, "r", encoding="utf-8") as tweet_file:
                for line in tweet_file:
                    line=line.strip()
                    if line == "":
                        continue
                    line_split = line.split("\t")
                    tweet_tokens.append(line_split[0].split())
                    tweet_pos.append(line_split[1].split())
                    tweet_ids.append(line_split[3])
            
            already_collected = set()
            lines_to_rewrite = []
            #check if file exists to avoid redoing a lot of effort
            glove_fileloc = "{}.glove_kl".format(f)
            if os.path.isfile(glove_fileloc): #if the file exists
                with codecs.open(glove_fileloc, "r", encoding="utf-8") as resume_file:
                    header = resume_file.readline().strip()
                    if header.split() == hashtag_words: #only if the header matches what we've extracted
                        for line in resume_file:
                            line_split = line.split("\t")
                            if len(line_split) != (len(hashtag_words) +2): #if we don't have enough columns
                                print(u"ERROR - previously collected tweet is incomplet: {}".format(line))
                                continue
                            
                            tweet_id = line_split[0]
                            min_val = line_split[1].split()[0]
                            
                            if min_val == "0":
                                print(u"Tweet {} has a 0 value. Will retry".format(tweet_id))
                                continue
                            
                            already_collected.add(tweet_id)
                            lines_to_rewrite.append(line)
            
            done = 0
            with codecs.open(glove_fileloc, "w", encoding="utf-8") as out_file:
                out_file.write(u"{}\n".format(u" ".join(hashtag_words)))
                for line in lines_to_rewrite:
                    out_file.write(line)
                
                for tokens, pos, tweet_id in zip(tweet_tokens,tweet_pos, tweet_ids):
                    if tweet_id in already_collected: #skip it if we already have a valid reading
                        done+=1
                        continue
                    
                    glove_results_by_word = []
                    for word in hashtag_words:
                        gloves_by_hashtag_word=[]
                        for token, tag in zip(tokens, pos):
                            token=token.lower()
                            if (tag in pos_to_ignore) or (token in english_stopwords):
                                continue
                            if (token == "@midnight") or (token == hashtag): #if it's the @midnight account of the game's hashtag
                                    continue #we don't want to process it
                            
                            try:
                                glove_val = model.get_relative_entropy(word, token)
                            except KeyError:
                                print(u"Can't get vector for {} or {}".format(word, token))
                                glove_val = float("inf")
                            gloves_by_hashtag_word.append(glove_val)
                            
                        if len(gloves_by_hashtag_word) == 0:
                            print(u"ERRORL no valid tokens\t{}".format(u" ".join(tokens)))
                            gloves_by_hashtag_word = [0]
                        
                        
                        glove_results_by_word.append((min(gloves_by_hashtag_word), mean(gloves_by_hashtag_word), max(gloves_by_hashtag_word)))
                    
                    mins, avgs, maxes = zip(*glove_results_by_word) #separate out the columns
                    
                    overall = (min(mins), mean(avgs), max(maxes))
                
                    per_word_gloves = u"\t".join([u"{} {} {}".format(*res) for res in glove_results_by_word])
                    overall_gloves = u"{} {} {}".format(*overall)
                    line = u"{}\t{}\t{}\n".format(tweet_id, overall_gloves, per_word_gloves) 
                    out_file.write(line)
                    done+=1
                    if done % 20 == 0:
                        print("{}\t{}\t{} of {} completed".format(strftime("%y-%m-%d_%H:%M:%S"), name, done, len(tweet_ids)))
            print("{}\tfinished {}".format(strftime("%y-%m-%d_%H:%M:%S"),name))