'''
Created on Jan 22, 2017

@author: Andrew
'''
from gensim.models import TfidfModel, LdaModel, LsiModel
from gensim.corpora import Dictionary
from numpy import zeros, isnan, mean
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
import os
import glob
import re
from time import strftime
import codecs

class GensimDocSummarization(object):
    def __init__(self, word_ids_loc, tfidf_model_loc, dimensions=300):
        self.id2word = Dictionary.load_from_text(word_ids_loc)
        self.tfidf = TfidfModel.load(tfidf_model_loc)
        self.dimensions = dimensions
        self.model=None
    
    def _convert_to_numpy_array(self, tuples):
        #initialize vector to all 0s
        vec = zeros(self.dimensions)
        
        for i, val in tuples:
            vec[i] = val
        
        return vec
    
    def get_vector(self,document):
        #get bag of words features for document
        tokens = document.split()
        bow = self.id2word.doc2bow(tokens)
        tfidf_tokens = self.tfidf[bow]
        tuples = self.model[tfidf_tokens]
        return self._convert_to_numpy_array(tuples)
    
    def get_similarity(self, word1, word2):
        return 1-cosine(self.get_vector(word1), self.get_vector(word2)) #since it's cosine distance
    
    def get_relative_entropy(self, word1, word2):
        return entropy(self.get_vector(word1), self.get_vector(word2))

class LDAVectorizer(GensimDocSummarization):
    def __init__(self, lda_fileloc, word_ids_loc, tfidf_model_loc):
        super(LDAVectorizer, self).__init__(word_ids_loc, tfidf_model_loc)
        self.model = LdaModel.load(lda_fileloc)

class LSIVectorizer(GensimDocSummarization):
    def __init__(self, lsi_fileloc, word_ids_loc, tfidf_model_loc):
        super(LSIVectorizer, self).__init__(word_ids_loc, tfidf_model_loc)
        self.model = LsiModel.load(lsi_fileloc)

if __name__ == "__main__":
#     lsi = LSIVectorizer(r"C:\Users\Andrew\Desktop\vectors\lda_prep_no_lemma\no_lemma.lsi", r"C:\Users\Andrew\Desktop\vectors\lda_prep_no_lemma\lda_no_lemma_wordids.txt.bz2", r"C:\Users\Andrew\Desktop\vectors\lda_prep_no_lemma\lda_no_lemma.tfidf_model")
#     print lsi.get_vector("The quick brown fox")
#     
    lda = LDAVectorizer(r"C:\Users\Andrew\Desktop\vectors\lda_prep_no_lemma\no_lemma.lda", r"C:\Users\Andrew\Desktop\vectors\lda_prep_no_lemma\lda_no_lemma_wordids.txt.bz2", r"C:\Users\Andrew\Desktop\vectors\lda_prep_no_lemma\lda_no_lemma.tfidf_model")
#     print lda.get_vector("The quick brown fox")
    
#     # load id->word mapping (the dictionary),
#     id2word = gensim.corpora.Dictionary.load_from_text(bz2.BZ2File('lda_no_lemma_wordids.txt.bz2'))
#     # load tfidf model
#     tfidf = gensim.models.TfidfModel.load('lda_no_lemma.tfidf_model')
#     #load LSI model
#     lsi = gensim.models.lsimodel.LSIModel.load("no_lemma.lsi")
#     
#     vector = lsi[tfidf[id2word.doc2bow(["apple"])]] #as [(topic_id, topic_score)]
    english_stopwords = stopwords.words("english")
    pos_to_ignore = ["D","P","X","Y", "T", "&", "~", ",", "!", "U", "E"]
    semeval_dir = r"c:/Users/Andrew/Desktop/SemEval Data"
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
            
            print "{}\tprocessing {}".format(strftime("%y-%m-%d_%H:%M:%S"),name)
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
            lda_fileloc = "{}.lda_sim".format(f)
            if os.path.isfile(lda_fileloc): #if the file exists
                with codecs.open(lda_fileloc, "r", encoding="utf-8") as resume_file:
                    header = resume_file.readline().strip()
                    if header.split() == hashtag_words: #only if the header matches what we've extracted
                        for line in resume_file:
                            line_split = line.split("\t")
                            if len(line_split) != (len(hashtag_words) +2): #if we don't have enough columns
                                print u"ERROR - previously collected tweet is incomplet: {}".format(line)
                                continue
                            
                            tweet_id = line_split[0]
                            min_val = line_split[1].split()[0]
                            
                            if min_val == "0":
                                print u"Tweet {} has a 0 value. Will retry".format(tweet_id)
                                continue
                            
                            already_collected.add(tweet_id)
                            lines_to_rewrite.append(line)
            
            done = 0
            with codecs.open(lda_fileloc, "w", encoding="utf-8") as out_file:
                out_file.write(u"{}\n".format(u" ".join(hashtag_words)))
                for line in lines_to_rewrite:
                    out_file.write(line)
                
                for tokens, pos, tweet_id in zip(tweet_tokens,tweet_pos, tweet_ids):
                    if tweet_id in already_collected: #skip it if we already have a valid reading
                        done+=1
                        continue
                    
                    lda_results_by_word = []
                    for word in hashtag_words:
                        ldas_by_hashtag_word=[]
                        for token, tag in zip(tokens, pos):
                            token=token.lower()
                            if (tag in pos_to_ignore) or (token in english_stopwords):
                                continue
                            if (token == "@midnight") or (token == hashtag): #if it's the @midnight account of the game's hashtag
                                    continue #we don't want to process it
                            
                            lda_val = lda.get_similarity(word, token)
                            ldas_by_hashtag_word.append(lda_val)
                            
                        if len(ldas_by_hashtag_word) == 0:
                            print u"ERRORL no valid tokens\t{}".format(u" ".join(tokens))
                            ldas_by_hashtag_word = [0]
                        
                        
                        lda_results_by_word.append((min(ldas_by_hashtag_word), mean(ldas_by_hashtag_word), max(ldas_by_hashtag_word)))
                    
                    mins, avgs, maxes = zip(*lda_results_by_word) #separate out the columns
                    
                    overall = (min(mins), mean(avgs), max(maxes))
                
                    per_word_ldas = u"\t".join([u"{} {} {}".format(*res) for res in lda_results_by_word])
                    overall_ldas = u"{} {} {}".format(*overall)
                    line = u"{}\t{}\t{}\n".format(tweet_id, overall_ldas, per_word_ldas) 
                    out_file.write(line)
                    done+=1
                    if done % 20 == 0:
                        print "{}\t{}\t{} of {} completed".format(strftime("%y-%m-%d_%H:%M:%S"), name, done, len(tweet_ids))
            print "{}\tfinished {}".format(strftime("%y-%m-%d_%H:%M:%S"),name)