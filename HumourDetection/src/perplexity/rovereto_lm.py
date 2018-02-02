'''
Created on Jan 28, 2017

@author: Andrew
'''
from __future__ import print_function, division
import codecs
from math import log10
import os
import glob
import re
from time import strftime
from nltk.corpus import stopwords

#TODO: use an NLTK FreqDist instead of python dict? Then pass that the KneserNeyProbDist? http://www.nltk.org/_modules/nltk/probability.html

def convert_from_rovereto_format(fileloc,outputloc):
    with codecs.open(fileloc, "r", encoding="utf-8") as ngram_file, codecs.open(outputloc, "w", encoding="utf-8") as outfile:
        for line in ngram_file:
            line = line.strip()
            line_split = line.split("\t")
            
            ngram=line_split[0]
            
            count = 0
            #https://web.archive.org/web/20160501211838/http://clic.cimec.unitn.it/amac/twitter_ngram/header-line.txt
            #for each day of the week, hour, and gender they list the frequency and the number of users who used it. We only care about frequencies
            for i in range(1, len(line_split), 2):
                count += int(line_split[i])
            
            outfile.write(u"{}\t{}\n".format(ngram, count))





class RoveretoLM(object):
    def __init__(self, unigram_loc, bigram_loc=None):
        self.unigram_counts = self._read_file(unigram_loc)
        self.unigram_total = 0
        for unigram in self.unigram_counts:
            self.unigram_total+=self.unigram_counts[unigram]
        self.bigram_counts = None
        if bigram_loc:
            self.bigram_counts = self._read_file(bigram_loc)
    
    def _read_file(self,ngram_loc):
        ngram_counts = {}
        with codecs.open(ngram_loc, "r", encoding="utf-8") as ngram_file:
            for line in ngram_file:
                ngram, count = line.split("\t")
                count = int(count)
                ngram_counts[ngram] = count
                
        return ngram_counts
    
    def _unigram_log_prob(self, unigram, smoothing=True):
        count = self.unigram_counts.get(unigram, 0)
        total = self.unigram_total
        if smoothing:
            count +=1
            total += len(self.unigram_counts)
        
        return log10(count) - log10(total)
    
    def _bigram_log_prob(self, bigram, smoothing=True):
        b_count = self.bigram_counts.get(bigram,0)
        u_count = self.unigram_counts.get(bigram.split()[0], 0)
        if smoothing:
            b_count+=1
            u_count+=len(self.bigram_counts)
        
        return log10(b_count) - log10(u_count)
    
    def get_unigram_log_prob(self,document, smoothing=True):
        log_prob = 0.0
        for word in document.split():
            log_prob += self._unigram_log_prob(word,smoothing=smoothing)
        
        return log_prob
    
    def get_bigram_log_prob(self,document, smoothing=True):
        log_prob = 0.0
        words = document.split()
        if len(words) > 0:
            log_prob=self._unigram_log_prob(words[0], smoothing)
            for i in range(1,len(words)):
                bigram = u"{} {}".format(words[i-1],words[i])
                log_prob += self._bigram_log_prob(bigram, smoothing)
        
        return log_prob
    
    def get_unigram_perplexity(self,document,smoothing=True):
        log_prob = self.get_unigram_log_prob(document, smoothing)
        return 10 ** (-log_prob/len(document.split()))
    
    def get_bigram_perplexity(self,document,smoothing=True):
        log_prob = self.get_bigram_log_prob(document, smoothing)
        return 10 ** (-log_prob/len(document.split()))

if __name__ == "__main__":
#     convert_from_rovereto_format("/home/acattle/vectors/en.1grams", "/home/acattle/vectors/en.1grams.counts")
#     lm_dir = "/home/acattle/vectors"
    lm_dir = "c:/users/andrew/desktop/vectors"
    unigram_loc = "en.1grams.counts"
    bigram_loc = "en.2grams.counts"
    model = RoveretoLM(os.path.join(lm_dir, unigram_loc), os.path.join(lm_dir,bigram_loc))
    semeval_dir = r"c:/users/andrew/desktop/SemEval Data"
    dirs = [r"trial_dir/trial_data",
            r"train_dir/train_data",
            r"evaluation_dir/evaluation_data"]
    tagged_dir = "tagged"
    english_stopwords = stopwords.words("english")
     
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
            with codecs.open(f, "r", encoding="utf-8") as tweet_file, codecs.open("{}.perplexity".format(f), "w", encoding="utf-8") as out_file:
                #get the perplexity for the hashtag words. Might be useful in finding setup later
                hash_perplex = []
                for word in hashtag_words:
                    hash_perplex.append(model.get_unigram_perplexity(word))
                out_file.write(u"{}\t{}\n".format(u" ".join(hashtag_words), u" ".join([str(perp) for perp in hash_perplex])))
                
                for line in tweet_file:
                    line=line.strip()
                    if line == "":
                        continue
                    line_split = line.split("\t")
                    tokens = line_split[0].split()
                    tweet_id = line_split[3]
                    
                    tokens_sanitized = []
                    for token in tokens:
                        token = token.lower()
                        if (token == "@midnight") or (token == hashtag):
                            continue
                        tokens_sanitized.append(token)
                    
                    #get the per word perplexity. May be useful later for finding punchlines
                    unigram_perp = []
                    for token in tokens_sanitized:
                        unigram_perp.append(model.get_unigram_perplexity(token))
                    
                    tweet_sanitized = u" ".join(tokens_sanitized)
                    if len(tokens_sanitized) > 0:
                        uni_perp = model.get_unigram_perplexity(tweet_sanitized)
                        bi_perp = model.get_bigram_perplexity(tweet_sanitized)
                    else:
                        uni_perp = None
                        bi_perp = None
                    
                    out_file.write(u"{}\t{}\t{}\t{}\t{}\n".format(tweet_id, uni_perp, bi_perp, tweet_sanitized, u" ".join([str(perp) for perp in unigram_perp])))
                    
            print("{}\tfinished {}".format(strftime("%y-%m-%d_%H:%M:%S"),name))