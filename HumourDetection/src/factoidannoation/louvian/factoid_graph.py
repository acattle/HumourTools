'''
Created on 18 Jul 2016

@author: andrew
'''
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.util import pad_sequence, trigrams
from pymongo import MongoClient
from math import sqrt
from igraph import *
import re

LEFT_PAD_SYMBOL = (u"<s>", u"<s>")
RIGHT_PAD_SYMBOL = (u"</s>", u"</s>")
POS_PREFIX = u"POS_"
TRIGRAM_PREFIX = u"TRIGRAM_"

class LexicalSimilarityGraph:
    '''
    Lexical Similarity Graph for use in Louvian Factoid annotation method described
    in http://www.aclweb.org/anthology/P13-2045
    
    Each node corresponds to a word in the lexicon and each edge weight is
    proportional to the cosine similarity between the two nodes.
    '''


    def __init__(self, texts):
        '''
        Constructor which takes in a list of texts to be processed. For each word
        in the lexicon, Context Features, Context POS Tag Features, and Spelling
        Features are extracted and cosine similarities are used as edge weights
        '''
        
        wordVectors = {}
        
        #Process context and context POS features
        for text in texts:
            sentences = sent_tokenize(text)
            for sentence in sentences:
                words = word_tokenize(sentence)
                wordsAndPOS = pos_tag(words) 
                
                #pad left and right. We use a 5 word window, meaning we need n=3 since the farthest distance is 3 words (inclusive)
                #convertion to list to allow access by index
                wordsAndPOS = list(pad_sequence(wordsAndPOS, 3, pad_left=True, pad_right=True, left_pad_symbol=LEFT_PAD_SYMBOL, right_pad_symbol=RIGHT_PAD_SYMBOL))
                
                
                for i in range(2, len(words) - 2): #to avoid processing <s> and </s> padding
                    word, pos = wordsAndPOS[i]
                    word = word.lower() #convert word to lowercase
                    pos = u"{}{}".format(POS_PREFIX, pos) #escape POS tags to avoid confusion with words
                    
                    if word not in wordVectors: #if the word hasn't been initialize
                        wordVectors[word] = {} #initialize to a blank dictionary
                    
                    wordVec = wordVectors[word]
                    
                    for offset in [-2, -1, 1, 2]: #5 word window, we don't care about the word itself (offset 0)
                        windowWord, windowPOS = wordsAndPOS[i + offset]
                        windowWord = windowWord.lower()
                        windowPOS = u"{}{}".format(POS_PREFIX, windowPOS)
                        
                        wordVec[windowWord] = wordVec.get(windowWord, 0) + 1 #using get() avoids KeyError
                        wordVec[windowPOS] = wordVec.get(windowPOS, 0) + 1
        
        #Process spelling features
        for word in wordVectors:
            for trigram in trigrams(word):
                trigram = u"{}{}{}{}".format(TRIGRAM_PREFIX, trigram[0], trigram[1], trigram[2])
                
                wordVectors[word][trigram] = wordVectors[word].get(trigram, 0) + 1
            
            #Normalize vector length for easier cosine similarity later
            vectorLength = 0
            for value in wordVectors[word].itervalues():
                vectorLength = vectorLength + value**2
            vectorLength = sqrt(vectorLength)
            for key in wordVectors[word]:
                wordVectors[word][key] = float(wordVectors[word][key])/vectorLength
                
        
        #create graph
        words = wordVectors.keys()
        self.graph = Graph.Full(len(words)) #create a fully connected graph with as many vertices as words in the lexicon
        #Adding the edges individually causes the graph to be reindexed for each edge.
        #Creating all the edges at once is significantly faster
        self.graph.vs["name"] = words #set the names of the vertices to the words in the lexicon
        self.graph.es["weight"] = None #make graph weighted
        
        for i in range(0, len(words)-1):
            print "{}/{}".format (i, len(words)-2)
            wordVecI = wordVectors[words[i]]
            
            for j in range(i+1, len(words)):
                wordVecJ = wordVectors[words[j]]
                
                cosineSim = sum(wordVecI[k] * wordVecJ.get(k, 0) for k in wordVecI)
                self.graph[words[i], words[j]] = cosineSim #assign weight
                
                    
if __name__ == "__main__":
#     client = MongoClient()
#     gentlerSongs = client.tweets.GentlerSongs.find()
#     
#     texts = []
#     for tweet in gentlerSongs:
#         text = tweet["text"]
#         text = re.sub("@midnight", "", text, flags=re.I)
#         text = re.sub("#GentlerSongs", "", text, flags=re.I)
#         texts.append(text)
#     
#     lsg = LexicalSimilarityGraph(texts)
    graph = Graph.Read("gentlersongs.pickle", "pickle")
    communities = graph.community_multilevel(weights="weight")
    plot(communities, "communities.png")
    
    
#     lsg.graph.write("gentlersongs.graphml", "graphml")
#     lsg.graph.write("gentlersongs.pickle", "pickle")
        