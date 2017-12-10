'''
Created on 18 Jul 2016

@author: andrew
'''
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.util import pad_sequence, trigrams
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from pymongo import MongoClient
from math import sqrt
import igraph
import re
from util.treebank_to_wordnet import get_wordnet_pos
import math
import codecs

LEFT_PAD_SYMBOL = (u"<s>", u"<s>")
RIGHT_PAD_SYMBOL = (u"</s>", u"</s>")
POS_PREFIX = u"POS_"
TRIGRAM_PREFIX = u"TRIGRAM_"
ALPHANUM_PAT = re.compile(r"\W+")

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
        
#         wnl = WordNetLemmatizer()
        porter = PorterStemmer()
        wordVectors = {}
        engStopWords = stopwords.words("english")
        
        #Process context and context POS features
        for text in texts:
            wordsAndPOSNoPunc = [] #reset per text, not per sentence
            sentences = sent_tokenize(text)
            for sentence in sentences:
                words = word_tokenize(sentence)
                wordsAndPOS = pos_tag(words)
                for word, pos in wordsAndPOS:
                    word = word.lower()
                    wordNoPunc = ALPHANUM_PAT.sub("", word)
                    if len(wordNoPunc) > 0:
                        if word not in engStopWords: #ignore stopwords
#                             wordLemma = wnl.lemmatize(wordNoPunc, pos=get_wordnet_pos(pos))
                            wordStem = porter.stem(word)
                            wordsAndPOSNoPunc.append((wordStem, pos))
                wordsAndPOS = wordsAndPOSNoPunc
                
                
                #pad left and right. We use a 5 word window, meaning we need n=3 since the farthest distance is 3 words (inclusive)
                #convertion to list to allow access by index
                wordsAndPOS = list(pad_sequence(wordsAndPOS, 3, pad_left=True, pad_right=True, left_pad_symbol=LEFT_PAD_SYMBOL, right_pad_symbol=RIGHT_PAD_SYMBOL))
                
                
                for i in range(2, len(wordsAndPOS) - 2): #to avoid processing <s> and </s> padding
                    word, pos = wordsAndPOS[i]
                    pos = u"{}{}".format(POS_PREFIX, pos) #escape POS tags to avoid confusion with words
                    
                    if word not in wordVectors: #if the word hasn't been initialize
                        wordVectors[word] = {} #initialize to a blank dictionary
                    
                    wordVec = wordVectors[word]
                    
                    for offset in [-2, -1, 1, 2]: #5 word window, we don't care about the word itself (offset 0)
                        windowWord, windowPOS = wordsAndPOS[i + offset]
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
        print "vectors calculated"
        words = wordVectors.keys()
        self.graph = igraph.Graph.Full(len(words)) #create a fully connected graph with as many vertices as words in the lexicon
        #Adding the edges individually causes the graph to be reindexed for each edge.
        #Creating all the edges at once is significantly faster
        self.graph.vs["name"] = words #set the names of the vertices to the words in the lexicon
#         self.graph.vs["label"] = words #set the labels for the graph accordingly
        self.graph.es["weight"] = None #make graph weighted
        toRemove = [] #the edges which will be removed
        
        for i in range(0, len(words)-1):
            print "{}/{}".format (i, len(words)-2)
            wordVecI = wordVectors[words[i]]
            
            for j in range(i+1, len(words)):
                wordVecJ = wordVectors[words[j]]
                
                cosineSim = sum(wordVecI[k] * wordVecJ.get(k, 0) for k in wordVecI)
                if cosineSim == 0.0: #there is no similarity between the two
                    toRemove.append((i, j)) #add the index pair to the list of edges to be removed (cuts down on reindexing time)
                else:
                    self.graph[words[i], words[j]] = cosineSim #assign weight
        
        print "removing edges"
        self.graph.delete_edges(toRemove)
        print "done"
                    
if __name__ == "__main__":
#     client = MongoClient()
#     gentlerSongs = client.tweets.GentlerSongs.find()
#      
#     texts = []
#     for tweet in gentlerSongs:
#         text = tweet["text"]
# #         text = re.sub("@midnight", "", text, flags=re.I)
#         text = re.sub(r"@\S*", "", text, flags=re.I)
# #         text = re.sub("#GentlerSongs", "", text, flags=re.I)
#         text = re.sub(r"#\S*", "", text, flags=re.I)
#         texts.append(text)
    texts = []
    with codecs.open("C:\\Users\\Andrew\\Desktop\\radev -caption corpus\\data\\445.data", "r", "utf-8") as hybridCarSubmissions:
        hybridCarSubmissions.readline() #we can ignore the first line
        for line in hybridCarSubmissions:
            caption = line.split("\t")[1] #get the second item in the tab delimited file
            texts.append(caption)
     
    lsg = LexicalSimilarityGraph(texts)
#     lsg.graph.write("hybridcar.Lemma.27.pickle", "pickle")
#     graph = igraph.Graph.Read("gentlersongs.pickle", "pickle")
#     print "pickle written"
    communities = lsg.graph.community_multilevel(weights=lsg.graph.es["weight"], return_levels=True)
    print "clustering done"
    with open("clusters.hybridcar.tfidf.nostopwords.stem.resetpercaption.txt", "w") as f:
        j=0
        for level in communities:
            f.write(u"Level {}\n".format(j))
            j+=1
            subgraphs = level.subgraphs()
            for i in range(len(subgraphs)):
                f.write(u"\tCluster {}\n".format(i).encode("utf-8"))
                for n in subgraphs[i].vs["name"]:
                    f.write(u"\t\t{}\n".format(n).encode("utf-8"))
                f.write(u"\n\n\n\n".encode("utf-8"))
    print "clusters written"
    
#     lsg.graph.write("gentlersongs.graphml", "graphml")
#     lsg.graph.write("gentlersongs.pickle", "pickle")
        