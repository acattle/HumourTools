'''
Created on 18 Jul 2016

@author: andrew
'''
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from pymongo import MongoClient
import igraph
import re
import codecs
from classifier import text
# from util.treebank_to_wordnet import get_wordnet_pos

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
        wordCooccurance = {}
        engStopWords = stopwords.words("english")
        
        #https://www.cs.cmu.edu/~ark/TweetNLP/gimpel+etal.acl11.pdf
        omit_pos = set(["#", "@", "E", "U", "~", ","])
        
        #Process context and context POS features
        for text in texts:
            tokens, pos = text
            tokens = tokens.split()
            pos = pos.split()
#             text = unicode(text)
            wordsNoPunc = [] #reinitialize on a per text basis, not per sentence
            for i in range(len(pos)):
                if pos[i] not in omit_pos:
                    word = tokens[i].lower()
            
#             sentences = sent_tokenize(text)
#             for sentence in sentences:
#                 words = word_tokenize(sentence)
# #                 wordsAndPOS = pos_tag(words)
# #                 for word, pos in wordsAndPOS:
#                 for word in words:
                    word = word.lower()
                    wordNoPunc = ALPHANUM_PAT.sub("", word)
                    if len(wordNoPunc) > 0:
                        if word not in engStopWords: #ignore stopwords
#                             wordLemma = wnl.lemmatize(wordNoPunc, pos=get_wordnet_pos(pos))
#                             wordsNoPunc.append(wordLemma.lower())
                            wordsNoPunc.append(porter.stem(wordNoPunc))
                words = wordsNoPunc          
                
                for i in range(len(words)):
                    for j in range(len(words)):
                        if i != j:
                            word1 = words[i]
                            word2 = words[j]
                     
                            if word1 not in wordCooccurance: #if the word hasn't been initialize
                                wordCooccurance[word1] = {} #initialize to a blank dictionary
                            wordCooccurance[word1][word2] = wordCooccurance[word1].get(word2, 0) + 1 #increment the count
                            
                            if word2 not in wordCooccurance: #if the word hasn't been initialize
                                wordCooccurance[word2] = {} #initialize to a blank dictionary
                            wordCooccurance[word2][word1] = wordCooccurance[word2].get(word1, 0) + 1 #increment the count
        
        #create graph
        print "word cooccurance calculated"
        words = wordCooccurance.keys()
        self.graph = igraph.Graph.Full(len(words)) #create a fully connected graph with as many vertices as words in the lexicon
        #Adding the edges individually causes the graph to be reindexed for each edge.
        #Creating all the edges at once is significantly faster
        self.graph.vs["name"] = words #set the names of the vertices to the words in the lexicon
#         self.graph.vs["label"] = words #set the labels for the graph accordingly
        self.graph.es["weight"] = None #make graph weighted
        toRemove = [] #the edges which will be removed
        
        for i in range(0, len(words)-1):
            print "{}/{}".format (i, len(words)-2)
            
            for j in range(i+1, len(words)):
                cooccurance = wordCooccurance[words[i]].get(words[j], 0)
                if cooccurance == 0: #there is no similarity between the two
                    toRemove.append((i, j)) #add the index pair to the list of edges to be removed (cuts down on reindexing time)
                else:
                    self.graph[words[i], words[j]] = cooccurance #assign weight
        
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
#     with codecs.open("C:\\Users\\Andrew\\Desktop\\radev -caption corpus\\data\\445.data", "r", "utf-8") as hybridCarSubmissions:
#         hybridCarSubmissions.readline() #we can ignore the first line
#         for line in hybridCarSubmissions:
#             caption = line.split("\t")[1] #get the second item in the tab delimited file
#             texts.append(caption)
    client = MongoClient()
    tweets = client.tweets.GentlerSongs.find({"ark token" : {"$exists":True}})
    for tweet in tweets:
        texts.append((tweet["ark token"], tweet["ark pos"]))
     
    lsg = LexicalSimilarityGraph(texts)
#     lsg.graph.write("hybridcar.Lemma.27.pickle", "pickle")
#     graph = igraph.Graph.Read("gentlersongs.pickle", "pickle")
#     print "pickle written"
    communities = lsg.graph.community_multilevel(weights=lsg.graph.es["weight"], return_levels=True)
    print "clustering done"
    with open("clusters.gentlersongs", "w") as f:
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
        