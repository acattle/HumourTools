'''
Created on Jan 27, 2017

@author: Andrew
'''
from __future__ import print_function, division #For Python 2.7 compatibility
from networkx import read_pajek, shortest_path_length
from numpy import mean
from networkx.classes.function import set_edge_attributes
from networkx.classes.digraph import DiGraph
from math import log
import pickle
from time import strftime
# from igraph import Graph, OUT
# import os
# import glob
# import re
# import codecs
# from nltk.corpus import stopwords
# from multiprocessing import Pool
from collections import defaultdict
from os.path import join
from networkx.algorithms.centrality.load import load_centrality
from networkx.algorithms.centrality.betweenness import betweenness_centrality
import xml.etree.ElementTree as ET

class EATGraph(object):
    def __init__(self, eat_pajek_loc):
        #networkx
        self.graph = read_pajek(eat_pajek_loc)
        self.graph = DiGraph(self.graph) #for some reason pajek defaults to multi
        weights_as_proportions = {}
        neg_log_proportions = {}
        for stimuli, response, weight in self.graph.edges(data='weight', default=0):
            s_degree = self.graph.degree(stimuli, weight="weight")
            proportion = float(weight)/s_degree
            weights_as_proportions[(stimuli, response)] = proportion
            neg_log_proportions[(stimuli, response)] = -log(proportion)
         
        set_edge_attributes(self.graph, weights_as_proportions, "weight")
        set_edge_attributes(self.graph, neg_log_proportions, "-log weight")
         
        #igraph
#         self.graph = Graph.Read_Pajek(eat_pajek_loc)
#         self.graph.vs["name"] = self.graph.vs["id"]#work around for bug: https://github.com/igraph/python-igraph/issues/86
#         proportions = []
#         neg_log_proportions = []
#         out_degrees = self.graph.strength(self.graph.vs, mode=OUT, weights="weight")
#         for e in self.graph.es:
#             proportion = float(e["weight"])/out_degrees[e.source]
#             proportions.append(proportion)
#             neg_log_proportions.append(-log(proportion))
#         
#         self.graph.es["weight"] = proportions
#         self.graph.es["-log weight"] = neg_log_proportions
        
    
    def get_all_associations(self):
        return self.graph.edges(data='weight', default=0)

class EAT_XML_Reader():
    def __init__(self, xml_loc):
        self.eat_root = ET.parse(xml_loc).getroot()
        
    def get_all_associations(self):
        associations=[]
        
        for stimulus_element in self.eat_root:
            stimuli = stimulus_element.attrib["word"]
            total = float(stimulus_element.attrib["all"])
            
            for response_element in stimulus_element:
                response = response_element.attrib["word"]
                count = float(response_element.attrib["n"])
                
                associations.append((stimuli, response, count/total))
        
        return associations

class USFGraph(object):
    def __init__(self, usf_pajek_loc):
        #igraph
#         self.graph = Graph.Read_Pajek(usf_pajek_loc)
#         self.graph.vs["name"] = self.graph.vs["id"]#work around for bug: https://github.com/igraph/python-igraph/issues/86
#         neg_log_proportions = []
#         for e in self.graph.es:
#         for e in self.graph.edges:
#             neg_log_proportions.append(-log(e["weight"]))
#         
#         self.graph.es["-log weight"] = neg_log_proportions

        #networkx
        self.graph = read_pajek(usf_pajek_loc)
        self.graph = DiGraph(self.graph) #for some reason pajek defaults to multi
        neg_log_proportions = {}
        for stimuli, response, weight in self.graph.edges(data='weight', default=0):
            neg_log_proportions[(stimuli, response)] = -log(weight)
         
        set_edge_attributes(self.graph, neg_log_proportions, "-log weight")
    
    def get_all_associations(self):
        return self.graph.edges(data='weight', default=0)
    
class EvocationDataset(object):
    def __init__(self, evocation_dir, mt_group="mt_all"):
        """ Read evocation scores from "controlled" group plus an optional Mechanical Turk group.
            
            @param evocation_dir: the directory where the evocation files are stored
            @type evocation_dir: str
            @param mt_group: The Mechanical Turk group of judgements to read in addition to the controlled group. Must be one of "mt_all", "mt_most". "mt_some", or None.
            @type mt_group: str or None
        """
        
        #read in the control data
        evocation_scores = self._read_scores(evocation_dir, "controlled")
        
        #if we're using MT data
        if mt_group:
            #add mt judgements
            evocation_scores = self._read_scores(evocation_dir, mt_group, evocation_scores)
        
        #take simple average of scores
        self.evocation = {}
        for synset_pair, judgements in evocation_scores.items():
            if len(synset_pair) != 2:
                print(synset_pair)
            self.evocation[synset_pair] = mean(judgements)/100.0 #average judgements and convert to a probability
                
    def _read_scores(self, evocation_dir, group="controlled", evocation_scores=defaultdict(list)):
        """ Convenience method for reading WordNet Evocation files
            
            @param evocation_dir: The directory where the evocation files are stored
            @type evocation_dir: str
            @param group: The group of judgements to read. Must be one of "controlled", "mt_all", "mt_most", or "mt_some"
            @type group: str
            @param evocation_scores: A dictionary containing already counted evocation scores. Used for combining scores from multiple groups. Keys are (first synset, second synset). Values are a list of floats
            @type evocation_scores: defaultdict(list)
            
            @return A dictionary containing evocation scores. Keys are (first synset, second synset). Values are a list of floats
            @rtype defaultdict(list)
        """
        word_pos_loc = join(evocation_dir, "{}.word-pos-sense".format(group))
        raw_loc = join(evocation_dir, "{}.raw".format(group))
        with open(word_pos_loc, "r") as word_pos_file, open(raw_loc, "r") as raw_file:
            for word_pos_line, raw_line in zip(word_pos_file, raw_file):
                synset1, synset2 = word_pos_line.strip().split(",")
                scores = [float(score) for score in raw_line.split()]
            
                evocation_scores[(synset1, synset2)].extend(scores)
        
        return evocation_scores
    
    def get_all_associations(self):
        """ Method for getting all assocation scores
            
            @return A list of tuples representing (first synset, second synset, evocation score as a probability)
            @rtype [(str, str, float)]
        """
        associations = []
        for synset_pair, evocation_score in self.evocation.items():
            if len(synset_pair) != 2:
                print(synset_pair)
            synset1, synset2 = synset_pair
            associations.append((synset1, synset2, evocation_score))
                
        return associations  

                                   
if __name__ == "__main__":
#     eat_loc = "C:/Users/Andrew/git/HumourDetection/HumourDetection/src/Data/eat/pajek/EATnew2.net"
    eat_loc = "../shortest_paths/EATnew2.net"
#     usf_loc = "C:/Users/Andrew/git/HumourDetection/HumourDetection/src/Data/PairsFSG2.net"
    usf_loc = "../shortest_paths/PairsFSG2.net"
#     english_stopwords = stopwords.words("english")
#     pos_to_ignore = ["D","P","X","Y", "T", "&", "~", ",", "!", "U", "E"]
#     semeval_dir = r"C:/Users/Andrew/Desktop/SemEval Data"
#     dirs = [r"trial_dir/trial_data",
#             r"train_dir/train_data",
#             r"evaluation_dir/evaluation_data"]
#     tagged_dir = "tagged"
#      
#     eat = EATGraph(eat_loc)
#     usf = USFGraph(usf_loc)
#      
#     cached_scores={}
#     def process_file(f):
#         name = os.path.splitext(os.path.basename(f))[0]
#         hashtag = "#{}".format(re.sub("_", "", name.lower()))
#         hashtag_words = name.split("_")        
#         #remove swords that don't give you some idea of the domain
#         hashtag_words = [word.lower() for word in hashtag_words if word.lower() not in english_stopwords]
#         #the next 3 are to catch "<blank>In#Words" type hashtags
#         hashtag_words = [word for word in hashtag_words if word != "in"]
#         hashtag_words = [word for word in hashtag_words if not ((len(word) == 1) and (word.isdigit()))]
#         hashtag_words = [word for word in hashtag_words if word != "words"]
#          
#         print("{}\tprocessing {}".format(strftime("%y-%m-%d_%H:%M:%S"),name))
#         tweet_ids = []
#         tweet_tokens = []
#         tweet_pos = []
#         with codecs.open(f, "r", encoding="utf-8") as tweet_file:
#             for line in tweet_file:
#                 line=line.strip()
#                 if line == "":
#                     continue
#                 line_split = line.split("\t")
#                 tweet_tokens.append(line_split[0].split())
#                 tweet_pos.append(line_split[1].split())
#                 tweet_ids.append(line_split[3])
#          
#         eat_fileloc = u"{}.eat".format(f)
#         usf_fileloc = u"{}.usf".format(f)
#         done = 0
#         with codecs.open(eat_fileloc, "w", encoding="utf-8") as eat_file, codecs.open(usf_fileloc, "w", encoding="utf-8") as usf_file:
#             for tokens, pos, tweet_id in zip(tweet_tokens,tweet_pos, tweet_ids):
#                 eat_f_results_by_word = []
#                 eat_b_results_by_word = []
#                 usf_f_results_by_word = []
#                 usf_b_results_by_word = []
#                  
#                 for word in hashtag_words:
#                     eat_f_by_hashtag_word=[]
#                     eat_b_by_hashtag_word=[]
#                     usf_f_by_hashtag_word=[]
#                     usf_b_by_hashtag_word=[]
#                     for token, tag in zip(tokens, pos):
#                         token=token.lower()
#                         if (tag in pos_to_ignore) or (token in english_stopwords):
#                             continue
#                         if (token == "@midnight") or (token == hashtag): #if it's the @midnight account of the game's hashtag
#                                 continue #we don't want to process it
#                          
#                         forward = u" ".join([word, token])
#                         backward = u" ".join([token, word])
#                         eat_f=None
#                         eat_b=None
#                         usf_f=None
#                         usf_b=None
#                         if forward in cached_scores:
#                             eat_f, usf_f = cached_scores[forward]
#                         if backward in cached_scores:
#                             eat_b, usf_b = cached_scores[backward]
#                         if (eat_f == None) or (usf_f == None) or (eat_b == None) or (usf_b == None):
#                             try:
#                                 eat_word = eat.graph.vs.find(word.upper())
#                                 eat_token = eat.graph.vs.find(token.upper())
#                                 eat_f = eat.graph.shortest_paths(eat_word, eat_token, weights="-log weight")[0][0]
#                                 eat_b = eat.graph.shortest_paths(eat_token, eat_word, weights="-log weight")[0][0]
#                             except ValueError:
#                                 #that word is not in our vocab.
#                                 eat_f = float("inf")
#                                 eat_b = float("inf")
#                                  
#                             try:
#                                 usf_word = usf.graph.vs.find(word.upper())
#                                 usf_token = usf.graph.vs.find(token.upper())
#                                 usf_f = usf.graph.shortest_paths(usf_word, usf_token, weights="-log weight")[0][0]
#                                 usf_b = usf.graph.shortest_paths(usf_token, usf_word, weights="-log weight")[0][0]
#                             except ValueError:
#                                 #that word is not in our vocab.
#                                 usf_f = float("inf")
#                                 usf_b = float("inf")
#                                  
#                             cached_scores[forward] = (eat_f, usf_f)
#                             cached_scores[backward] = (eat_b, usf_b)
#                                                          
#                         eat_f_by_hashtag_word.append(eat_f)
#                         eat_b_by_hashtag_word.append(eat_b)
#                         usf_f_by_hashtag_word.append(usf_f)
#                         usf_b_by_hashtag_word.append(usf_b)
#                          
#                     if len(eat_f_by_hashtag_word) == 0:
#                         print(u"ERRORL no valid tokens\t{}".format(u" ".join(tokens)))
#                         eat_f_by_hashtag_word=[float("inf")]
#                         eat_b_by_hashtag_word=[float("inf")]
#                         usf_f_by_hashtag_word=[float("inf")]
#                         usf_b_by_hashtag_word=[float("inf")]
#                      
#                     #TODO: should max and min be swapped? the smallest probability will have the largest negative log
#                     eat_f_results_by_word.append((min(eat_f_by_hashtag_word), mean(eat_f_by_hashtag_word), max(eat_f_by_hashtag_word)))
#                     eat_b_results_by_word.append((min(eat_b_by_hashtag_word), mean(eat_b_by_hashtag_word), max(eat_b_by_hashtag_word)))
#                     usf_f_results_by_word.append((min(usf_f_by_hashtag_word), mean(usf_f_by_hashtag_word), max(usf_f_by_hashtag_word)))
#                     usf_b_results_by_word.append((min(usf_b_by_hashtag_word), mean(usf_b_by_hashtag_word), max(usf_b_by_hashtag_word)))
#                  
#                 eat_f_mins, eat_f_avgs, eat_f_maxes = zip(*eat_f_results_by_word) #separate out the columns
#                 eat_b_mins, eat_b_avgs, eat_b_maxes = zip(*eat_b_results_by_word)
#                 usf_f_mins, usf_f_avgs, usf_f_maxes = zip(*usf_f_results_by_word)
#                 usf_b_mins, usf_b_avgs, usf_b_maxes = zip(*usf_b_results_by_word)
#                 
#                 #TODO: should max and min be swapped? the smallest probability will have the largest negative log
#                 eat_f_overall = (min(eat_f_mins), mean(eat_f_avgs), max(eat_f_maxes))
#                 eat_b_overall = (min(eat_b_mins), mean(eat_b_avgs), max(eat_b_maxes))
#                 usf_f_overall = (min(usf_f_mins), mean(usf_f_avgs), max(usf_f_maxes))
#                 usf_b_overall = (min(usf_b_mins), mean(usf_b_avgs), max(usf_b_maxes))
#              
#                 per_word_eat_f = u" ".join([u"{}:{}:{}".format(*res) for res in eat_f_results_by_word])
#                 per_word_eat_b = u" ".join([u"{}:{}:{}".format(*res) for res in eat_b_results_by_word])
#                 per_word_usf_f = u" ".join([u"{}:{}:{}".format(*res) for res in usf_f_results_by_word])
#                 per_word_usf_b = u" ".join([u"{}:{}:{}".format(*res) for res in usf_b_results_by_word])
#                  
#                 overall_eat_f = u"{} {} {}".format(*eat_f_overall)
#                 overall_eat_b = u"{} {} {}".format(*eat_b_overall)
#                 overall_usf_f = u"{} {} {}".format(*usf_f_overall)
#                 overall_usf_b = u"{} {} {}".format(*usf_b_overall)
#                  
#                 eat_line = u"{}\t{}\t{}\t{}\t{}\n".format(tweet_id, overall_eat_f, overall_eat_b, per_word_eat_f, per_word_eat_b)
#                 eat_file.write(eat_line)
#                 usf_line = u"{}\t{}\t{}\t{}\t{}\n".format(tweet_id, overall_usf_f, overall_usf_b, per_word_usf_f, per_word_usf_b)
#                 usf_file.write(usf_line)
#                 done+=1
#                 if done % 20 == 0:
#                     print("{}\t{}\t{} of {} completed".format(strftime("%y-%m-%d_%H:%M:%S"), name, done, len(tweet_ids)))
#         print("{}\tfinished {}".format(strftime("%y-%m-%d_%H:%M:%S"),name))
#          
#     filenames = []
#     for d in dirs:
#         os.chdir(os.path.join(semeval_dir, d, tagged_dir))
#         for f in glob.glob("*.tsv"):
#             filenames.append(os.path.join(semeval_dir, d, tagged_dir,f))
#      
#     p=Pool(8)
#      
#     p.map(process_file, filenames)
            
    
#     
#     try:
    print("{} loading EAT".format(strftime("%y-%m-%d_%H:%M:%S")))
    g = EATGraph(eat_loc)
    
    load = load_centrality(g.graph, weight="weight")
    with open("eat_load_weighted.pkl", "wb") as load_file:
        pickle.dump(load,load_file)
         
    betweenness = betweenness_centrality(g.graph, weight="weight")
    with open("eat_betweenness_weighted.pkl", "wb") as betweenness_file:
        pickle.dump(betweenness, betweenness_file)
#         with open("eat_graph.pkl", "wb") as graph_file:
#             pickle.dump(g, graph_file)
#         print("{} starting EAT paths".format(strftime("%y-%m-%d_%H:%M:%S")))
#         path_matrix = g.graph.shortest_paths(weights="-log weight")
#         print("{} EAT paths finished".format(strftime("%y-%m-%d_%H:%M:%S")))
#         with open("eat_path_matrix.pkl", "wb") as matrix_file:
#             pickle.dump(path_matrix, matrix_file)
#     except Exception,e:
#         print(e)
#         
#     try:    
    print("{} loading USF".format(strftime("%y-%m-%d_%H:%M:%S")))
    g = USFGraph(usf_loc)
    
    load = load_centrality(g.graph, weight="weight")
    with open("usf_load_weighted.pkl", "wb") as load_file:
        pickle.dump(load,load_file)
         
    betweenness = betweenness_centrality(g.graph, weight="weight")
    with open("usf_betweenness_weighted.pkl", "wb") as betweenness_file:
        pickle.dump(betweenness, betweenness_file)
    
    print("done")
#         with open("usf_graph.pkl", "wb") as graph_file:
#             pickle.dump(g, graph_file)
#         print("{} starting USF paths".format(strftime("%y-%m-%d_%H:%M:%S")))
#         path_matrix = g.graph.shortest_paths(weights="-log weight")
#         print("{} USF paths finished".format(strftime("%y-%m-%d_%H:%M:%S")))
#         with open("usf_path_matrix.pkl", "wb") as matrix_file:
#             pickle.dump(path_matrix, matrix_file)
#     except Exception,e:
#         print(e  )

#     print("{} loading Evocation".format(strftime("%y-%m-%d_%H:%M:%S")))

#     evocation_dir = "C:/Users/Andrew/git/HumourDetection/HumourDetection/src/Data/evocation/"
#     e = EvocationDataset(evocation_dir)
#     e.get_all_associations()
# #     with open("evocation_reader.pkl", "wb") as evocation_file:
# #         pickle.dump(e, evocation_file)