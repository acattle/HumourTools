'''
Created on Jan 27, 2017

@author: Andrew Cattle <acattle@connect.ust.hk>

This module contains methods for reading and working with word association files
using the NetworkX library.

Please not that all constructers expect the respective pajek files available at
http://vlado.fmf.uni-lj.si/pub/networks/data/dic/eat/Eat.htm for EAT
and http://vlado.fmf.uni-lj.si/pub/networks/data/dic/fa/FreeAssoc.htm for USF.
'''
from __future__ import print_function, division #For Python 2.7 compatibility
from networkx import read_pajek, shortest_path_length, write_pajek
from networkx.classes.function import set_edge_attributes
from networkx.classes.digraph import DiGraph
from math import log10
from networkx.exception import NetworkXNoPath, NodeNotFound
from functools import lru_cache
import itertools

class AssociationNetworkx:
    
    def get_all_associations(self):
        """
            Return all word pairs and their association strength
            
            :returns: An edge view representing all word pairs and their strengths (effectively a list of (stim, resp, weight) tuples)
            :rtype: networkx.EdgeView
        """
        return self.graph.edges(data='weight', default=0)
    
    def get_association_strengths(self, word_pairs):
        """
            Get association strengths between word_pairs.
            
            Association strengths are equal to the maximal path-product between
            nodes representing the word pairs. I.e. if edge weights represent
            the probability of moving from U to V, we find the path between
            U and V which maximizes this probability.
            
            :param word_pairs: the word pairs to calculate strengths for
            :type word_pairs: Iterable[Tuple[str, str]]
            
            :returns: the calculated strengths
            :rtype: List[float]
        """
        
        strengths = []
        for a, b in word_pairs:
            strengths.append(self.get_association_strength(a,b))
        
        return strengths
    
    @lru_cache(maxsize=None)
    def get_association_strength(self, cue, response):
        """
            Get association strength between a single word pair.
            
            Association strength is equal to the maximal path-product between
            nodes representing the word pairs. I.e. if edge weights represent
            the probability of moving from U to V, we find the path between
            U and V which maximizes this probability.
            
            :param cue: the cue word to calculate strength for
            :type cue: str
            :param response: the response word to calculate strength for
            :type response: str
            
            :returns: the calculated strength
            :rtype: float
        """
        
        strength = 0
        try:
            #Finding the shortest path between nodes according to -log weight isequivalent to finding the maximal product path
            #i.e. the path which maximizes the chain probability
            strength = 10 ** -(shortest_path_length(self.graph, cue.upper(), response.upper(), "-log weight"))
        except (NodeNotFound, NetworkXNoPath):
            #either one of the words is out-of-vocabulary
            #or there is no path between them
            #Either way, fail silent and default to 0
            pass
            
        
        return strength

class EATNetworkx(AssociationNetworkx):
    def __init__(self, eat_pajek_loc):
        self.graph = read_pajek(eat_pajek_loc)
        self.graph = DiGraph(self.graph) #for some reason pajek defaults to multi
        
        
        #Convert edge weights from count to percentage
        weights_as_proportions = {}
        #Also calculate the negative log of the percentage for use finding maximal product path
        neg_log_proportions = {}
        for stimuli, response, weight in self.graph.edges(data='weight', default=0):
            s_degree = self.graph.degree(stimuli, weight="weight")
            proportion = float(weight)/s_degree
            weights_as_proportions[(stimuli, response)] = proportion
            neg_log_proportions[(stimuli, response)] = -log10(proportion)
         
        set_edge_attributes(self.graph, weights_as_proportions, "weight")
        set_edge_attributes(self.graph, neg_log_proportions, "-log weight")
        
        
        
        
        
class USFNetworkx(AssociationNetworkx):
    def __init__(self, usf_pajek_loc):
        self.graph = read_pajek(usf_pajek_loc)
        self.graph = DiGraph(self.graph) #for some reason pajek defaults to multi
        
        #calculate the negative log of the percentage for use finding maximal product path
        neg_log_proportions = {}
        for stimuli, response, weight in self.graph.edges(data='weight', default=0):
            neg_log_proportions[(stimuli, response)] = -log10(weight)
         
        set_edge_attributes(self.graph, neg_log_proportions, "-log weight")


def  networkxFromTuples(association_tuples):
    """
    Creates an AssociationNetworkx from a list of strength tuples
    
    :param: association_tuples: list of association tuples of form ((stim, resp), stren)
    :type: association_tuples: List[Tuple[Tuple[str,str], float]]
    
    :return: Networkx of the association tuples
    :rtype: AssociationNetworkx
    """
    
    #get unique words
    vocab = set()
    for (s,r), _ in association_tuples:
        vocab.add(s.upper())
        vocab.add(r.upper())
    vocab = list(vocab) #convert to ordered list
    
    
    graph = DiGraph()
    graph.add_nodes_from(vocab)
    association_tuples = [(s.upper(),r.upper(),stren) for (s,r), stren in association_tuples]
    graph.add_weighted_edges_from(association_tuples)
    
    #get negative log weights
#     neg_log_proportions = {}
#     for stimuli, response, weight in graph.edges(data='weight', default=0):
#         neg_log_proportions[(stimuli, response)] = -log10(weight)
#     set_edge_attributes(graph, neg_log_proportions, "-log weight")
    
    
    assoc_object = AssociationNetworkx()
    assoc_object.graph = graph
    return assoc_object


import os
import re
import codecs
import glob
from multiprocessing.pool import Pool
from util.misc import mean
from nltk.corpus import stopwords
from time import strftime
english_stopwords = stopwords.words("english")
pos_to_ignore = ["D","P","X","Y", "T", "&", "~", ",", "!", "U", "E"]
semeval_dir = r"D:/datasets/SemEval Data"
dirs = [r"trial_dir/trial_data",
        r"train_dir/train_data",
        r"evaluation_dir/evaluation_data"]
tagged_dir = "tagged"
from word_associations.association_readers.xml_readers import SWoW_Dataset, SWoW_Strengths_Dataset
from word_associations.association_readers.igraph_readers import iGraphFromTuples,\
    EATIGraph
swow_all = SWoW_Dataset("D:/datasets/SWoW/SWOW-EN.complete.csv").get_all_associations()
swow_all_graph = iGraphFromTuples(swow_all)
def process_file(f):
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
      
#         eat_fileloc = u"{}.eat".format(f)
#         usf_fileloc = u"{}.usf".format(f)
    swow_fileloc = f"{f}.swow"
    done = 0
    with codecs.open(swow_fileloc, "w", encoding="utf-8") as swow_file:#, codecs.open(eat_fileloc, "w", encoding="utf-8") as eat_file, codecs.open(usf_fileloc, "w", encoding="utf-8") as usf_file:
        for tokens, pos, tweet_id in zip(tweet_tokens,tweet_pos, tweet_ids):
#                 eat_f_results_by_word = []
#                 eat_b_results_by_word = []
#                 usf_f_results_by_word = []
#                 usf_b_results_by_word = []
            swow_f_results_by_word = []
            swow_b_results_by_word = []
              
            for word in hashtag_words:
#                     eat_f_by_hashtag_word=[]
#                     eat_b_by_hashtag_word=[]
#                     usf_f_by_hashtag_word=[]
#                     usf_b_by_hashtag_word=[]
                swow_f_by_hashtag_word=[]
                swow_b_by_hashtag_word=[]
                for token, tag in zip(tokens, pos):
                    token=token.lower()
                    if (tag in pos_to_ignore) or (token in english_stopwords):
                        continue
                    if (token == "@midnight") or (token == hashtag): #if it's the @midnight account of the game's hashtag
                            continue #we don't want to process it
                      
                    forward = u" ".join([word, token])
                    backward = u" ".join([token, word])
#                         eat_f=None
#                         eat_b=None
#                         usf_f=None
#                         usf_b=None
                    swow_f=None
                    swow_b=None
#                         if forward in cached_scores:
#                             eat_f, usf_f = cached_scores[forward]
#                         if backward in cached_scores:
#                             eat_b, usf_b = cached_scores[backward]
#                         if (eat_f == None) or (usf_f == None) or (eat_b == None) or (usf_b == None):
#                         try:
#                             eat_word = eat.graph.vs.find(word.upper())
#                             eat_token = eat.graph.vs.find(token.upper())
#                             eat_f = eat.graph.shortest_paths(eat_word, eat_token, weights="-log weight")[0][0]
#                             eat_b = eat.graph.shortest_paths(eat_token, eat_word, weights="-log weight")[0][0]
#                         except ValueError:
#                             #that word is not in our vocab.
#                             eat_f = float("inf")
#                             eat_b = float("inf")
#                               
#                         try:
#                             usf_word = usf.graph.vs.find(word.upper())
#                             usf_token = usf.graph.vs.find(token.upper())
#                             usf_f = usf.graph.shortest_paths(usf_word, usf_token, weights="-log weight")[0][0]
#                             usf_b = usf.graph.shortest_paths(usf_token, usf_word, weights="-log weight")[0][0]
#                         except ValueError:
#                             #that word is not in our vocab.
#                             usf_f = float("inf")
#                             usf_b = float("inf")
                    
                    try:
                        swow_word = swow_all_graph.graph.vs.find(word.upper())
                        swow_token = swow_all_graph.graph.vs.find(token.upper())
#                         swow_f = swow_all_graph.graph.shortest_paths(swow_word, swow_token, weights="-log weight")[0][0]
#                         swow_b = swow_all_graph.graph.shortest_paths(swow_token, swow_word, weights="-log weight")[0][0]
                        swow_f = swow_strengths_map.get((word.lower(),token.lower()))
                        swow_b = swow_strengths_map.get((token.lower(),word.lower()))
                    except ValueError:
                        #that word is not in our vocab.
                        swow_f = float("inf")
                        swow_b = float("inf")
                              
#                         cached_scores[forward] = (eat_f, usf_f)
#                         cached_scores[backward] = (eat_b, usf_b)
                                                      
#                         eat_f_by_hashtag_word.append(eat_f)
#                         eat_b_by_hashtag_word.append(eat_b)
#                         usf_f_by_hashtag_word.append(usf_f)
#                         usf_b_by_hashtag_word.append(usf_b)
                    swow_f_by_hashtag_word.append(swow_f)
                    swow_b_by_hashtag_word.append(swow_b)
                      
                if len(swow_f_by_hashtag_word) == 0:#len(eat_f_by_hashtag_word) == 0:
                    print(u"ERRORL no valid tokens\t{}".format(u" ".join(tokens)))
#                         eat_f_by_hashtag_word=[float("inf")]
#                         eat_b_by_hashtag_word=[float("inf")]
#                         usf_f_by_hashtag_word=[float("inf")]
#                         usf_b_by_hashtag_word=[float("inf")]
                    swow_f_by_hashtag_word=[float("inf")]
                    swow_b_by_hashtag_word=[float("inf")]
                  
                #TODO: should max and min be swapped? the smallest probability will have the largest negative log
#                     eat_f_results_by_word.append((min(eat_f_by_hashtag_word), mean(eat_f_by_hashtag_word), max(eat_f_by_hashtag_word)))
#                     eat_b_results_by_word.append((min(eat_b_by_hashtag_word), mean(eat_b_by_hashtag_word), max(eat_b_by_hashtag_word)))
#                     usf_f_results_by_word.append((min(usf_f_by_hashtag_word), mean(usf_f_by_hashtag_word), max(usf_f_by_hashtag_word)))
#                     usf_b_results_by_word.append((min(usf_b_by_hashtag_word), mean(usf_b_by_hashtag_word), max(usf_b_by_hashtag_word)))
                swow_f_results_by_word.append((min(swow_f_by_hashtag_word), mean(swow_f_by_hashtag_word), max(swow_f_by_hashtag_word)))
                swow_b_results_by_word.append((min(swow_b_by_hashtag_word), mean(swow_b_by_hashtag_word), max(swow_b_by_hashtag_word)))
              
#                 eat_f_mins, eat_f_avgs, eat_f_maxes = zip(*eat_f_results_by_word) #separate out the columns
#                 eat_b_mins, eat_b_avgs, eat_b_maxes = zip(*eat_b_results_by_word)
#                 usf_f_mins, usf_f_avgs, usf_f_maxes = zip(*usf_f_results_by_word)
#                 usf_b_mins, usf_b_avgs, usf_b_maxes = zip(*usf_b_results_by_word)
            swow_f_mins, swow_f_avgs, swow_f_maxes = zip(*swow_f_results_by_word)
            swow_b_mins, swow_b_avgs, swow_b_maxes = zip(*swow_b_results_by_word)
             
            #TODO: should max and min be swapped? the smallest probability will have the largest negative log
#                 eat_f_overall = (min(eat_f_mins), mean(eat_f_avgs), max(eat_f_maxes))
#                 eat_b_overall = (min(eat_b_mins), mean(eat_b_avgs), max(eat_b_maxes))
#                 usf_f_overall = (min(usf_f_mins), mean(usf_f_avgs), max(usf_f_maxes))
#                 usf_b_overall = (min(usf_b_mins), mean(usf_b_avgs), max(usf_b_maxes))
            swow_f_overall = (min(swow_f_mins), mean(swow_f_avgs), max(swow_f_maxes))
            swow_b_overall = (min(swow_b_mins), mean(swow_b_avgs), max(swow_b_maxes))
          
#                 per_word_eat_f = u" ".join([u"{}:{}:{}".format(*res) for res in eat_f_results_by_word])
#                 per_word_eat_b = u" ".join([u"{}:{}:{}".format(*res) for res in eat_b_results_by_word])
#                 per_word_usf_f = u" ".join([u"{}:{}:{}".format(*res) for res in usf_f_results_by_word])
#                 per_word_usf_b = u" ".join([u"{}:{}:{}".format(*res) for res in usf_b_results_by_word])
            per_word_swow_f = u" ".join([u"{}:{}:{}".format(*res) for res in swow_f_results_by_word])
            per_word_swow_b = u" ".join([u"{}:{}:{}".format(*res) for res in swow_b_results_by_word])
              
#                 overall_eat_f = u"{} {} {}".format(*eat_f_overall)
#                 overall_eat_b = u"{} {} {}".format(*eat_b_overall)
#                 overall_usf_f = u"{} {} {}".format(*usf_f_overall)
#                 overall_usf_b = u"{} {} {}".format(*usf_b_overall)
            overall_swow_f = u"{} {} {}".format(*swow_f_overall)
            overall_swow_b = u"{} {} {}".format(*swow_b_overall)
              
#                 eat_line = u"{}\t{}\t{}\t{}\t{}\n".format(tweet_id, overall_eat_f, overall_eat_b, per_word_eat_f, per_word_eat_b)
#                 eat_file.write(eat_line)
#                 usf_line = u"{}\t{}\t{}\t{}\t{}\n".format(tweet_id, overall_usf_f, overall_usf_b, per_word_usf_f, per_word_usf_b)
#                 usf_file.write(usf_line)
            swow_line = u"{}\t{}\t{}\t{}\t{}\n".format(tweet_id, overall_swow_f, overall_swow_b, per_word_swow_f, per_word_swow_b)
            swow_file.write(swow_line)
            done+=1
            if done % 20 == 0:
                print("{}\t{}\t{} of {} completed".format(strftime("%y-%m-%d_%H:%M:%S"), name, done, len(tweet_ids)))
    print("{}\tfinished {}".format(strftime("%y-%m-%d_%H:%M:%S"),name))
                    
if __name__ == "__main__":
    import os
    import re
    import codecs
    import glob
    from multiprocessing.pool import Pool
    from util.misc import mean
    from nltk.corpus import stopwords
    from time import strftime
    from word_associations.association_readers.xml_readers import SWoW_Dataset, SWoW_Strengths_Dataset
    from word_associations.association_readers.igraph_readers import iGraphFromTuples,\
        EATIGraph
    
    
    
#     write_pajek(swow_all_graph.graph, "D:/datasets/SWoW/swow_all.net")
#     
#     swow_100 = SWoW_Dataset("D:/datasets/SWoW/SWOW-EN.R100.csv",complete=False).get_all_associations()
#     swow_100_graph = networkxFromTuples(swow_100)
#     write_pajek(swow_100_graph.graph, "D:/datasets/SWoW/swow_100.net")
#     
#     swow_stren = SWoW_Strengths_Dataset("D:/datasets/SWoW/strength.SWOW-EN.R123.csv").get_all_associations()
#     swow_stren_graph = networkxFromTuples(swow_stren)
#     write_pajek(swow_stren_graph.graph, "D:/datasets/SWoW/swow_stren.net")
    
    
    
    
    
    
    
    
    
    
    
    
    english_stopwords = stopwords.words("english")
    pos_to_ignore = ["D","P","X","Y", "T", "&", "~", ",", "!", "U", "E"]
    semeval_dir = r"D:/datasets/SemEval Data"
    dirs = [r"trial_dir/trial_data",
            r"train_dir/train_data",
            r"evaluation_dir/evaluation_data"]
    tagged_dir = "tagged"
    swow_all = SWoW_Dataset("D:/datasets/SWoW/SWOW-EN.complete.csv").get_all_associations()
    swow_all_graph = iGraphFromTuples(swow_all)                    
#     from networkx.algorithms.centrality.load import load_centrality
#     from networkx.algorithms.centrality.betweenness import betweenness_centrality
#     eat_loc = "C:/Users/Andrew/git/HumourDetection/HumourDetection/src/Data/eat/pajek/EATnew2.net"
    
    eat_loc = "../shortest_paths/EATnew2.net"
#     usf_loc = "C:/Users/Andrew/git/HumourDetection/HumourDetection/src/Data/PairsFSG2.net"
    usf_loc = "../shortest_paths/PairsFSG2.net"

      
#     eat = EATGraph(eat_loc)
#     usf = USFGraph(usf_loc)
    
#     cached_scores={}
    
          
    filenames = []
    word_pairs = set()
    for d in dirs:
        os.chdir(os.path.join(semeval_dir, d, tagged_dir))
        for f in glob.glob("*.tsv"):
            filenames.append(os.path.join(semeval_dir, d, tagged_dir,f))
            
            name = os.path.splitext(os.path.basename(f))[0]
            hashtag = "#{}".format(re.sub("_", "", name.lower()))
            hashtag_words = name.split("_")        
            #remove swords that don't give you some idea of the domain
            hashtag_words = [word.lower() for word in hashtag_words if word.lower() not in english_stopwords]
            #the next 3 are to catch "<blank>In#Words" type hashtags
            hashtag_words = [word for word in hashtag_words if word != "in"]
            hashtag_words = [word for word in hashtag_words if not ((len(word) == 1) and (word.isdigit()))]
            hashtag_words = [word for word in hashtag_words if word != "words"]
              
            tweet_ids = []
            tweet_tokens = []
            tweet_pos = []
            with codecs.open(f, "r", encoding="utf-8") as tweet_file:
                for line in tweet_file:
                    line=line.strip().lower()
                    if line == "":
                        continue
                    line_split = line.split("\t")
                    tweet_tokens.append(line_split[0].split())
                    tweet_pos.append(line_split[1].split())
                    tweet_ids.append(line_split[3])
                      
                    filtered_tokens = [token.lower() for tokens, pos in zip(tweet_tokens,tweet_pos) for token, tag in zip(tokens, pos) if (tag not in pos_to_ignore) and (token.lower() not in english_stopwords) and (token.lower() != "@midnight") and (token.lower() != hashtag)]
                      
                    word_pairs.update(itertools.product(hashtag_words, filtered_tokens))
                    word_pairs.update(itertools.product(filtered_tokens, hashtag_words))
  
    word_pairs = list(word_pairs)

    import pickle
    swow_strengths_map = dict(zip(word_pairs, swow_all_graph.get_association_strengths(word_pairs)))
    with open("swow_graph_map_semeval.pkl", "wb") as f:
        pickle.dump(swow_strengths_map, f)
#     with open("swow_graph_map_semeval.pkl","rb") as f:
#         swow_strengths_map = pickle.load(f)
    
#     p=Pool(8)
#        
#     p.map(process_file, filenames)
    for f in filenames:
        process_file(f)
            
    
#     
#     try:

#     from networkx import all_pairs_dijkstra_path_length
#     print("{} loading EAT".format(strftime("%y-%m-%d_%H:%M:%S")))
#     g = EATNetworkx(eat_loc)
#     
# #     load = load_centrality(g.graph, weight="weight")
# #     with open("eat_load_weighted.pkl", "wb") as load_file:
# #         pickle.dump(load,load_file)
# #          
# #     betweenness = betweenness_centrality(g.graph, weight="weight")
# #     with open("eat_betweenness_weighted.pkl", "wb") as betweenness_file:
# #         pickle.dump(betweenness, betweenness_file)
# #         with open("eat_graph.pkl", "wb") as graph_file:
# #             pickle.dump(g, graph_file)
# 
#     print("{} starting EAT paths".format(strftime("%y-%m-%d_%H:%M:%S")))
#     path_matrix = all_pairs_dijkstra_path_length(g.graph, weights="-log weight")
#     print("{} EAT paths finished".format(strftime("%y-%m-%d_%H:%M:%S")))
#     with open("eat_path_matrix.pkl", "wb") as matrix_file:
#         pickle.dump(path_matrix, matrix_file)
# #     except Exception,e:
# #         print(e)
# #         
# #     try:    
#     print("{} loading USF".format(strftime("%y-%m-%d_%H:%M:%S")))
#     g = USFNetworkx(usf_loc)
#     
# #     load = load_centrality(g.graph, weight="weight")
# #     with open("usf_load_weighted.pkl", "wb") as load_file:
# #         pickle.dump(load,load_file)
# #          
# #     betweenness = betweenness_centrality(g.graph, weight="weight")
# #     with open("usf_betweenness_weighted.pkl", "wb") as betweenness_file:
# #         pickle.dump(betweenness, betweenness_file)
# #     
# #     print("done")
# #         with open("usf_graph.pkl", "wb") as graph_file:
# #             pickle.dump(g, graph_file)
#     print("{} starting USF paths".format(strftime("%y-%m-%d_%H:%M:%S")))
#     path_matrix = all_pairs_dijkstra_path_length(g.graph, weights="-log weight")
#     print("{} USF paths finished".format(strftime("%y-%m-%d_%H:%M:%S")))
#     with open("usf_path_matrix.pkl", "wb") as matrix_file:
#         pickle.dump(path_matrix, matrix_file)
# #     except Exception,e:
# #         print(e  )
# 
# #     print("{} loading Evocation".format(strftime("%y-%m-%d_%H:%M:%S")))
# 
# #     evocation_dir = "C:/Users/Andrew/git/HumourDetection/HumourDetection/src/Data/evocation/"
# #     e = EvocationDataset(evocation_dir)
# #     e.get_all_associations()
# # #     with open("evocation_reader.pkl", "wb") as evocation_file:
# # #         pickle.dump(e, evocation_file)