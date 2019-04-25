'''
Created on Feb 7, 2018

@author: Andrew Cattle <acattle@connect.ust.hk>

This module contains methods for reading and working with word association files
using the IGraph library.

Please not that all constructers expect the respective pajek files available at
http://vlado.fmf.uni-lj.si/pub/networks/data/dic/eat/Eat.htm for EAT
and http://vlado.fmf.uni-lj.si/pub/networks/data/dic/fa/FreeAssoc.htm for USF.
Also note that you may need to modify the pajek file headers to work with IGraph
'''
from __future__ import print_function, division #For Python 2.7 compatibility
from math import log10
from igraph import Graph, OUT, InternalError
import sys

#TODO: consts for "name", "id", "weight", and "-log weight"

class AssociationIGraph:
    #eat_f = eat.graph.shortest_paths(eat_word, eat_token, weights="-log weight")[0][0]
    
    def get_all_associations(self):
        """
            Return all word pairs and their association strength
            
            :returns: An edge view representing all word pairs and their strengths (effectively a list of (stim, resp, weight) tuples)
            :rtype: Tuple[str, str, float]
        """
        return
    
    def get_association_strengths(self, word_pairs):
        """
            Get association strengths between word pairs.
            
            Association strengths are equal to the maximal path-product between
            nodes representing the word pairs. I.e. if edge weights represent
            the probability of moving from U to V, we find the path between
            U and V which maximizes this probability.
            
            :param word_pairs: the word pairs to calculate strengths for
            :type word_pairs: Iterable[Tuple[str, str]]
            
            :returns: the calculated strengths
            :rtype: List[float]
        """
        
        word_pairs = [(s.upper(), t.upper()) for s, t in word_pairs] #node names are all uppercase
    
        sources = set()
        targets = set()
        vocab = set(self.graph.vs["name"])
        for s, t in word_pairs:
            if s in vocab:
                sources.add(s)
            if t in vocab:
                targets.add(t)
            
        sources = list(sources) #enforce consistent order
        targets = list(targets)
        s_map = {s:i for i, s in enumerate(sources)} #get indexes
        t_map = {t:i for i, t in enumerate(targets)}
        
        #For large numbers of pairs, it's quicker to get all the paths at once than to get them one at a time
        matrix = self.graph.shortest_paths(sources, targets, weights="-log weight", mode=OUT)
        
        strengths = []
        for s, t in word_pairs:
            neg_log_dist = matrix[s_map[s]][t_map[t]] if (s in s_map and t in t_map) else float("inf")
            strengths.append(10**-neg_log_dist)
        
        return strengths
        
        
        
        
        
#         strengths = []
#         for a, b in word_pairs:
#             strength = 0
#             try:
#                 #Finding the shortest path between nodes according to -log weight isequivalent to finding the maximal product path
#                 #i.e. the path which maximizes the chain probability
#                 #If no path exists, shortest_paths returns inf, and strength becomes 0
#                 strength = 10.0 ** -(self.graph.shortest_paths(a.upper(), b.upper(), weights="-log weight", mode=OUT)[0][0])
#             except (InternalError, ValueError):
#                 #one of the words is out-of-vocabulary
#                 #fail silent and default to 0
#                 pass
#             
#             strengths.append(strength)
#         
#         return strengths

class EATIGraph(AssociationIGraph):
    def __init__(self, eat_pajek_loc):
        #igraph
        self.graph = Graph.Read_Pajek(eat_pajek_loc)
        self.graph.vs["name"] = self.graph.vs["id"]#work around for bug: https://github.com/igraph/python-igraph/issues/86
        proportions = []
        neg_log_proportions = []
        out_degrees = self.graph.strength(self.graph.vs, mode=OUT, weights="weight")
        for e in self.graph.es:
            proportion = float(e["weight"])/out_degrees[e.source]
            proportions.append(proportion)
            neg_log_proportions.append(-log10(proportion))
         
        self.graph.es["weight"] = proportions
        self.graph.es["-log weight"] = neg_log_proportions

class USFIGraph(AssociationIGraph):
    def __init__(self, usf_pajek_loc):
        #igraph
        try:
            self.graph = Graph.Read_Pajek(usf_pajek_loc)
        except Exception as e:
#             import sys;sys.path.append(r'/mnt/c/Users/Andrew/.p2/pool/plugins/org.python.pydev_6.2.0.201711281614/pysrc')
#             import pydevd;pydevd.settrace(stdoutToServer=True, stderrToServer=True)
#             print(e)
            pass
        self.graph.vs["name"] = self.graph.vs["id"]#work around for bug: https://github.com/igraph/python-igraph/issues/86
        neg_log_proportions = []
        for e in self.graph.es:
            neg_log_proportions.append(-log10(e["weight"]))
         
        self.graph.es["-log weight"] = neg_log_proportions
        

def  iGraphFromTuples(association_tuples):
    """
    Creates an AssociationIGraph from a list of strength tuples
    
    :param: association_tuples: list of association tuples of form ((stim, resp), stren)
    :type: association_tuples: List[Tuple[Tuple[str,str], float]]
    
    :return: iGraph of the association tuples
    :rtype: AssociationIGraph
    """
    
#     #get unique words
#     vocab = set()
#     uppercase_tuples = []
#     for (s,r), stren in association_tuples:
#         uppercase_tuples.append((s.upper(), r.upper(), stren))
#         vocab.update(word_pair)
    
#     vocab = list(vocab) #convert to ordered list
#     
#     
#     graph = Graph(len(vocab), directed=True)
#     graph.vs["name"] = vocab #set vertex names
#     edges, _ = zip(*association_tuples)
#     graph.add_edges(edges)
    #association_tuples = [(s.upper(),r.upper(),stren) for (s,r), stren in association_tuples]
    association_tuples = [(s,r,stren) for (s,r), stren in association_tuples]
    graph = Graph.TupleList(association_tuples, directed=True, weights=True)
    
    graph.vs["id"] = graph.vs["name"]
    
    #add weights
#     for s, r , stren in association_tuples:
#         graph[(s,r)] = stren
    neg_log_proportions = []
    for e in graph.es:
        neg_log_proportions.append(-log10(e["weight"]))
     
    graph.es["-log weight"] = neg_log_proportions
    
    assoc_object = AssociationIGraph()
    assoc_object.graph = graph
    return assoc_object
    

def main():
    USF = "usf"
    EAT = "eat"
    
#     import sys;sys.path.append(r'/mnt/c/Users/Andrew/.p2/pool/plugins/org.python.pydev_6.2.0.201711281614/pysrc')
#     import pydevd;pydevd.settrace(stdoutToServer=True, stderrToServer=True)
    
    try:
        dataset = sys.argv[1]
        pajek_loc = sys.argv[2]
    except IndexError:
        raise Exception("Invalid input format. Please call as 'python query_igraph.py <dataset> <pajek loc>")
    
    graph = None
    dataset = dataset.lower()
    if dataset == EAT:
        graph = EATIGraph(pajek_loc)
    elif dataset == USF:
        graph = USFIGraph(pajek_loc)
    
    #we have a lot of word pairs and we have reason to believe it's quicker to get the distances between a bunch of pairs at once instead of calling them individually
    #therefore, save a list of unique sources and targets, and get the distances between all combinations there of (even though a lot of thos distances will be ignored)    
    vocab = set(graph.graph.vs["name"])
    
    word_pairs = []
    sources = set()
    targets = set()
#     with open("/mnt/d/temp/tmptbpj6u9o/word_pairs", "r") as f:
    for line in sys.stdin:
        a,b = line.split("\t")
        a, b = a.strip().upper(), b.strip().upper()
        word_pairs.append((a,b))
        if a in vocab:
            sources.add(a)
        if b in vocab:
            targets.add(b)
        
    sources = list(sources)
    targets=list(targets)
    
#     sys.path.append(r'/mnt/c/Users/Andrew/.p2/pool/plugins/org.python.pydev_6.2.0.201711281614/pysrc') #import sys;
#     import pydevd;pydevd.settrace(stdoutToServer=True, stderrToServer=True)
    
#     for strength in strengths:
#     strengths = graph.get_association_strengths([(a.strip(),b.strip())])[0]
    matrix = graph.graph.shortest_paths(sources, targets, weights="-log weight", mode=OUT)
    
    s_map = {s:i for i, s in enumerate(sources)}
    t_map = {t:i for i, t in enumerate(targets)}
    
    for a,b in word_pairs:
        dist = matrix[s_map[a]][t_map[b]] if (a in s_map and b in t_map) else float("inf")
        print(10**-dist)

if __name__=="__main__":
#     main()
    
    from word_associations.association_readers.xml_readers import SWoW_Dataset,SWoW_Strengths_Dataset
    swow_all = SWoW_Dataset("D:/datasets/SWoW/SWOW-EN.complete.csv").get_all_associations()
    swow_all_graph = iGraphFromTuples(swow_all)
    swow_all_graph.graph.write_pajek("D:/datasets/SWoW/swow_all.net")
    
    swow_100 = SWoW_Dataset("D:/datasets/SWoW/SWOW-EN.R100.csv",complete=False).get_all_associations()
    swow_100_graph = iGraphFromTuples(swow_100)
    swow_100_graph.graph.write_pajek("D:/datasets/SWoW/swow_100.net")
    
    swow_stren = SWoW_Strengths_Dataset("D:/datasets/SWoW/strength.SWOW-EN.R123.csv").get_all_associations()
    swow_stren_graph = iGraphFromTuples(swow_stren)
    swow_stren_graph.graph.write_pajek("D:/datasets/SWoW/swow_stren.net")

#     from time import strftime
#     import pickle
#     import numpy as np
#       
#     eat_loc = "../../Data/eat/pajek/EATnew2.net"
#     usf_loc = "../../Data/PairsFSG2.net"
#       
#     print("{} loading EAT".format(strftime("%y-%m-%d_%H:%M:%S")))
#     g = EATIGraph(eat_loc)
#     name_map = {(name, i) for i, name in enumerate(g.graph.vs["name"])}
#     print("{} Writing name map to disk".format(strftime("%y-%m-%d_%H:%M:%S")))
#     with open("eat_name_map.pkl", "wb") as f:
#         pickle.dump(name_map, f)
#     del name_map
#     print("{} starting EAT paths".format(strftime("%y-%m-%d_%H:%M:%S")))
#     dist_matrix = g.graph.shortest_paths(weights="-log weight", mode=OUT)
#     print("{} EAT paths finished".format(strftime("%y-%m-%d_%H:%M:%S")))
#     del g
#     with open("eat_raw.txt", "w") as f:
#         for row in dist_matrix:
#             f.write(" ".join(str(i) for i in row))
#             f.write("\n") 
#     del dist_matrix
#     dist_matrix = np.loadtxt("eat_raw.txt")
# #     dist_matrix = np.array(dist_matrix)
#     prob_matrix = 10.0 ** -dist_matrix
#     del dist_matrix
#     print("{} Writing to disk".format(strftime("%y-%m-%d_%H:%M:%S")))
#     np.save("eat_prob_matrix", prob_matrix)
#     print("{} EAT done\n".format(strftime("%y-%m-%d_%H:%M:%S")))
#     del prob_matrix
#     
#     print("{} loading USF".format(strftime("%y-%m-%d_%H:%M:%S")))
#     g = USFIGraph(usf_loc)
#     name_map = {(name, i) for i, name in enumerate(g.graph.vs["name"])}
#     print("{} Writing name map to disk".format(strftime("%y-%m-%d_%H:%M:%S")))
#     with open("usf_name_map.pkl", "wb") as f:
#         pickle.dump(name_map, f)
#     del name_map
#     print("{} starting USF paths".format(strftime("%y-%m-%d_%H:%M:%S")))
#     dist_matrix = g.graph.shortest_paths(weights="-log weight", mode=OUT)
#     print("{} USF paths finished".format(strftime("%y-%m-%d_%H:%M:%S")))
#     del g
#     dist_matrix = np.array(dist_matrix)
#     prob_matrix = 10.0 ** -dist_matrix
#     del dist_matrix
#     print("{} Writing to disk".format(strftime("%y-%m-%d_%H:%M:%S")))
#     np.save("usf_prob_matrix", prob_matrix)
#     print("{} USF done\n".format(strftime("%y-%m-%d_%H:%M:%S")))