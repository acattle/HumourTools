'''
Created on Feb 7, 2018

@author: Andrew
'''
from __future__ import print_function, division #For Python 2.7 compatibility
from math import log10
from igraph import Graph, OUT, InternalError
import sys


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
        
        strengths = []
        for a, b in word_pairs:
            strength = 0
            try:
                #Finding the shortest path between nodes according to -log weight isequivalent to finding the maximal product path
                #i.e. the path which maximizes the chain probability
                #If no path exists, shortest_paths returns inf, and strength becomes 0
                strength = 10.0 ** -(self.graph.shortest_paths(a.upper(), b.upper(), weights="-log weight", mode=OUT)[0][0])
            except (InternalError, ValueError):
                #one of the words is out-of-vocabulary
                #fail silent and default to 0
                pass
            
            strengths.append(strength)
        
        return strengths

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
        self.graph = Graph.Read_Pajek(usf_pajek_loc)
        self.graph.vs["name"] = self.graph.vs["id"]#work around for bug: https://github.com/igraph/python-igraph/issues/86
        neg_log_proportions = []
        for e in self.graph.es:
            neg_log_proportions.append(-log10(e["weight"]))
         
        self.graph.es["-log weight"] = neg_log_proportions

def main():
    USF = "usf"
    EAT = "eat"
    
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
#
#     for strength in strengths:
#     strengths = graph.get_association_strengths([(a.strip(),b.strip())])[0]
    matrix = graph.graph.shortest_paths(sources, targets, weights="-log weight", mode=OUT)
    
    s_map = {(s,i) for i, s in enumerate(sources)}
    t_map = {(t,i) for i, t in enumerate(targets)}
    
    for a,b in word_pairs:
        dist = matrix[s_map[a]][t_map[b]] if (a in s_map and b in t_map) else float("inf")
        print(10**-dist)

if __name__=="__main__":
    main()
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