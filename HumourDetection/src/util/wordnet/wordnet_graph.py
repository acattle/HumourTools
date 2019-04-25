'''
Created on Jan 9, 2017

@author: Andrew
'''
from __future__ import print_function #for Python 2.7 compatibility
from nltk.corpus import wordnet
from functools import lru_cache
import networkx

_cache_size=None #Size of LRU cache. None means no limit

class WordNetGraph(object):
    def __init__(self, cache=True):
        self.graph = networkx.Graph()
        for synset in wordnet.all_synsets():
            self._addAllRelations(synset)
    
    def _addRelations(self, synset_name, relation):
        for related_synset in relation:
            related_synset_name = related_synset.name()
            if not self.graph.has_node(related_synset_name):
                self.graph.add_node(related_synset_name) #duplicates will be ignored
            if self.graph.has_edge(synset_name, related_synset_name) or self.graph.has_edge(related_synset_name, synset_name):
                self.graph[synset_name][related_synset_name]['weight'] += 1
            else:
                self.graph.add_edge(synset_name, related_synset_name, weight=1)
            
    def _addAllRelations(self,synset):
        name = synset.name()
        if not self.graph.has_node(name):
            self.graph.add_node(name) #duplicates will be ignored
        
        #most relations are symmetrical so we don't need to add both
        #hypernym/hyponyms
        self._addRelations(name, synset.hypernyms())
        self._addRelations(name, synset.instance_hypernyms())
        #holonym/meronym
        self._addRelations(name, synset.member_holonyms())
        self._addRelations(name, synset.substance_holonyms())
        self._addRelations(name, synset.part_holonyms())
        
        #TODO: what about topic/region/usage domains?
        #TODO: what about attributes/entailments?
        #TODO: what about lemma antonyms?
    
    @lru_cache(maxsize=_cache_size)
    def _get_neighbours(self,synset):
        return self.graph[synset].keys()
    
    def _get_extended_neighbours(self, center_nodes, radius=3):
        """
            Wrapper for self._get_extended_neighbours_cached(). Converts
            center_nodes from unhashable lists to hashable tuples.
            
            Get all the nodes within the specified radius of the central nodes
            
            :param center_nodes: starting nodes for building the extended neighbourhood
            :type center_nodes: list(str)
            :param radius: the maximum radius to search for neighbours
            :type radius: int
            
            :return: set of nodes within radius of center_nodes
            :rtype: set(str)
        """
        return self._get_extended_neighbours_cached(tuple(center_nodes), radius)
        
    @lru_cache(maxsize=_cache_size)
    def _get_extended_neighbours_cached(self, center_nodes, radius=3):
        """
            Get all the nodes within the specified radius of the central nodes
            
            :param center_nodes: starting nodes for building the extended neighbourhood
            :type center_nodes: list(str)
            :param radius: the maximum radius to search for neighbours
            :type radius: int
            
            :return: set of nodes within radius of center_nodes
            :rtype: set(str)
        """
        #TODO: include/omit center?
        #TODO: handle edge weights?
        
        extended_neighbouhood = set(center_nodes)
        processed_nodes = set() #set of already visited nodes
        for _ in range(radius):
            for synset in (extended_neighbouhood-processed_nodes): #For any node we haven't visited yet
                #add all synsets with and edge connecting it to this one
                #TODO: deal with 0 weight edges?
                neighbours = self._get_neighbours(synset)
                
                extended_neighbouhood.update(neighbours) 
                processed_nodes.add(synset)
        
        return extended_neighbouhood
    
    def get_directional_relativity(self,synsets1, synsets2, radius=3):
        dir_rel = 0.0 #default to 0.0
            
        synsets1_neighbours = self._get_extended_neighbours(synsets1, radius)
        synsets2_neighbours = self._get_extended_neighbours(synsets2, radius)
            
        intersection = synsets1_neighbours & synsets2_neighbours
         
        if len(synsets1_neighbours) != 0: #check for divide by zero error
            dir_rel = float(len(intersection))/len(synsets1_neighbours) #assign the actual value
        
        return dir_rel
                    

def invert_weights(graph):
    max_weight = -float("inf")
    
    for _, _, data in graph.edges(data=True):
        w = data["weight"]
        
        if w > max_weight:
            max_weight = w
    
    for u, v, data in graph.edges(data=True):
        graph[u][v]["inv weight"] = max_weight - data["weight"]
    
    return graph
    


if __name__ == '__main__':
#     wg = WordNetGraph()
#     with open("wordnet_graph.pkl", "wb") as wordnet_pickle:
#         wg = pickle.dump(wg, wordnet_pickle, protocol=2) #protocol 2 for python2 compatibility

#     with open("wordnet_graph.pkl", "rb") as wordnet_pickle:
#         wg = pickle.load(wordnet_pickle)
     
    from word_associations.association_readers import USFGraph
    usf = USFGraph("../Data/PairsFSG2.net")
    usf_associations=usf.get_all_associations()
    del usf
    from timeit import timeit
    association_tuples = [ ([s.name() for s in wordnet.synsets(stimuli.lower())], [s.name() for s in wordnet.synsets(response.lower())]) for stimuli, response, _ in usf_associations]#[:100]
    
    wg = WordNetGraph(cache=True)
    print(timeit("[wg.get_directional_relativity(a,b) for a, b in association_tuples]", "from __main__ import association_tuples,wg", number=1))
    wg = WordNetGraph(cache=False)
    print(timeit("[wg.get_directional_relativity(a,b) for a, b in association_tuples]", "from __main__ import association_tuples,wg", number=1))
    
     
#     wg.graph = invert_weights(wg.graph)
#     with open("wordnet_graph_w_inverted_weights.pkl", "wb") as wordnet_file:
#         pickle.dump(wg, wordnet_file)
#         
#     load = networkx.algorithms.centrality.load_centrality(wg.graph, weight="weight")
#     with open("wordnet_load_weighted.pkl", "wb") as load_file:
#         pickle.dump(load,load_file)
#         
#     betweenness = networkx.algorithms.centrality.betweenness_centrality(wg.graph, weight="weight")
#     with open("wordnet_betweenness_weighted.pkl", "wb") as betweenness_file:
#         pickle.dump(betweenness, betweenness_file)
# 
#     with open("features/usf/dirrel.pkl", "rb") as f:
#         dr = pickle.load(f)
#     with open("features/usf/word_pairs.pkl", "rb") as f:
#         wp=pickle.load(f)
#     with open("wordnet_graph.pkl", "rb") as f:
#         wg=pickle.load(f)
#     
#     import matplotlib.pyplot as plt
#     
#     for i in [45208]:#range(len(wp)):
#         w1, w2 = wp[i]
#         
#         s1 = []
#         s2=[]
#         for s in wordnet.synsets(w1):
#             s1.append(s.name())
#         for s in wordnet.synsets(w2):
#             s2.append(s.name())
#         
#         wg.get_directional_relativity(s1,s2)
#         
#         #9121 decency modesty
#         #10039 adjective adverb
#         #34533 weekly monthly
#         #35026 monthly weekly
#         #45208 rumor hearsay
#         #53489 pint gallon
#         #57138 tonight now
#         #67018 quart gallon



#leftover visualization code
#             
#             if (len(synsets1_neighbours|synsets2_neighbours) < 50) and (dir_rel > 0) and (dir_rel < 1) and ("n" in [s.split(".")[1] for s in synsets1]) and ("n" in [s.split(".")[1] for s in synsets2]) and (len(synsets2)>1) and (len(synsets1)>1):
#                 print("found one")
#                 
#                 plt.clf()
#                 sg = self.graph.subgraph(synsets1_neighbours|synsets2_neighbours)
#                 l=networkx.spring_layout(sg)
#                 networkx.draw_networkx(sg, l)
#                 networkx.draw_networkx_nodes(sg,l, synsets1_neighbours, node_color="#cd5c5c")#node_size, node_color, node_shape, alpha, cmap, vmin, vmax, ax, linewidths, label)
#                 networkx.draw_networkx_nodes(sg,l, synsets2_neighbours, node_color="#4169e1")#node_size, node_color, node_shape, alpha, cmap, vmin, vmax, ax, linewidths, label)
#                 networkx.draw_networkx_nodes(sg,l, synsets1_neighbours&synsets2_neighbours, node_color="#9932cc")#node_size, node_color, node_shape, alpha, cmap, vmin, vmax, ax, linewidths, label)
#                 networkx.draw_networkx_nodes(sg,l, synsets1, node_color="#800000")#node_size, node_color, node_shape, alpha, cmap, vmin, vmax, ax, linewidths, label)
#                 networkx.draw_networkx_nodes(sg,l, synsets2, node_color="#0000cd")#node_size, node_color, node_shape, alpha, cmap, vmin, vmax, ax, linewidths, label)
#                 plt.show(