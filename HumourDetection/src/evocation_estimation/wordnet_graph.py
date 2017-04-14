'''
Created on Jan 9, 2017

@author: Andrew
'''
from nltk.corpus import wordnet
import networkx
import pickle
from networkx.generators.ego import ego_graph
from numpy import nan

class WordNetGraph():
    def __init__(self, cache=True):
        self.graph = networkx.Graph() #graph or multigraph? multigraph allows suplicate edges
        for synset in wordnet.all_synsets():
            self._addAllRelations(synset)
        self._cache=None
        self._synset_cache = None
        if cache:
            self._cache = {}
            self._synset_cache = {}
    
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
        
        #what about topic/region/usage domains?
        #what about attributes/entailments?
        #what about lemma antonyms?
    
    def get_directional_relativity(self,synsets1, synsets2, radius=3):
        dir_rel = nan
        try:
            synsets1_str = str(synsets1)            
            synsets1_neighbours = set()
            if (self._cache != None) and (synsets1_str in self._cache):
                synsets1_neighbours = self._cache[synsets1_str]
            else:
                for synset1 in synsets1:
                    ego_nodes = []
                    if (self._synset_cache != None) and (synset1 in self._synset_cache):
                        ego_nodes = self._synset_cache[synset1]
                    else:
                        #https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.generators.ego.ego_graph.html
                        ego_nodes = ego_graph(self.graph, synset1, radius).nodes()
                        
                        if (self._synset_cache != None):
                            self._synset_cache[synset1] = ego_nodes
                            
                    synsets1_neighbours.update(ego_nodes)
                    
                if (self._cache != None):
                    self._cache[synsets1_str] = synsets1_neighbours
                
            synsets2_str = str(synsets2)
            synsets2_neighbours = set()
            if (self._cache != None) and (synsets2_str in self._cache):
                synsets2_neighbours = self._cache[synsets2_str]
            else:
                for synset2 in synsets2:
                    ego_nodes = []
                    if (self._synset_cache != None) and (synset2 in self._synset_cache):
                        ego_nodes = self._synset_cache[synset2]
                    else:
                        #https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.generators.ego.ego_graph.html
                        ego_nodes = ego_graph(self.graph, synset2, radius).nodes()
                        
                        if (self._synset_cache != None):
                            self._synset_cache[synset2] = ego_nodes
                            
                    synsets2_neighbours.update(ego_nodes)
                    
                if (self._cache != None):
                    self._cache[synsets2_str] = synsets2_neighbours
                
            intersection = synsets1_neighbours & synsets2_neighbours
            
            if len(synsets1_neighbours) != 0: #check for divide by zero error
                dir_rel = float(len(intersection))/len(synsets1_neighbours) #assign the actual value
        except KeyError:
            #one of the synsets must be none)
            pass
        
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
    wg = WordNetGraph()
    with open("wordnet_graph.pkl", "wb") as wordnet_pickle:
        wg = pickle.dump(wg, wordnet_pickle)
     
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