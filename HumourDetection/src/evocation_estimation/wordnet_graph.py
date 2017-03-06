'''
Created on Jan 9, 2017

@author: Andrew
'''
from nltk.corpus import wordnet
import networkx
import pickle
from networkx.generators.ego import ego_graph

class WordNetGraph():
    def __init__(self):
        self.graph = networkx.Graph() #graph or multigraph? multigraph allows suplicate edges
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
        
        #what about topic/region/usage domains?
        #what about attributes/entailments?
        #what about lemma antonyms?
    
    def get_directional_relativity(self,synset1, synset2, radius=3):
        dir_rel = 0.0
        try:
            #https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.generators.ego.ego_graph.html
            synset1_neighbours = set(ego_graph(self.graph, synset1, radius).nodes())
            synset2_neighbours = set(ego_graph(self.graph, synset2, radius).nodes())
            intersection = synset1_neighbours & synset2_neighbours
            dir_rel = float(len(intersection))/len(synset1_neighbours)
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
    with open("wordnet_graph.pkl", "rb") as wordnet_pickle:
        wg = pickle.load(wordnet_pickle)
    
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