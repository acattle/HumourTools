'''
Created on Jan 27, 2017

@author: Andrew
'''
from numpy import zeros, hstack
from math import log, isinf
<<<<<<< HEAD
from six import iteritems
=======
from nltk.compat import iteritems
>>>>>>> refs/remotes/origin/master

_pos = ['a',
        's',
        'r',
        'n',
        'v']

_pos_index = dict(zip(_pos, range(len(_pos))))

_lexnames = ["adj.all",
             "adj.pert",
             "adv.all",
             "noun.Tops",
             "noun.act",
             "noun.animal",
             "noun.artifact",
             "noun.attribute",
             "noun.body",
             "noun.cognition",
             "noun.communication",
             "noun.event",
             "noun.feeling",
             "noun.food",
             "noun.group",
             "noun.location",
             "noun.motive",
             "noun.object",
             "noun.person",
             "noun.phenomenon",
             "noun.plant",
             "noun.possession",
             "noun.process",
             "noun.quantity",
             "noun.relation",
             "noun.shape",
             "noun.state",
             "noun.substance",
             "noun.time",
             "verb.body",
             "verb.change",
             "verb.cognition",
             "verb.communication",
             "verb.competition",
             "verb.consumption",
             "verb.contact",
             "verb.creation",
             "verb.emotion",
             "verb.motion",
             "verb.perception",
             "verb.possession",
             "verb.social",
             "verb.stative",
             "verb.weather",
             "adj.ppl"]

_lexname_index = dict(zip(_lexnames, range(len(_lexnames))))

class WordNetUtils(object):
    def __init__(self, cache=True):
        self._dist_cache=None
        self._depth_cache=None
        if cache:
            self._dist_cache={}
            self._depth_cache={}
    
    def _min_depth(self, synset):
        depth = None
        if (self._depth_cache != None) and (synset.name() in self._depth_cache):
            depth = self._depth_cache[synset.name()]
        else:
            depth = synset.min_depth()
            if (self._depth_cache !=None):
                self._depth_cache[synset.name()] = depth
        
        return depth

    def _shortest_path_distance(self, synset1, synset2, simulate_root=False):
        if synset1 == synset2:
            return 0

        dist_dict1 = None
        if (self._dist_cache != None) and (synset1.name() in self._dist_cache):
            dist_dict1 = self._dist_cache[synset1.name()]
        else:
            dist_dict1 = synset1._shortest_hypernym_paths(simulate_root)
            if (self._dist_cache !=None):
                self._dist_cache[synset1.name()] = dist_dict1
                
        dist_dict2 = None
        if (self._dist_cache != None) and (synset2.name() in self._dist_cache):
            dist_dict2 = self._dist_cache[synset2.name()]
        else:
            dist_dict2 = synset2._shortest_hypernym_paths(simulate_root)
            if (self._dist_cache !=None):
                self._dist_cache[synset2.name()] = dist_dict2

        # For each ancestor synset common to both subject synsets, find the
        # connecting path length. Return the shortest of these.

        inf = float('inf')
        path_distance = self._min_depth(synset1) + self._min_depth(synset2) + 2 #+2 for going to virtual root and coming back
        for synset, d1 in iteritems(dist_dict1):
            d2 = dist_dict2.get(synset, inf)
            path_distance = min(path_distance, d1 + d2)

        return path_distance
    
    def get_lex_vector(self,synsets):
        pos_vector = zeros(len(_pos))
        lex_vector = zeros(len(_lexnames))
        for synset in synsets:
            pos_vector[_pos_index[synset.pos()]] += 1    
            lex_vector[_lexname_index[synset.lexname()]] += 1
        
        return hstack([pos_vector, lex_vector])

    def path_similarity_w_root(self,synset1, synset2):
        distance = self._shortest_path_distance(synset1, synset2, simulate_root=True)
        if distance is None or distance < 0:
            return None
        return 1.0 / (distance + 1)
        
    def modified_lch_similarity_w_root(self,synset1, synset2):
        """
        Modified LCH similarity which is capable of comparing different POS
        """
        if synset1._pos not in synset1._wordnet_corpus_reader._max_depth:
            synset1._wordnet_corpus_reader._compute_max_depth(synset1._pos, True)
        if synset2._pos not in synset1._wordnet_corpus_reader._max_depth:
            synset1._wordnet_corpus_reader._compute_max_depth(synset2._pos, True)
    
        depth = max(synset1._wordnet_corpus_reader._max_depth[synset1._pos], synset1._wordnet_corpus_reader._max_depth[synset2._pos])
    
        distance = self._shortest_path_distance(synset1, synset2, simulate_root=True)
    
        if distance is None or distance < 0 or depth == 0:
            return None
            
        return -log((distance + 1) / (2.0 * depth))
    
    def wup_similarity_w_root(self,synset1, synset2):
        wup = 0.0
        if synset1 and synset2:
            subsumers = synset1.lowest_common_hypernyms(synset2, use_min_depth=True)
        
            # +1s account for simulated root
            depth = 1
            len1 = self._min_depth(synset1) + 1
            len2 = self._min_depth(synset2) + 1
            # If no LCS was found
            if len(subsumers) > 0:
                subsumer = subsumers[0]
            
                # +2 instead of +1 to account for root node
                depth = subsumer.max_depth() + 2
            
                len1 = self._shortest_path_distance(synset1, subsumer)
                len2 = self._shortest_path_distance(synset2, subsumer)
                if len1 is None or len2 is None:
                    return None #TODO: raise error?
                len1 += depth
                len2 += depth
            
            wup = (2.0 * depth) / (len1 + len2)
        
        return wup