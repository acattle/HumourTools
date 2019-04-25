'''
Created on Jan 27, 2017

@author: Andrew
'''
import numpy as np
import math
from six import iteritems
from functools import lru_cache
from nltk.corpus import wordnet as wn

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




############################## Cached function wrappers ##############################

#controls size of lru_cache
#Larger caches mean faster function calls but more memory usage
#_cache_sie of None means no limit on cache size
_cache_size = None

@lru_cache(maxsize=_cache_size)
def get_synsets(word, pos=None):
    """
    Cached wrapper for NLTK's wordnet.synsets()
    
    :param word: word to retrieve synsets for
    :type word: str
    :param pos: POS of retrieved synsets. If None, all POS will be considered
    :type: str
    
    :returns: All sysnets for word of POS pos
    :rtype: List[Synset]
    """
    return wn.synsets(word,pos)

@lru_cache(maxsize=_cache_size)
def _shortest_hypernym_paths(synset, simulate_root=False):
    """
    Cached wrapper for NLTK's Synset._shortest_hypernym_paths()
    """
    return synset._shortest_hypernym_paths(simulate_root)

@lru_cache(maxsize=_cache_size) #TODO: does that actaully help much if we're already caching _shortest_hypernym
def _shortest_path_distance(synset1, synset2, simulate_root=False):
    """
    Modified version of NLTK's Synset.shortest_path_distance()
    capable of simulating a common root that connects all POS trees.
    
    Results are cached according to an LRU cache.
    
    :param synset1: The first Synset to which the shortest path will be found.
    :type synset1: Synset
    :param synset2: The second Synset to which the shortest path will be found.
    :type synset2: Synset
    :param simulate_root: Specifies whether a virtual root node should be inserted that connects the various POS trees
    :type simulate_root: bool

    :return: The number of edges in the shortest path connecting the two
        nodes, or None if no path exists.
    """
    if synset1 == synset2:
        return 0

    dist_dict1 = _shortest_hypernym_paths(synset1, simulate_root)            
    dist_dict2 = _shortest_hypernym_paths(synset2, simulate_root)
    
    # For each ancestor synset common to both subject synsets, find the
    # connecting path length. Return the shortest of these.

    inf = float('inf')
    path_distance = inf
    if simulate_root: #if we want to simulate a common root
        path_distance = synset1.min_depth() + synset2.min_depth()
    
    for synset, d1 in iteritems(dist_dict1):
        d2 = dist_dict2.get(synset, inf)
        path_distance = min(path_distance, d1 + d2)

    return None if math.isinf(path_distance) else path_distance




############################## Modified functions ##############################
def get_lex_vector(synsets):
    return _get_lex_vector(tuple(synsets))

@lru_cache(_cache_size)
def _get_lex_vector(synsets):
    pos_vector = np.zeros(len(_pos))
    lex_vector = np.zeros(len(_lexnames))
    for synset in synsets:
        pos_vector[_pos_index[synset.pos()]] += 1    
        lex_vector[_lexname_index[synset.lexname()]] += 1
    
    return np.hstack([pos_vector, lex_vector])

def path_similarity(synset1, synset2,simulate_root=False):
    """
    Modified version of NLTK's wordnet.path_similarity() capable of comparing
    different POS by simulating a global root. Utilizes caching to increase
    performance.
    
    :param synset1: The first Synset to compare.
    :type synset1: Synset
    :param synset2: The second Synset to compare.
    :type synset2: Synset
    :param simulate_root: Specifies whether a virtual root node should be inserted that connects the various POS trees
    :type simulate_root: bool

    :return: The path similarity of synset1 and synset2, or None if no path exists.
    :rtype: float
    """
    distance = _shortest_path_distance(synset1, synset2, simulate_root=simulate_root)
    if distance is None or distance < 0:
        return None
    return 1.0 / (distance + 1)
    
def lch_similarity(synset1, synset2, simulate_root=False):
    """
    Modified version of NLTK's wordnet.lch_similarity() capable of comparing
    different POS by simulating a global root. Utilizes caching to increase
    performance.
    
    :param synset1: The first Synset to compare.
    :type synset1: Synset
    :param synset2: The second Synset to compare.
    :type synset2: Synset
    :param simulate_root: Specifies whether a virtual root node should be inserted that connects the various POS trees
    :type simulate_root: bool

    :return: The LCH similarity of synset1 and synset2, or None if no path exists.
    :rtype: float
    """
    if synset1._pos not in synset1._wordnet_corpus_reader._max_depth:
        synset1._wordnet_corpus_reader._compute_max_depth(synset1._pos, True)
    if synset2._pos not in synset1._wordnet_corpus_reader._max_depth:
        synset2._wordnet_corpus_reader._compute_max_depth(synset2._pos, True)

    depth = max(synset1._wordnet_corpus_reader._max_depth[synset1._pos], synset1._wordnet_corpus_reader._max_depth[synset2._pos])

    distance = _shortest_path_distance(synset1, synset2, simulate_root=simulate_root)

    if distance is None or distance < 0 or depth == 0:
        return None
        
    return -math.log((distance + 1) / (2.0 * depth))

def wup_similarity(synset1, synset2, simulate_root=False):
    """
    Modified version of NLTK's wordnet.wup_similarity() capable of comparing
    different POS by simulating a global root. Utilizes caching to increase
    performance.
    
    :param synset1: The first Synset to compare.
    :type synset1: Synset
    :param synset2: The second Synset to compare.
    :type synset2: Synset
    :param simulate_root: Specifies whether a virtual root node should be inserted that connects the various POS trees
    :type simulate_root: bool

    :return: The WUP similarity of synset1 and synset2, or None if no path exists.
    :rtype: float
    """
    wup = 0.0
    if synset1 and synset2:
        #TODO: Would adding a cached wrapper to this function be worth it?
        subsumers = synset1.lowest_common_hypernyms(synset2, use_min_depth=True)
    
        # +1s account for simulated root
        depth = 1
        len1 = synset1.min_depth() + 1
        len2 = synset2.min_depth() + 1
        # If no LCS was found
        if len(subsumers) > 0:
            subsumer = subsumers[0]
        
            depth = subsumer.max_depth() + 1
            if simulate_root:
                depth += 1 #add root to depth
        
            len1 = _shortest_path_distance(synset1, subsumer, simulate_root)
            len2 = _shortest_path_distance(synset2, subsumer, simulate_root)
            if len1 is None or len2 is None:
                return None #TODO: raise error?
            len1 += depth
            len2 += depth
        
        wup = (2.0 * depth) / (len1 + len2)
    
    return wup