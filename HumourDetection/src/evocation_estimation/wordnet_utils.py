'''
Created on Jan 27, 2017

@author: Andrew
'''
from numpy import zeros, hstack

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

def get_lex_vector(synset):
    pos_vector = zeros(len(_pos))
    lex_vector = zeros(len(_lexnames))
    if synset:
        pos_vector[_pos_index[synset.pos()]] = 1    
        lex_vector[_lexname_index[synset.lexname()]] = 1
    
    return hstack([pos_vector, lex_vector])

def wup_similarity_w_root(synset1, synset2):
    wup = 0.0
    if synset1 and synset2:
        subsumers = synset1.lowest_common_hypernyms(synset2, use_min_depth=True)
    
        # +1s account for simulated root
        depth = 1
        len1 = synset1.min_depth() + 1
        len2 = synset2.min_depth() + 1
        # If no LCS was found
        if len(subsumers) > 0:
            subsumer = subsumers[0]
        
            # +2 instead of +1 to account for root node
            depth = subsumer.max_depth() + 2
        
            len1 = synset1.shortest_path_distance(subsumer)
            len2 = synset2.shortest_path_distance(subsumer)
            if len1 is None or len2 is None:
                return None #TODO: raise error?
            len1 += depth
            len2 += depth
        
        wup = (2.0 * depth) / (len1 + len2)
    
    return wup