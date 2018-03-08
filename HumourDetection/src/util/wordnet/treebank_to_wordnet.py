'''
Created on Aug 19, 2016

@author: Andrew
'''
from nltk.corpus import wordnet as wn

def get_wordnet_pos(treebank_tag, default=wn.NOUN):
    #https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
    if treebank_tag[0].upper() == 'J':
        return wn.ADJ
    elif treebank_tag[0].upper() == 'V':
        return wn.VERB
    elif treebank_tag[0].upper() == 'N':
        return wn.NOUN
    elif treebank_tag[0].upper() == 'R':
        return wn.ADV
    else:
        return default #return default markers