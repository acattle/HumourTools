'''
Created on Aug 19, 2016

@author: Andrew
'''
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):
    #https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
    if treebank_tag.startswith('j'):
        return wordnet.ADJ
    elif treebank_tag.startswith('v'):
        return wordnet.VERB
    elif treebank_tag.startswith('n'):
        return wordnet.NOUN
    elif treebank_tag.startswith('r'):
        return wordnet.ADV
    else:
        return 'n' #default to Noun