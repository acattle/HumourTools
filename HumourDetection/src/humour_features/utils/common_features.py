'''
    Created on Dec 19, 2017

    :author: Andrew Cattle <acattle@cse.ust.hk>

    This module provides utility functions for extracting features which appear
    in multiple humour recognition papers
'''
import nltk
import numpy as np

def get_alliteration_and_rhyme_features(documents):
    """
        Utility method for extracting alliteration and rhyme chain features from
        a tokenized document based on word pronunciations in the CMU Pronouncing
        Dictionary (https://github.com/cmusphinx/cmudict).
        
        Alliteration and rhyme are both calculated naively. If a word has
        multiple pronunciations, all pronunciations will be considered equally.
        Alliteration is limited to the first phoneme only. Rhyme is limited to
        final vowel phoneme + trailing codas.
        
        :param documents: documents to be processed. Each document is a sequence of tokens
        :type documents: list(list(str))
        
        :return: a matrix where columns represent extracted phonetic style features in the form (alliteration_num, alliteration_len, rhyme_num, rhyme_len) and rows are documents
        :rtype: numpy.array
    """
    
    cmu = nltk.corpus.cmudict.dict() #This is slow and will get called each time the function is called (training and test)
    
    feature_vects = []
    
    for document in documents:
    
        alliteration_chains = {} #will hold alliteration chains in the form {first_phoneme:count}
        rhyme_chains = {}
        for word in document:
            if word in cmu:
                pronunciations = cmu[word]
                
                first_phonemes = set()
                end_rhymes = set()
                for pronunciation in pronunciations:
                    #Since there's no easy way to identify with pronunciation is the correct one
                    #It's just easier to not worry about it and double count all possible first phonemes and end rhymes
                    
                    first_phonemes.add(pronunciation[0])
                    
                    i=0
                    for i, phoneme in reversed(list(enumerate(pronunciation))):
                        #go backwards through the pronunciation
                        if phoneme[0] in "AEIOU": #until we find the final vowel
                            break #then stop looking but make note of the vowel's index
                        
                    end_rhymes.add("".join(pronunciation[i:])) #concatenate the final vowel with any codas
                
                for first_phoneme in first_phonemes:
                    #iterate count for each possible first phoneme
                    alliteration_chains[first_phoneme] = alliteration_chains.get(first_phoneme,0) + 1
                for end_rhyme in end_rhymes:
                    #iterate count for each possible first phoneme
                    rhyme_chains[end_rhyme] = rhyme_chains.get(end_rhyme, 0) + 1
                    
        #trim chains of length 1
        alliteration_chains = {phoneme:count for phoneme, count in alliteration_chains.items() if count > 1}
        rhyme_chains = {rhyme:count for rhyme, count in rhyme_chains.items() if count > 1}
        
        alliteration_num = len(alliteration_chains) #number of chains
        max_alliteration_len = max(alliteration_chains.values()) if alliteration_num > 0 else 0#length of longest chain
        rhyme_num = len(rhyme_chains)
        max_rhyme_len = max(rhyme_chains.values()) if rhyme_num > 0 else 0 #TODO: else 1?
        
        feature_vects.append((alliteration_num, max_alliteration_len, rhyme_num, max_rhyme_len))
    
    return np.vstack(feature_vects)