'''
    Created on Dec 19, 2017

    :author: Andrew Cattle <acattle@connect.ust.hk>

    This module provides utility functions for extracting features which appear
    in multiple humour recognition papers
'''
from __future__ import division #for maintaining Python 2.7 support
import numpy as np
from nltk.corpus import cmudict
from util.misc import mean
from util.text_processing import get_word_pairs

def get_interword_score_features(documents, scorer, token_filter=None, pair_generator=None):
    """
        A convenience wrappper for obtaining min, max, and micro average scores
        between all word pairs in a document, according to the supplied scorer
        function.
        
        Documents can be filtered using the optional token_filter argument. E.g.
        if token_filter is a function that removes stopwords from a document
        then no stopwords will appear in the word pairs.
        
        Note: this function calls scorer for each word pair and thus is not
        suitable for scorers with large setups/teardowns
        
        :param documents: documents to be processed. Each document should be a sequence of tokens
        :type documents: Iterable[Iterable[str]]
        :param scorer: the scoring function to use when comparing words. Must take two strings as input and return a score as a float.
        :type scorer: Callable[[str, str], Number]
        :param token_filter: function for filtering the tokens in a document. If None, no token filtering will be done
        :type token_filter: Callable[[Iterable[str]], Iterable[str]]
        :param pair_generator: function for generating word pairs. Must take list of documents (as a list of tokens) and a token_filter fucntion. If None, default word pair generator will be used.
        :type pair_generator: Callable[[Iterable[str], Callable[[Iterable[str]], Iterable[str]]], List[List[Tuple[str, str]]]]
        
        :return: A matrix in the form (min_score, avg_score, max_score) x # of documents
        :rtype: numpy.array
    """
    
    if not pair_generator:
        pair_generator = get_word_pairs
    documents = pair_generator(documents, token_filter)

    feature_vectors = []
    for document in documents:
        scores = []
        for word1, word2 in document:
            #TODO: ignore OOVs? How? Failing silently on keyerrors?
            scores.append(scorer(word1, word2))
        
        max_score = 0
        avg_score = 0
        min_score = 0
        if len(scores) > 0:
            max_score = max(scores)
            avg_score = mean(scores)
            min_score = min(scores)
        
        feature_vectors.append((min_score, avg_score, max_score))
    
    return np.vstack(feature_vectors)

def get_alliteration_and_rhyme_features(documents, cmu_dict=None):
    """
        Utility method for extracting alliteration and rhyme chain features from
        a tokenized document based on word pronunciations in the CMU Pronouncing
        Dictionary (https://github.com/cmusphinx/cmudict).
        
        Alliteration and rhyme are both calculated naively. If a word has
        multiple pronunciations, all pronunciations will be considered equally.
        Alliteration is limited to the first phoneme only. Rhyme is limited to
        final vowel phoneme + trailing codas.
        
        :param documents: documents to be processed. Each document is a sequence of tokens
        :type documents: Iterable[Iterable[str]]
        
        :return: a matrix where columns represent extracted phonetic style features in the form (alliteration_num, alliteration_len, rhyme_num, rhyme_len) and rows are documents
        :rtype: numpy.array
    """
    
    if not cmu_dict:
        cmu_dict = cmudict.dict() #This is can be slow if called multiple times. Therefore we give users to option of preloading it
    
    feature_vects = []
    
    for document in documents:
    
        alliteration_chains = {} #will hold alliteration chains in the form {first_phoneme:count}
        rhyme_chains = {}
        for word in document:
            if word in cmu_dict:
                pronunciations = cmu_dict[word]
                
                first_phonemes = set()
                end_rhymes = set()
                for pronunciation in pronunciations:
                    #Since there's no easy way to identify with pronunciation is the correct one
                    #It's just easier to not worry about it and double count all possible first phonemes and end rhymes
                    #TODO: but doesn't this lead to double counting?
                    
                    first_phonemes.add(pronunciation[0])
                    
                    #TODO: Yang only looks at last vowel, not last vowel and coda
                    
                    i=0
                    for i, phoneme in reversed(list(enumerate(pronunciation))):
                        #go backwards through the pronunciation until we find the final vowel
                        if phoneme[0] in "AEIOU": #check the first letter of the phoneme to see if it's a vowel
                            break #then stop looking but make note of the vowel's index
                    
                    end_rhymes.add("".join(pronunciation[i])) #concatenate the final vowel only
                    #TODO: uncomment
#                     end_rhymes.add("".join(pronunciation[i:])) #concatenate the final vowel with any codas
                
                for first_phoneme in first_phonemes:
                    #iterate count for each possible first phoneme
                    alliteration_chains[first_phoneme] = alliteration_chains.get(first_phoneme,0) + 1
                for end_rhyme in end_rhymes:
                    #iterate count for each possible first phoneme
                    rhyme_chains[end_rhyme] = rhyme_chains.get(end_rhyme, 0) + 1
                    
        #trim chains of length 1
        alliteration_chains = {phoneme:count for phoneme, count in alliteration_chains.items() if count > 1} #TODO: any reason not to make a list of just counts?
        rhyme_chains = {rhyme:count for rhyme, count in rhyme_chains.items() if count > 1}
        
        alliteration_num = len(alliteration_chains) #number of chains
        max_alliteration_len = max(alliteration_chains.values()) if alliteration_num > 0 else 0#length of longest chain
        rhyme_num = len(rhyme_chains)
        max_rhyme_len = max(rhyme_chains.values()) if rhyme_num > 0 else 0 #TODO: else 1?
        
        feature_vects.append((alliteration_num, max_alliteration_len, rhyme_num, max_rhyme_len))
    
    return np.vstack(feature_vects)