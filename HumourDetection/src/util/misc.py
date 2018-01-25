'''
    Created on Jan 24, 2018
    
    @author: Andrew Cattle <acattle@cse.ust.hk>
    
    This module contains miscellaneous utility functions 
'''
from __future__ import division #maintain Python 2.7 compatibility
from nltk import word_tokenize
from string import punctuation

def default_preprocessing_and_tokenization(documents):
    """
        A method for preprocessing and tokening documents that should be good
        enough for the majority of contexts.
        
        Each document is lowercased, tokenized using nltk.word_tokenize, and
        has its punctuation removed.
        
        :param documents: the documents to be processed
        :type documents: list(str)
    """
    preprocessed_documents = []
    
    for document in documents:
        tokens = word_tokenize(document.lower()) #Tokenize and lowercase document
        
        for i in range(len(tokens)):
            tokens[i] = tokens[i].translate(None, punctuation) #remove punctuation
            #https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
            #TODO: should be doing this in place like this?
        
        preprocessed_documents.append(tokens)
    
    return preprocessed_documents

def mean(l):
    """
        Calculates the arithmetic mean of the elements in list l using built-in
        functions.
        
        Quicker than numpy.mean for short lists since l does not need to be
        converted to a numpy.array first.
        
        :param l: the list of values to be averaged
        :type l: list(float)
        
        :returns: the arithmetic mean of l
        :rtype: float
    """
    
    return sum(l)/max(len(l), 1) #use of max() is to prevent divide by 0 errors if l == []
