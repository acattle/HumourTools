'''
Created on Feb 6, 2018

@author: Andrew Cattle <acattle@cse.ust.hk>
'''
from nltk import word_tokenize
from string import punctuation
from itertools import combinations
from nltk.corpus import stopwords
import re

ENGLISH_STOPWORDS = set(stopwords.words('english'))
ENGLISH_STOPWORDS.add("n't") #added for compatibility with word_tokenize
#TODO: this set of stopwords will remove negations like "no", "not", etc. Is this what we want to do?

def default_preprocessing_and_tokenization(documents, stopwords=ENGLISH_STOPWORDS):
    """
        A method for preprocessing and tokening documents that should be good
        enough for the majority of contexts.
        
        Each document is lowercased, tokenized using nltk.word_tokenize, and
        has its punctuation removed.
        
        :param documents: the documents to be processed
        :type documents: Iterable[str]
        :param stopwords: stopwords to be removed
        :type stopwords: Iterable[str]
        
        :returns: preprocessed and tokenized document
        :rtype: List[List[str]]
    """
    punc_re = re.compile(f'[{re.escape(punctuation)}]')
    
    preprocessed_documents = []
    
    for document in documents:
        tokens = word_tokenize(document.lower()) #Tokenize and lowercase document
        #TODO: keep sentence information?
        
        processed_tokens = []
        for token in tokens:
            if token not in stopwords:
                token = punc_re.sub("", token) #remove punctuation
                #https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
                
                if token: #if this isn't an empty string
                    processed_tokens.append(token)
        
        preprocessed_documents.append(processed_tokens)
    
    return preprocessed_documents

def get_word_pairs(documents, token_filter=None):
    """
        Generate word pairs for each document.
        
        Documents can be filtered using the optional token_filter argument. E.g.
        if token_filter is a function that removes stopwords from a document
        then no stopwords will appear in the word pairs.
        
        :param documents: the documents to be processed
        :type documents: Iterable[Iterable[str]]
        :param token_filter: function for filtering the tokens in a document. If None, no token filtering will be done
        :type token_filter: Callable[[Iterable[str]], Iterable[str]]
        
        :returns: documents a series of word pairs
        :rtype: List[List[Tuple[str, str]]] 
    """
    
    if not token_filter: #if no filter is specified
        def token_filter(x): return x #just return the entire document
    
    documents_word_pairs = []
    for document in documents:
        document = token_filter(document) #get only the words of interest
        #TODO: would it be more efficient to do all documents at once?
        documents_word_pairs.append(list(combinations(document, 2)))
    
    return documents_word_pairs