'''
Created on Feb 6, 2018

@author: Andrew Cattle <acattle@cse.ust.hk>
'''
from nltk import word_tokenize, sent_tokenize
from string import punctuation
from itertools import combinations
from nltk.corpus import stopwords
import re

ENGLISH_STOPWORDS = set(stopwords.words('english'))
ENGLISH_STOPWORDS.add("n't") #added for compatibility with word_tokenize
#TODO: this set of stopwords will remove negations like "no", "not", etc. Is this what we want to do?

def default_preprocessing_and_tokenization(documents, stopwords=ENGLISH_STOPWORDS, flatten_sents=True):
    """
        A method for preprocessing and tokening documents that should be good
        enough for the majority of contexts.
        
        Each document is lowercased, tokenized using nltk.word_tokenize, and
        has its punctuation removed.
        
        :param documents: the documents to be processed
        :type documents: Iterable[str]
        :param stopwords: stopwords to be removed
        :type stopwords: Iterable[str]
        :param flatten_sents: specifies whether sentences should be flattened into a single list of tokens
        :type flatten_sents: bool
        
        :returns: preprocessed and tokenized document
        :rtype: List[List[str]] or List[List[List[str]]]
    """
    
    if not stopwords:
        stopwords=set()
    punc_re = re.compile(f'[{re.escape(punctuation)}]')
    
    preprocessed_documents = []
    
    for document in documents:
        processed_doc = []
        for sent in sent_tokenize(document.lower()):
            tokens = word_tokenize(sent)
            
            processed_tokens = []
            for token in tokens:
                if token not in stopwords:
                    token = punc_re.sub("", token) #remove punctuation
                    #https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
                    
                    if punc_re.sub("", token): #if we have at least one non-punctuation character
                        #This will remove punctuation-only tokens while preserving contractions and posessives which are important for parsing.
                        processed_tokens.append(token)
                        
            if processed_tokens: #omit empty sentences. They screw up parsing.
                if flatten_sents:
                    #document should be a single list
                    processed_doc.extend(processed_tokens)
                else:
                    #document should be a list of sentences
                    processed_doc.append(processed_tokens)
            
        preprocessed_documents.append(processed_doc)
    
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