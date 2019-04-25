'''
Created on Feb 6, 2018

@author: Andrew Cattle <acattle@connect.ust.hk>
'''
from nltk import word_tokenize, sent_tokenize
from string import punctuation
from itertools import combinations
from nltk.corpus import stopwords
import re
from nltk.tag import pos_tag_sents

ENGLISH_STOPWORDS = set(stopwords.words('english'))
ENGLISH_STOPWORDS.add("n't") #added for compatibility with word_tokenize
#TODO: this set of stopwords will remove negations like "no", "not", etc. Is this what we want to do?

POS_TO_IGNORE = set(["DT","POS","PRP","PRP$","TO","UH","MD",".",",",":","(",")","-RRB-","-LRB-","-RSB-","-LSB-","-RCB-","-LCB-","``","''"])

def default_preprocessing_and_tokenization(documents, lowercase=True, stopwords=ENGLISH_STOPWORDS, pos_to_ignore=POS_TO_IGNORE, flatten_sents=True, leave_pos=False):
    """
        A method for preprocessing and tokening documents that should be good
        enough for the majority of contexts.
        
        :param documents: the documents to be processed
        :type documents: Iterable[str]
        :param lowercase: whether documents should be lowercased
        :type lowercase: bool
        :param stopwords: stopwords to be removed
        :type stopwords: Set[str]
        :param pos_to_ignore: pos tags to be ignored
        :type pos_to_ignore: Set[str]
        :param flatten_sents: specifies whether sentences should be flattened into a single list of tokens
        :type flatten_sents: bool
        :param leave_pos: whether we should remove the pos tags from the documents or not
        :type leave_pos: bool
        
        :returns: preprocessed and tokenized document
        :rtype: List[List[str]] or List[List[List[str]]]
    """
    
    doc_indexes = []
    docs_to_pos_tag = []
    for i, document in enumerate(documents):
        for sent in sent_tokenize(document):
            docs_to_pos_tag.append(word_tokenize(sent))
            doc_indexes.append(i)
    
    
    pos_tagged_documents = [[] for i in range(len(documents))]
    for i, pos_tagged_sent in zip(doc_indexes, pos_tag_sents(docs_to_pos_tag)):
        pos_tagged_documents[i].append(pos_tagged_sent)
    
    pos_tagged_documents = default_filter_tokens(pos_tagged_documents, lowercase, stopwords, pos_to_ignore, flatten_sents)
    
    documents = pos_tagged_documents #assume we want pos tags
    if not leave_pos:
        documents = strip_pos(documents, flatten_sents)
    return documents
#             processed_tokens = []
#             for token in tokens:
#                 if token not in stopwords:
# #                     token = punc_re.sub("", token) #remove punctuation
#                     #https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
#                     
#                     if punc_re.sub("", token): #if we have at least one non-punctuation character
#                         #This will remove punctuation-only tokens while preserving contractions and posessives which are important for parsing.
#                         processed_tokens.append(token)
#                         
#             if processed_tokens: #omit empty sentences. They screw up parsing.
#                 if flatten_sents:
#                     #document should be a single list
#                     processed_doc.extend(processed_tokens)
#                 else:
#                     #document should be a list of sentences
#                     processed_doc.append(processed_tokens)
#         
#         if not processed_doc: #if we have an empty doc
#             processed_doc = ["."]
#             if not flatten_sents:
#                 processed_doc = [processed_doc]
#         preprocessed_documents.append(processed_doc)
#     
#     return preprocessed_documents

def default_filter_tokens(documents, lowercase=True, stopwords=ENGLISH_STOPWORDS, pos_to_ignore=POS_TO_IGNORE, flatten_sents=True):
    """
        A method for filtering pos tagged documents that should be good enough
        for most contexts.
        
        :param documents: the documents to be processed
        :type documents: Iterable[str]
        :param lowercase: whether documents should be lowercased
        :type lowercase: bool
        :param stopwords: stopwords to be removed
        :type stopwords: Set[str]
        :param pos_to_ignore: pos tags to be ignored
        :type pos_to_ignore: Set[str]
        :param flatten_sents: specifies whether sentences should be flattened into a single list of tokens
        :type flatten_sents: bool
        
        :returns: document, tokenized and filtered, as a list of token/pos pairs
        :rtype: List[List[Tuple[str, str]]] or List[List[List[Tuple[str,str]]]]
    """
    if not stopwords:
        stopwords=set()
    if not pos_to_ignore:
        pos_to_ignore = set()
    
    punc_re = re.compile(f'[{re.escape(punctuation)}]')
    
    filtered_docs =[]
    for doc in documents:
        filtered_doc = []
        for sent in doc:
            filtered_sent = []
            for token, pos in sent:
                if lowercase:
                    token = token.lower()
                    
                if (token not in stopwords) and \
                    (pos not in pos_to_ignore):
                    token = punc_re.sub("", token)
                    if token: #if we have at least one non-punctuation character
                        filtered_sent.append((token,pos))
            
            if filtered_sent: #ignore empty sentences
                if flatten_sents:
                    #document should be a single list
                    filtered_doc.extend(filtered_sent)
                else:
                    #document should be a list of sentences
                    filtered_doc.append(filtered_sent)
        
        filtered_docs.append(filtered_doc)
    
    return filtered_docs

def strip_pos_sent(sent):
    """
    Strip POS from a single sentence
    """
    return [tok for tok, _ in sent]

def strip_pos(documents, flatten_sents=True):
    stripped_docs = []
    if flatten_sents:
        #documents are a list of list of tokens
        stripped_docs = [strip_pos_sent(doc) for doc in documents]
    else:
        #documents are a list of list of list of tokens
        stripped_docs = [[strip_pos_sent(sent) for sent in doc] for doc in documents]
    
    return stripped_docs

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