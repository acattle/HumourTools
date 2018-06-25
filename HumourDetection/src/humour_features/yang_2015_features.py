'''
    Created on Dec 13, 2017

    :author: Andrew Cattle <acattle@cse.ust.hk>

    This module implements the humour features described in Yang et al. (2015).

        Yang, D., Lavie, A., Dyer, C., & Hovy, E. H. (2015). Humor Recognition
        and Humor Anchor Extraction. In EMNLP (pp. 2367-2376).
'''
from __future__ import print_function, division
from nltk import pos_tag_sents, word_tokenize, sent_tokenize
from nltk.corpus import cmudict, wordnet as wn
from nltk.tree import Tree
from math import log
from itertools import combinations, product
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from humour_features.utils.common_features import get_alliteration_and_rhyme_features,\
    get_interword_score_features
from util.loggers import LoggerMixin
from util.wordnet.treebank_to_wordnet import get_wordnet_pos
from util.text_processing import default_filter_tokens, POS_TO_IGNORE, strip_pos,\
    strip_pos_sent
import re
import numpy as np
import logging


from multiprocessing import Pool
import time
from _collections import defaultdict
from functools import lru_cache
from util.wordnet.wordnet_utils import get_synsets, path_similarity

#Size of LRU caches. None means no limit
_cache_size=None

def train_yang_et_al_2015_pipeline(X, y, w2v_model_getter, wilson_lexicon_loc, k=5, pretagged=False, **rf_kwargs):
    """
        Train a  humour classifier using Yang et al. (2015)'s Word2Vec+HCF
        feature set
        
        :param X: Training documents. Each document should be a sequence of tokens.
        :type X: Iterable[Iterable[str]]
        :param y: training labels
        :type y: Iterable[int]
        :param w2v_model_getter: function for retrieving the Word2Vec model
        :type w2v_model_getter: Callable[[], GensimVectorModel]
        :param wilson_lexicon_loc: location of Wilson et al. (2005) lexicon file
        :type wilson_lexicon_loc: str
        :param k: the number of neighbors to use for KNN features
        :type k: int
        :param rf_kwargs: keyword arguments to be passed to the RandomForestClassifier
        :type rf_kwargs: Dict[str, Any]
        
        :return: A trained pipeline that takes in tokenized documents and outputs predictions
        :rtype: sklearn.pipeline.Pipeline
    """
    
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble.forest import RandomForestClassifier
    yang_pipeline = Pipeline([("extract_features", YangHumourFeatureExtractor(w2v_model_getter,wilson_lexicon_loc,k,pretagged=pretagged)),
                              ("random_forest_classifier", RandomForestClassifier(**rf_kwargs))
                              ])
    yang_pipeline.fit(X,y)
    
    return yang_pipeline

def run_yang_et_al_2015_baseline(train, test, w2v_model_getter, wilson_lexicon_loc, **rf_kwargs):
    """
        Convenience method for running Yang et al. (2015) humour classification
        experiment on a specified dataset. This is equivalent to running
        Yang et al. (2015)'s Word2Vec+HCF
        
        :param train: A tuple containing training documents and labels. Each document should be a sequence of tokens.
        :type train: Tuple[Iterable[Iterable[str]], Iterable[int]]
        :param test: A tuple containing training documents and labels. Each document should be a sequence of tokens.
        :type test: Tuple[Iterable[Iterable[str]], Iterable[int]]
        :param w2v_model_getter: function for retrieving the Word2Vec model
        :type w2v_model_getter: Callable[[], GensimVectorModel]
        :param wilson_lexicon_loc: location of Wilson et al. (2005) lexicon file
        :type wilson_lexicon_loc: str
        :param rf_kwargs: keyword arguments to be passed to the RandomForestClassifier
        :type rf_kwargs: Dict[str, Any]
    """
    
    train_X, train_y = train
    yang_pipeline = train_yang_et_al_2015_pipeline(train_X, train_y, w2v_model_getter, wilson_lexicon_loc, k=5, **rf_kwargs) #Yang et al. (2015) use a K of 5
    
    test_X, test_y = test
    pred_y = yang_pipeline.predict(test_X)
    
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    p,r,f,_ = precision_recall_fscore_support(test_y, pred_y, average="binary")
    a = accuracy_score(test_y, pred_y)
    print("Yang et al. (2015) Humour Classification")
    print("Accuracy: {}".format(a))
    print("Precision: {}".format(p))
    print("Recall: {}".format(r))
    print("F-Score: {}".format(f))

class YangHumourFeatureExtractor(TransformerMixin, LoggerMixin):
    """
        A class for implementing the features used in Yang et al. (2015) as a
        scikit-learn transformer, suitable for use in scikit-learn pipelines.
    """
    #TODO: What do they mean by "closest in meaning" for KNN?
    
    def __init__(self, w2v_model_getter, wilson_lexicon_loc, k=5, pretagged=False, verbose=True, cache=True):
        """
            Configure Yang et al. (2015) feature extraction options including
            Word2Vec model and Wilson et al. (2005) subjectivity lexicon
            locations.
            
                Theresa Wilson, Janyce Wiebe and Paul Hoffmann (2005).
                Recognizing Contextual Polarity in Phrase-Level Sentiment
                Analysis. Proceedings of HLT/EMNLP 2005, Vancouver, Canada.
            
            :param w2v_model_getter: function for retrieving the Word2Vec model
            :type w2v_model_getter: Callable[[], GensimVectorModel]
            :param wilson_lexicon_loc: location of Wilson et al. (2005) lexicon file
            :type wilson_lexicon_loc: str
            :param k: Specifies the number of KNN labels to extract
            :type k: int
            :param pretagged: specifies whether documents are pre-POS tagged or not
            :type pretagged: bool
            :param verbose: whether we should use verbose mode or not
            :type verbose: bool
            :param cache: whether ambiguity results should be cached
            :type cache: bool
        """
        self.get_w2v_model = w2v_model_getter
        self.wilson_lexicon_loc = wilson_lexicon_loc
        self.wilson_lexicon=None
        self.k=k
        self.knn_model = None
        self.train_y = None #will be used for KNN features
        self.knn_vectorizer = None #will be used to transform documents into input for self.knn_model
        self.cmu_dict = cmudict.dict() #if we load this at init, it will save having to load it for each transform call
        self.pretagged = pretagged
        self.verbose_interval = 1000
        if verbose:
            self.logger.setLevel(logging.DEBUG)
    
    
    def _get_wilson_lexicon(self):
        """
            Method for lazy loading Wilson et al. (2005) lexicon.
            
            :return: a tuple in the form of (strongsubj, weaksubj, positive, negative) containing the words that belong to each tag
            :rtype: (set, set, set, set)
        """
        
        if self.wilson_lexicon == None:
            #these will hold the relevant word lists
            strongsubj=[]
            weaksubj=[]
            negative=[]
            positive=[]
            with open(self.wilson_lexicon_loc, "r") as wilson_f:
                for line in wilson_f:
                    #each line is of the form "type=strongsubj len=1 word1=abuse pos1=verb stemmed1=y priorpolarity=negative"
                    line = line.split()
                    subjectivity = line[0].split("=")[1]
                    word = line[2].split("=")[1]
                    polarity = line[5].split("=")[1]
                    #exactly 2 lines, 5549 and 5550, had "stemmed1=n m". Had to manually edit file to get it to work
                    
                    if subjectivity == "strongsubj":
                        strongsubj.append(word)
                    elif subjectivity == "weaksubj":
                        weaksubj.append(word)
                    
                    if polarity == "negative":
                        negative.append(word)
                    elif polarity == "positive":
                        positive.append(word)
            
            #convert to sets for fast lookup later
            self.wilson_lexicon = (set(strongsubj), set(weaksubj), set(positive), set(negative))
        
        return self.wilson_lexicon

    def get_incongruity_features(self, documents):
        """
            Calculates the incongruity features described in Section 4.1 of
            Yang et al. (2015)
            
            :param documents: documents to be processed. Each document should be a sequence of tokens
            :type documents: Iterable[Iterable[str]]
            
            :return: A matrix representing the extracted incongruity features in the form (disconnection, repetition) x # of documents
            :rtype: numpy.array
        """
        
        scorer=self.get_w2v_model().get_similarity
        incong_features = get_interword_score_features(documents, scorer)
        #Yang et al. (2015) only care about max and min Word2Vec similarities
        #Therefore we should delete column 1 (first 1 is the column index. second 1 is axis)
        incong_features = np.delete(incong_features, 1, 1)
        
        return incong_features
    
    def get_ambiguity_features(self, documents):
        """
            Calculates the ambiguity features described in Section 4.2 of
            Yang et al. (2015)
            
            :param documents: documents to be processed. Each document should be a sequence of tokens
            :type documents: Iterable[Iterable[str]]
            
            :return: The extracted ambiguity features in the form ndarray(sense_combination, sense_farmost, sense_closest)
            :rtype: numpy.array
        """
        
        pos_tagged_documents = documents #assume documents are POS tagged
        if not self.pretagged: #if documents haven't be POS tagged yat
            #TODO: currently I am assuming that each document is 1 sentence. If this isn't True it might affect pos_tag performance
            #nltk's pos_tag and pos_tag_sents both load a POS tagger from pickle.
            #Therefore, POS tagging all documents at once reduces disk i/o; faster
            pos_tagged_documents = pos_tag_sents(documents)
        
        feature_vectors = []
        processed=0
        total = len(pos_tagged_documents)
        for pos_tagged_document in pos_tagged_documents:
            word_senses = set()
            sense_product = 1 #the running product of the number of word senses
            sense_farmost = 1 #the least similar pair of senses
            sense_closest = 0 #the most similar pair of senses
            for token, pos in pos_tagged_document:
                wn_pos = get_wordnet_pos(pos) # returns anrsv or empty string
                
                if wn_pos: #no use doing any of this unless we have a valid POS
                    senses=get_synsets(token,wn_pos) #uses cached function
                    
                    if senses: #this avoids log(0) later
                        sense_product *= len(senses)
                        
                        for s1 in senses:
                            if s1 not in word_senses: #avoid duplicating work
                                #TODO: does this help?
                                for s2 in word_senses:                                    
                                    #path_similarity is symmetric so reordering the synsets is OK and results in smaller cache
                                    path_sim=path_similarity(sorted((s1,s2))) #uses a chached function
                                     
                                    if path_sim != None: #if  we have a path similarity
                                        #TODO: is this what Yang did?
                                        #this is quicker than max() and min(), respectively
                                        sense_farmost = sense_farmost if sense_farmost > path_sim else path_sim
                                        sense_closest = sense_closest if sense_closest < path_sim else path_sim
                        
                        word_senses.update(senses)
             
            sense_combination = log(sense_product)
            feature_vectors.append((sense_combination, sense_farmost, sense_closest))
            processed+=1
            if processed%self.verbose_interval==0:
                self.logger.debug(f"{processed}/{total}")
        
        return np.vstack(feature_vectors)
        
    def get_interpersonal_features(self, documents):
        """
            Calculates the interpersonal effect features described in Section
            4.3 of Yang et al. (2015)
            
            :param documents: documents to be processed. Each document shoudl be a sequence of tokens
            :type documents: Iterable[Iterable[str]]
            
            :return: A matrix of the extracted interpersonal features where eat row is the form (neg_polarity, pos_polarity, weak_subjectivity, strong_subjectivity)
            :rtype: numpy.array
        """
        
        strongsubj, weaksubj, positive, negative = self._get_wilson_lexicon()
        
        feature_vectors = []
        for document in documents:
            strongsubj_count=0
            weaksubj_count=0
            positive_count=0
            negative_count=0
            for word in document:
                if word in strongsubj:
                    strongsubj_count = strongsubj_count + 1
                if word in weaksubj:
                    weaksubj_count = weaksubj_count + 1
                if word in positive:
                    positive_count = positive_count + 1
                if word in negative:
                    negative_count = negative_count + 1
            
            feature_vectors.append((negative_count, positive_count, weaksubj_count, strongsubj_count))
        
        return np.vstack(feature_vectors)
    
    def get_phonetic_features(self, documents):
        """
            Calculates the phonetic style features described in Section 4.4 of
            Yang et al. (2015)
            
            :param documents: documents to be processed. Each document is a sequence of tokens
            :type documents: Iterable[Iterable[str]]
            
            :return: a matrix where columns represent extracted phonetic style features in the form (alliteration_num, alliteration_len, rhyme_num, rhyme_len) and rows are documents
            :rtype: numpy.array
        """
        
        #add all 4 features from get_alliteration_and_rhyme_features()
        return get_alliteration_and_rhyme_features(documents, self.cmu_dict)
        
    def get_average_w2v(self,documents):
        """
            Calculates the average Word2Vec vector for all words in the
            document. This is equivalent to the Word2Vec baseline described in
            Section 6.1 of Yang et al. (2015).
            
            Although Yang et al. (2015) do specify how they construct their
            sentence-level embeddings, during their oral presentation the
            authors clarify that they simply average vectors for each word
            in the document.
            
            See https://vimeo.com/173994665 for details
            
            :param documents: documents to be processed. Each document should be as a sequence of tokens
            :type documents: Iterable[Iterable[str]]
            
            :return: a matrix of the averaged vectors of all words in each document
            :rtype: numpy.array
        """ 
        
        averaged_w2vs=[]
        for document in documents:
            #TODO: Should we omit OOV words?
            sum_vector = np.zeros(self.get_w2v_model().get_dimensions()) #initialize sum to be a vector of 0s the same dimensions as the model
            for word in document:
                sum_vector = sum_vector + self.get_w2v_model().get_vector(word)
            
            avg_vector = sum_vector / len(document) if document else np.zeros(self.get_w2v_model().get_dimensions())
            averaged_w2vs.append(avg_vector)
        
        return np.vstack(averaged_w2vs)
    
    def get_knn_features(self, documents):
        """
            Returns the labels of the K nearest neighbours in ascending order
            of distance. This corresponds to the KNN feature described in
            Section 6.1 of Yang et al. (2015).
            
            The K is specified during class initialization.
            
            :param documents: documents to be processed. Each document is a sequence of tokens
            :type documents: Iterable[Iterable[str]]
            
            :returns: a matrix where columns represent labels of the K nearest training examples and rows are documents
            :rtype: numpy.array
            
            :raises NotFittedError: If KNN model hasn't been initialized
        """
        
        #TODO: should I include an extra, majority label feature?
        if self.knn_model == None:
            raise NotFittedError("Must fit YangHumourFeatureExtractor before KNN features can be extracted.")
        
        #get indexes
        knn_indexes = self.knn_model.kneighbors(self.knn_vectorizer.transform(documents), return_distance=False)
        
        #convert indexes to labels
        def _replace_index_with_label(i):
            return self.train_y[i]
        return np.vectorize(_replace_index_with_label)(knn_indexes) #apply _replace_index_with_label to each item in knn_indexes matrix
    
    def fit(self,X,y):
        """
            Fits KNN model so that KNN features can be extracted
            
            :returns: self (for compatibility with sklearn pipelines)
            :rtype: TransformerMixin
        """
        
        self.train_y = y #save this to get labels later
        
        self.logger.debug("KNN fitting started")
        #TODO: is this the appropriate Vecotrizer?
        self.knn_vectorizer = CountVectorizer(tokenizer=lambda x:  x, preprocessor=lambda x: x) #skip tokenization and preporcessing
        X = self.knn_vectorizer.fit_transform(X)
        self.knn_model = NearestNeighbors(n_neighbors=self.k).fit(X)
        
        self.logger.debug("KNN fitting complete")
        
        return self
    
    def transform(self, X):
        """
            Takes in a  series of tokenized documents and extracts a set of
            features equivalent to the highest performing model in Yang et al.
            (2015)
            
            :param X: pre-tokenized documents
            :type X: Iterable[Iterable[str]]
            
            :return: highest performing Yang et al. (2015) features as a numpy array
            :rtype: numpy.array
        """
        
        features = []
        
        self.logger.debug("Starting ambiguity features")
        features.append(self.get_ambiguity_features(X))
        
        X_tokens = X #assume X is just tokens
        if self.pretagged: #if X is tokens with POS labels
            X_tokens = [[token for token, pos in document] for document in X] #get just the tokens        
        
        self.logger.debug("Starting interpersonal features")
        features.append(self.get_interpersonal_features(X_tokens))
        self.logger.debug("Starting phonetic features")
        features.append(self.get_phonetic_features(X_tokens))
        self.logger.debug("Starting KNN features")
        features.append(self.get_knn_features(X_tokens))
          
        #extract Word2Vec features
        self.logger.debug("Starting average w2v")
        features.append(self.get_average_w2v(X_tokens))
        self.logger.debug("Starting incongruity features")
        features.append(self.get_incongruity_features(X_tokens))
        
        self.logger.debug("All features extracted")
        return np.hstack(features)

class YangHumourAnchorExtractor(LoggerMixin):
    """
        A class for implementing Yang et al. (2015) as a scikit-learn
        transformer, suitable for use in scikit-learn pipelines.
    """
    
    def __init__(self,parser,humour_scorer, t=3, s=2, pos_class=1, verbose=True):
        """
            Specify parser and humour_scorer for use in Humour Anchor
            Extraction.
            
            :param parser: Parser function. Must take  a list of documents (one string per document) and return a list of nltk.Tree objects using Penn Treebank tags
            :type parser: Callable[[Iterable[str]], Iterable[nltk.Tree]]
            :param humour_scorer: pipeline for scoring humour. Must take bag-of-words as input
            :type humour_scorer: sklearn.pipeline.Pipeline
            :param t: the maximum number of humour anchors
            :type t: int
            :param s: the minimum number of humour anchors
            :type s: int
            :param pos_class: humour_scorer's posivite humour class
            :type pos_class: int
        """
        #TODO: instead of List[str] should we use just str?
        
        self.raw_parse_sents = parser
        self.humour_scorer = humour_scorer
        self.humour_scorer.named_steps["extract_features"].pretagged=True #TODO: more transparent
        self.t = t
        self.s = s
        self.pos_class = pos_class
        
        self.logInterval = 100
        if verbose:
            self.logger.setLevel(logging.DEBUG)
    
    def get_anchor_candidates(self, doc_parses):
        """
            Generate all candidate humour anchors according to method defined
            in Yang et al. (2015)
            
            :param doc_parses: documents to be processed as a sequence nltk.Tree objects
            :type doc_parses: List[List[nltk.Tree]]
            
            :return: list of candidate humour anchors
            :rtype: List[List[str]]
        """
        doc_candidates = []
        
        for doc_parse in doc_parses:
            candidates = set()
            for sent_parse in doc_parse:
        
                #add all NP, VP, ADJP, or AVBP that are "minimal parse subtrees". I think this means doesn't contain and phrases, just terminals
                for subtree in sent_parse.subtrees(lambda t: t.height() == 3):
                    #height==3 because according to nltk.Tree docs: "height of a tree containing only leaves is 2"
                    #we want one layer about that.
                    if subtree.label() in ["NP", "VP", "ADJP", "ADVP"]:
                        
                        #filter meaningless words out of humour anchors
                        filtered_words = tuple(strip_pos(default_filter_tokens([[subtree.pos()]], stopwords=[]))[0])
                        #default_filter_Tokens() expects a list of documents where each is a list of sents where each is a list of token/pos pairs
                        #That's why we put the [] around subtree.pos()
                        if filtered_words:
                            candidates.add(filtered_words) #add ordered tuple of leaves
                        #TODO: do we really want to ignore determiners?
                        #TODO: filter out determiners and other POSes?
                
                candidate_words = set([word for candidate in candidates for word in candidate])
                #add remaining nouns and verbs
                for word, pos in sent_parse.pos():
                    word=word.lower()
                    if pos.startswith("NN") or pos.startswith("VB"):
                        if word not in candidate_words: #if the word is not part of any existing candidate
                            candidates.add((word,))
                            #TODO: checking if we're looking at the same subtree?
                
            doc_candidates.append(candidates)
        
        return doc_candidates
    
    def find_humour_anchors(self, documents):
        """
        Find humour anchors for each document
        
        :param documents: documents to process as list sentences of tokens
        :type documents: List[List[List[str]]]
        
        :returns: extracted humour anchors
        :rtype: List[List[str]]
        """
        
        docs_to_parse = []
        doc_indexes = [] #used for mapping each parse to it's original document
        for doc_num, doc in enumerate(documents):
            #parse sentences separately, but remember which doc they belong to
            for sent in sent_tokenize(doc):
                #For Stanford parser, raw_parse_sents is more robust than parse_sents in terms of character escaping
                #but raw_parse_sents takes strings, not lists of tokens so we need to join the tokens
                docs_to_parse.append(sent)
                doc_indexes.append(doc_num)
        
        self.logger.debug("Starting parse")
        parses = self.raw_parse_sents(docs_to_parse)
        self.logger.debug("parsing done")
         
        parsed_docs = [[] for _ in range(len(documents))] #initialize an empty list for each document
        for i, parse in zip(doc_indexes, parses):
            parsed_docs[i].append(next(parse)) #just take the first parse
         
         
        doc_candidates = self.get_anchor_candidates(parsed_docs)
        self.logger.debug("got candidates")
        
        #Get document tokens and pos tags
        documents = [[sent.pos() for sent in doc] for doc in parsed_docs]
        #Filter the tokens and pos tags
        documents = default_filter_tokens(documents, stopwords=[], flatten_sents=True)
        
#         with open("oneliner candidates.pkl", "wb") as f:
#             pickle.dump(doc_candidates, f)
#         with open("oneliner candidates.pkl", "rb") as f:
#             doc_candidates = pickle.load(f)
        
        docs_to_score = []
        doc_baseline_indexes = []
        doc_anchors = []
        doc_humour_anchors = []
        total = len(doc_candidates)
        for doc_num, (candidates, doc) in enumerate(zip(doc_candidates, documents)):
#             print(doc_num)
#             doc = [tok_pos for sent in doc_parse for tok_pos in sent.pos()] #flatten the doc
            
            doc_baseline_indexes.append(len(docs_to_score)) #get the index of this doc's baseline score
            docs_to_score.append(doc) #we want to make sure we get a baseline humour score
            anchors = []
            for i in range(self.s, self.t+1):
                for anchor_comb in combinations(candidates, i):
                    #for all combinations of anchors <= t (including the empty set)
                    anchors.append(anchor_comb)
                    words_to_remove = set(word for anchor in anchor_comb for word in anchor) #flatten candidates
                    docs_to_score.append([(tok, pos) for tok, pos in doc if tok not in words_to_remove]) #remove words from document
            
            doc_anchors.append(anchors)
                    
    #         import sys;sys.path.append(r'/mnt/c/Users/Andrew/.p2/pool/plugins/org.python.pydev_6.2.0.201711281614/pysrc')
    #         import pydevd;pydevd.settrace(stdoutToServer=True, stderrToServer=True)
        
            #TODO: shift left
            humour_scores = self.humour_scorer.predict_log_proba(docs_to_score)
            
            
            #get only the positive label column
            pos_index = np.where(self.humour_scorer.classes_ == self.pos_class)[0][0] #get the positive column index
            humour_scores = humour_scores[:,pos_index] #all rows, only the positive column
            #TODO: end left shift
        
#         for baseline_index, anchors in zip(doc_baseline_indexes, doc_anchors):
            humour_anchors = [tok for tok, pos in docs_to_score[0]]#baseline_index]] #assume all words are humour anchors
            
            num_anchors = len(anchors)
            if num_anchors > 0: #if we have at least 1 valid humour anchor candidate combination
                baseline_score = humour_scores[0] #TODO:baseline_index] #get score for document with no anchors removed
                decrements =  baseline_score - humour_scores[(0+1) : ((0+1) + num_anchors)] #TODO: 0s to baseline_index
                max_decrement_arg = np.argmax(decrements)#find the index with the highest value
                
                humour_anchors =  anchors[max_decrement_arg]
            
            doc_humour_anchors.append(humour_anchors)
            docs_to_score = [] #TODO: remove
#             
            processed = doc_num + 1
            if processed%self.logInterval == 0:
                self.logger.debug(f"{processed}/{total}")
#             print(humour_anchors)
    
        return doc_humour_anchors
    
if __name__ == "__main__":
    from nltk.parse.stanford import StanfordParser
    import pickle
#     parser = StanfordParser()
    from string import punctuation
    potd_pos = "D:/datasets/pun of the day/puns_of_day.csv" #puns_pos_neg_data.csv
    potd_neg = "D:/datasets/pun of the day/new_select.txt"
    proverbs = "D:/datasets/pun of the day/proverbs.txt"
    oneliners_pos = "D:/datasets/16000 oneliners/Jokes16000.txt"
    oneliners_neg = "D:/datasets/16000 oneliners/MIX16000.txt"
#     potd_pos = "/bigstore/acattle/datasets/pun of the day/puns_of_day.csv" #puns_pos_neg_data.csv
#     potd_neg = "/bigstore/acattle/datasets/pun of the day/new_select.txt"
#     proverbs = "/bigstore/acattle/datasets/pun of the day/proverbs.txt"
#     oneliners_pos = "/bigstore/acattle/datasets/16000 oneliners/Jokes16000-utf8.txt"
#     oneliners_neg = "/bigstore/acattle/datasets/16000 oneliners/MIX16000-utf8.txt"
    
    docs_and_labels=[]
    with open(potd_pos, "r") as pos_f:
        pos_f.readline() #pop the header
                 
        for line in pos_f:
            label, doc = line.split(",", maxsplit=1) #some document conatin commas
            doc=doc.strip()[1:-1] #cut off quotation marks
            docs_and_labels.append((doc, 1)) #the labels in this file are incorrect. All are postive
    with open(potd_neg, "r") as neg_f:
        for line in neg_f:
            docs_and_labels.append((line.strip(), -1))
    with open(proverbs, "r") as neg_f:
        p = re.compile(f'[{re.escape(punctuation)}]')
        for line in neg_f:
            line=line.strip()
            if len(p.sub(" ", line).split()) >5:
                docs_and_labels.append((line, -1))
                #TODO: a couple documents are surrounded by quotes. Selectively remove them?
    
    oneliner_docs_and_labels = []
    with open(oneliners_pos, "r", encoding="ansi") as ol_pos_f:
#     with open(oneliners_pos, "r") as ol_pos_f:
        for line in ol_pos_f:
            oneliner_docs_and_labels.append((line.strip(),1))
    with open(oneliners_neg, "r", encoding="ansi") as ol_neg_f:
#     with open(oneliners_neg, "r") as ol_neg_f:
        for line in ol_neg_f:
            oneliner_docs_and_labels.append((line.strip(),-1))
#             parser.raw_parse_sents([line.strip()])
    
    
    
#     w2v_loc = "C:/vectors/GoogleNews-vectors-negative300.bin"
    
#     potd_loc = "/mnt/d/datasets/pun of the day/puns_pos_neg_data.csv"
#     oneliners_loc = "/mnt/d/datasets/16000 oneliners/Jokes16000.txt"
#     w2v_loc = "/mnt/c/vectors/GoogleNews-vectors-negative300.bin"
    
    wilson_lexicon_loc = "D:/datasets/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff"
#     wilson_lexicon_loc = "/bigstore/acattle/datasets/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff"
    
            
    import random
    random.seed(10)
    random.shuffle(docs_and_labels)
    random.shuffle(oneliner_docs_and_labels)
          
    from util.model_wrappers.common_models import get_google_word2vec
    from util.text_processing import default_preprocessing_and_tokenization
        
    with open("potd_raw_kfold.pkl", "wb") as f:
        pickle.dump(docs_and_labels, f)
    with open("oneliner_raw_kfold.pkl", "wb") as f:
        pickle.dump(oneliner_docs_and_labels, f)

#     X,y = zip(*docs_and_labels)
#     X = default_preprocessing_and_tokenization(X, stopwords=[], flatten_sents=True, leave_pos=True)
#                   
#     yang = train_yang_et_al_2015_pipeline(X, y, get_google_word2vec, wilson_lexicon_loc, pretagged=True, n_estimators=100, min_samples_leaf=100, n_jobs=1)
#     print("training complete\n\n")
#     yang.named_steps["extract_features"]._purge_cache()
#              
# #     from timeit import timeit
# #     timeit("train_yang_et_al_2015_pipeline(X, y, get_google_word2vec, wilson_lexicon_loc, n_estimators=100, min_samples_leaf=100, n_jobs=-1)", "from __main__ import *\nfrom __main__ import _convert_pos_to_wordnet", number=1)
#                
#     #save the model
# #     yang.named_steps["extract_features"]._purge_w2v_model() #smaller pkl
# #     from sklearn.externals import joblib
# #     joblib.dump(yang, "yang_pipeline_min100.pkl")
#     import dill
#     with open("yang_pipeline_potd_better_filter.dill", "wb") as yang_f:
#         dill.dump(yang, yang_f)
#               
#      
#     X,y = zip(*oneliner_docs_and_labels)
#     X = default_preprocessing_and_tokenization(X, stopwords=[], flatten_sents=True, leave_pos=True)
#     print("starting training")
#     yang = train_yang_et_al_2015_pipeline(X, y, get_google_word2vec, wilson_lexicon_loc, pretagged=True, n_estimators=100, min_samples_leaf=100, n_jobs=1)
#     print("training complete\n\n")
#     yang.named_steps["extract_features"]._purge_cache()
#     with open("yang_pipeline_ol_better_filter.dill", "wb") as yang_f:
#         dill.dump(yang, yang_f)
     
    
 
    parser = StanfordParser()
#     jar_loc = "/bigstore/acattle/stanford-parser-full-2017-06-09/stanford-parser.jar"
#     model_loc = "/bigstore/acattle/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar"
#     parser = StanfordParser(model_loc, jar_loc)
           
    
    from util.text_processing import default_preprocessing_and_tokenization
    import dill
# #     with open("oneliners.pkl", "wb") as f:
# #         pickle.dump(oneliner_docs_and_labels, f)
#     with open("oneliners.pkl", "rb") as f:
#         oneliner_docs_and_labels = pickle.load(f)
#     docs, labels = zip(*oneliner_docs_and_labels)
        
    
#     with open("potd_raw.pkl", "rb") as f:
#         docs_and_labels=pickle.load(f)

    #oneliners
#     with open("yang_pipeline_potd_better_filter.dill", "rb") as yang_f:
#         yang=dill.load(yang_f)
        
    docs, labels = zip(*oneliner_docs_and_labels)
#     docs = default_preprocessing_and_tokenization(docs, [], False)
    docs_tok = default_preprocessing_and_tokenization(docs, stopwords=[], flatten_sents=True, leave_pos=True)

    from sklearn.model_selection import StratifiedKFold
    kfold = StratifiedKFold(10)
    splits = list(kfold.split(docs,labels))
    with open("oneliner_splits.pkl", "wb") as f:
        pickle.dump(splits, f)
    
    #null documents for dummy fit. Will be refitted with real data
    #This way we can keep the cache
    null_X = [[("null", "null")]] * 6
    null_y = [0]*6
    yang = train_yang_et_al_2015_pipeline(null_X, null_y, get_google_word2vec, wilson_lexicon_loc, pretagged=True, n_estimators=100, min_samples_leaf=100, n_jobs=1)
    anchor_map = dict()
    for training, test in splits:
        X_train = [docs[i] for i in training]
        X_train_tok = [docs_tok[i] for i in training]
        y_train = [labels[i] for i in training]
        X_test = [docs[i] for i in test]
        
        yang.fit(X_train_tok, y_train)
        
        anchor_extractor = YangHumourAnchorExtractor(parser.raw_parse_sents, yang, 3)
        anchors = anchor_extractor.find_humour_anchors(X_test)
        anchor_map.update({d:a for d,a in zip(X_test, anchors)})
    
    with open("oneliner_kfold_anchor_map.pkl", "wb") as f:
        pickle.dump(anchor_map, f)
        
         
#     with open("ol_anchors_potd_model_better_filter.pkl", "wb") as f:
#         pickle.dump(anchors, f)
    
    
    
    #potd
#     with open("yang_pipeline_ol_better_filter.dill", "rb") as yang_f:
#         yang=dill.load(yang_f)
        
    docs, labels = zip(*docs_and_labels)
    docs_tok = default_preprocessing_and_tokenization(docs, stopwords=[], flatten_sents=True, leave_pos=True)
#     docs = default_preprocessing_and_tokenization(docs, [], False)
    
    kfold = StratifiedKFold(10)
    splits = list(kfold.split(docs,labels))
    with open("potd_splits.pkl", "wb") as f:
        pickle.dump(splits, f)
    
    anchor_map = dict()
    for training, test in splits:
        X_train = [docs[i] for i in training]
        X_train_tok = [docs_tok[i] for i in training]
        y_train = [labels[i] for i in training]
        X_test = [docs[i] for i in test]
        
#         yang = train_yang_et_al_2015_pipeline(X_train_tok, y_train, get_google_word2vec, wilson_lexicon_loc, pretagged=True, n_estimators=100, min_samples_leaf=100, n_jobs=1)
        yang.fit(X_train_tok, y_train)
    
        anchor_extractor = YangHumourAnchorExtractor(parser.raw_parse_sents, yang, 3)
        anchors= anchor_extractor.find_humour_anchors(X_test)
        anchor_map.update({d:a for d,a in zip(X_test, anchors)})
    
    with open("potd_kfold_anchor_map.pkl", "wb") as f:
        pickle.dump(anchor_map, f)
         
#     with open("potd_anchors_ol_model_better_filter.pkl", "wb") as f:
#         pickle.dump(anchors, f)