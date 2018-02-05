'''
    Created on Dec 13, 2017

    :author: Andrew Cattle <acattle@cse.ust.hk>

    This module implements the humour features described in Yang et al. (2015).

        Yang, D., Lavie, A., Dyer, C., & Hovy, E. H. (2015). Humor Recognition
        and Humor Anchor Extraction. In EMNLP (pp. 2367-2376).
'''
from __future__ import print_function, division
import re
from nltk import pos_tag_sents
from nltk.corpus import wordnet as wn
import numpy as np
from math import log
from itertools import combinations, product
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from humour_features.utils.common_features import get_alliteration_and_rhyme_features,\
    get_interword_score_features
import logging

def _convert_pos_to_wordnet(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return ''

def train_yang_et_al_2015_pipeline(X, y, w2v_model, wilson_lexicon_loc, k=5, **rf_kwargs):
    """
        Train a  humour classifier using Yang et al. (2015)'s Word2Vec+HCF
        feature set
        
        :param X: Training documents. Each document should be a sequence of tokens.
        :type X: Iterable[Iterable[str]]
        :param y: training labels
        :type y: Iterable[int]
        :param w2v_model: the Word2Vec model
        :type w2v_model: GensimVectorModel
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
    yang_pipeline = Pipeline([("extract_features", YangHumourFeatureExtractor(w2v_model,wilson_lexicon_loc,k)),
                              ("random_forest_classifier", RandomForestClassifier(**rf_kwargs))
                              ])
    yang_pipeline.fit(X,y)
    
    return yang_pipeline

def run_yang_et_al_2015_baseline(train, test, w2v_model, wilson_lexicon_loc):
    """
        Convenience method for running Yang et al. (2015) humour classification
        experiment on a specified dataset. This is equivalent to running
        Yang et al. (2015)'s Word2Vec+HCF
        
        :param train: A tuple containing training documents and labels. Each document should be a sequence of tokens.
        :type train: Tuple[Iterable[Iterable[str]], Iterable[int]]
        :param test: A tuple containing training documents and labels. Each document should be a sequence of tokens.
        :type test: Tuple[Iterable[Iterable[str]], Iterable[int]]
        :param w2v_model: the Word2Vec model
        :type w2v_model: GensimVectorModel
        :param wilson_lexicon_loc: location of Wilson et al. (2005) lexicon file
        :type wilson_lexicon_loc: str  
    """
    
    train_X, train_y = train
    yang_pipeline = train_yang_et_al_2015_pipeline(train_X, train_y, w2v_model, wilson_lexicon_loc, k=5) #Yang et al. (2015) use a K of 5
    
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

class YangHumourFeatureExtractor(TransformerMixin):
    """
        A class for implementing the features used in Yang et al. (2015) as a
        scikit-learn transformer, suitable for use in scikit-learn pipelines.
    """
    
    def __init__(self, w2v_model, wilson_lexicon_loc, k=5, verbose=False):
        """
            Configure Yang et al. (2015) feature extraction options including
            Word2Vec model and Wilson et al. (2005) subjectivity lexicon
            locations.
            
                Theresa Wilson, Janyce Wiebe and Paul Hoffmann (2005).
                Recognizing Contextual Polarity in Phrase-Level Sentiment
                Analysis. Proceedings of HLT/EMNLP 2005, Vancouver, Canada.
            
            :param w2v_model: the Word2Vec model
            :type w2v_model: GensimVectorModel
            :param wilson_lexicon_loc: location of Wilson et al. (2005) lexicon file
            :type wilson_lexicon_loc: str
            :param k: Specifies the number of KNN labels to extract
            :type k: int
            :param verbose: whether we should use verbose mode or not
            :type verbose: bool
        """
        self.w2v_model = w2v_model
        self.wilson_lexicon_loc = wilson_lexicon_loc
        self.wilson_lexicon=None
        self.k=k
        self.knn_model = None
        self.train_y = None #will be used for KNN features
        self.knn_vectorizer = None #will be used to transform documents into input for self.knn_model
        
        self.logger = logging.getLogger(__name__)
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
        scorer = self.w2v_model.get_similarity
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
        #TODO: currently I am assuming that each document is 1 sentence. If this isn't True it might affect pos_tag performance
        _cache = {}# caching results  during batch processing resulted in a 33% speedup on Pun of the Day dataset
        
        #nltk's pos_tag and pos_tag_sents both load a POS tagger from pickle.
        #Therefore, POS tagging all documents at once reduces disk i/o; faster
        pos_tagged_documents = pos_tag_sents(documents)
        
        feature_vectors = []
        for pos_tagged_document in pos_tagged_documents:
            word_senses = []
            sense_product = 1 #the running product of the number of word senses
            for token, pos in pos_tagged_document:
                wn_pos = _convert_pos_to_wordnet(pos) # returns anrsv or empty string
                senses = wn.synsets(token, wn_pos)  #using empty string as pos makes this function return an empty set
                
                if len(senses) > 0: #this avoids log(0) later
                    sense_product = sense_product * len(senses)
                    word_senses.append(senses)
             
            sense_combination = log(sense_product)
             
            sense_farmost = 1 #the least similar pair of senses
            sense_closest = 0 #the most similar pair of senses
            #Yang et al. (2015) defines "sense farmost" as "the largest path similarity" but that doesn't make sense.
            #They must have mixed up the definitions for farmost and closest
             
            #TODO: It's ambiguous in Yang et al. (2015) as to if we should include intra-word sense combinations
            #Or, conversely, if we should only examine intra-word sense combinations
            #we currently do neither, only looking at inter-word similarities
            for word1_senses, word2_senses in combinations(word_senses, 2): #for each word pair
                for sense_pair in product(word1_senses, word2_senses): #for each synset pair
                    _cache_key = tuple(sorted(sense_pair)) #sort to force consistent ordering. tuple to make it hashable
                    #path_similarity is symmetric so reordering the synsets is OK
                    
                    path_sim=None
                    if _cache_key in _cache:
                        path_sim = _cache[_cache_key]
                    else:
                        path_sim = wn.path_similarity(*sense_pair)
                        _cache[_cache_key] = path_sim
                     
                    if path_sim != None: #if  we have a path similarity
                        #TODO: is this what Yang did?
                        sense_farmost = min(sense_farmost, path_sim)
                        sense_closest = max(sense_closest, path_sim)
            feature_vectors.append((sense_combination, sense_farmost, sense_closest))
#             processed+=1
#             if processed%1000==0:
#                 print("{}/{}".format(processed, total))
        
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
        return get_alliteration_and_rhyme_features(documents)
        
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
            sum_vector = np.zeros(self.w2v_model.get_dimensions()) #initialize sum to be a vector of 0s the same dimensions as the model
            for word in document:
                sum_vector = sum_vector + self.w2v_model.get_vector(word)
            
            averaged_w2vs.append(sum_vector / len(document))
        
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
        self.logger.debug("Starting interpersonal features")
        features.append(self.get_interpersonal_features(X))
        self.logger.debug("Starting phonetic features")
        features.append(self.get_phonetic_features(X))
        self.logger.debug("Starting KNN features")
        features.append(self.get_knn_features(X))
         
        #extract Word2Vec features
        self.logger.debug("Starting average w2v")
        features.append(self.get_average_w2v(X))
        self.logger.debug("Starting incongruity features")
        features.append(self.get_incongruity_features(X))
        
        self.logger.debug("All features extracted")
        return np.hstack(features)

class YangHumourAnchorExtractor:
    """
        A class for implementing Yang et al. (2015) as a scikit-learn
        transformer, suitable for use in scikit-learn pipelines.
    """
    
    def __init__(self,parser,humour_scorer, t=3, s=2, pos_class=1):
        """
            Specify parser and humour_scorer for use in Humour Anchor
            Extraction.
            
            :param parser: Parser function. Must take  a list of strings and return an nltk.Tree using Penn Treebank tags
            :type parser: Callable[[Iterable[str]], nltk.Tree]
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
        
        self.parse = parser
        self.humour_scorer = humour_scorer
        self.t = t
        self.s = s
        self.pos_class = pos_class
    
    def get_anchor_candidates(self, document):
        """
            Generate all candidate humour anchors according to method defined
            in Yang et al. (2015)
            
            :param document: document to be processed as a sequence of tokens
            :type document: List[str]
            
            :return: list of candidate humour anchors
            :rtype: List[List[str]]
        """
        
        parse = self.parse(document)

        #add all NP, VP, ADJP, or AVBP that are "minimal parse subtrees". I think this means doesn't contain and phrases, just terminals
        candidates = set()
        for subtree in parse.subtrees(lambda t: t.height() == 3):
            #height==3 because according to nltk.Tree docs: "height of a tree containing only leaves is 2"
            #we want one layer about that.
            if subtree.label() in ["NP", "VP", "ADJP", "ADVP"]:
                
                filtered_words = tuple([word.lower() for word, pos in subtree.pos() if pos not in ["DT", "PRP", ",", ":", "."]])
                if filtered_words:
                    candidates.add(filtered_words) #add ordered tuple of leaves
                #TODO: do we really want to ignore determiners?
                #TODO: filter out determiners and other POSes?
            
        #add remaining nouns and verbs
        for word, pos in parse.pos():
            if pos.startswith("NN") or pos.startswith("VB"):
                if ((not pos.startswith("PRP")) and #ignore personal pronouns #TODO: Is this needed? I originally confused PRP for NNP
                        all(word not in candidate for candidate in candidates)): #if the word is not part of any existing candidate
                    candidates.add((re.sub("\W", "", word.lower()),))
                    #TODO: checking if we're looking at the same subtree?
        
        return candidates
    
    def find_humour_anchors(self, document):
        candidates = self.get_anchor_candidates(document)
        
        #TODO: better, more transparent implementation
        from nltk.tokenize import word_tokenize
        document = word_tokenize(document)
        
        documents_minus_subsets = [document] #we want to make sure we get a baseline humour score
        anchors = []
        for i in range(self.s, self.t+1):
            for anchor_comb in combinations(candidates, i):
                #for all combinations of anchors <= t (including the empty set)
                anchors.append(anchor_comb)
                words_to_remove = set(word for anchor in anchor_comb for word in anchor) #flatten candidates
                documents_minus_subsets.append([word for word in document if word not in words_to_remove]) #remove words from document
                
#         import sys;sys.path.append(r'/mnt/c/Users/Andrew/.p2/pool/plugins/org.python.pydev_6.2.0.201711281614/pysrc')
#         import pydevd;pydevd.settrace(stdoutToServer=True, stderrToServer=True)
        humour_probs = self.humour_scorer.predict_log_proba(documents_minus_subsets)
          
        #get only the positive label column
        pos_index = np.where(self.humour_scorer.classes_ == self.pos_class)[0][0] #get the positive column index
        humour_probs = humour_probs[:,pos_index] #all rows, only the positive column
        
        baseline_score = humour_probs[0] #get score for document with no anchors removed
        decrements =  baseline_score - humour_probs[1:]
        max_decrement_arg = np.argmax(decrements)#find the index with the highest value
        
        return anchors[max_decrement_arg]

    #TODO: What do they mean by "closest in meaning" for KNN?
    
if __name__ == "__main__":
    potd_loc = "D:/datasets/pun of the day/puns_pos_neg_data.csv"
    oneliners_loc = "D:/datasets/16000 oneliners/Jokes16000.txt"
#     w2v_loc = "C:/vectors/GoogleNews-vectors-negative300.bin"
    
#     potd_loc = "/mnt/d/datasets/pun of the day/puns_pos_neg_data.csv"
#     oneliners_loc = "/mnt/d/datasets/16000 oneliners/Jokes16000.txt"
#     w2v_loc = "/mnt/c/vectors/GoogleNews-vectors-negative300.bin"
    
    wilson_lexicon_loc = "subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff"
    
    docs_and_labels=[]
    with open(potd_loc, "r") as potd_f:
        potd_f.readline() #pop the header
              
        for line in potd_f:
            label, doc = line.split(",")
            docs_and_labels.append((doc.split(), int(label)))
            #potd is pre-tokenized
          
    import random
    random.seed(10)
    random.shuffle(docs_and_labels)
          
#     test_size = round(len(docs_and_labels)*0.1) #hold out 10% as test
#     
#     test_X, test_y = zip(*docs_and_labels[:test_size]) #unzip the documents and labels
#     train_X, train_y = zip(*docs_and_labels[test_size:])
          
          
#     yang = YangHumourFeatureExtractor(w2v_loc,wilson_lexicon_loc)
#     import timeit
#     t=timeit.timeit("YangHumourFeatureExtractor(None,None,n_jobs=3,manager=manager).get_ambiguity_features_pool(train_X)", "from __main__ import YangHumourFeatureExtractor, train_X, manager",number=1)
#     print("main + 3 threads + shared cache: {} seconds".format(t))
#     t=timeit.timeit("YangHumourFeatureExtractor(None,None).get_ambiguity_features_single(train_X)", "from __main__ import YangHumourFeatureExtractor, train_X",number=1)
#     print("main thread only + caching: {} seconds".format(t))
#     print(timeit.timeit("YangHumourFeatureExtractor(None,None,n_jobs=1).get_ambiguity_features_pool(train_X)", "from __main__ import YangHumourFeatureExtractor, train_X",number=1))
#     print(timeit.timeit("YangHumourFeatureExtractor(None,None).get_ambiguity_features_filtered(train_X,False)", "from __main__ import YangHumourFeatureExtractor, train_X",number=1))
          
#     print(timeit.timeit("run_yang_et_al_2015_baseline((train_X, train_y), (test_X, test_y), w2v_loc, wilson_lexicon_loc,n_jobs=4)", "from __main__ import run_yang_et_al_2015_baseline,train_X,train_y,test_X,test_y,w2v_loc,wilson_lexicon_loc",number=1))
      
    X,y = zip(*docs_and_labels)
    print("starting training")
    from util.common_models import get_google_word2vec
    yang = train_yang_et_al_2015_pipeline(X, y, get_google_word2vec(), wilson_lexicon_loc, n_estimators=100, min_samples_leaf=100, n_jobs=-1)
    print("training complete\n\n")
     
# #     #save the model
# #     yang.named_steps["extract_features"]._purge_w2v_model() #smaller pkl
# #     from sklearn.externals import joblib
# #     joblib.dump(yang, "yang_pipeline_ubuntu.pkl")
#     import dill
#     with open("yang_pipeline_ubuntu_100minsamples_class.dill", "wb") as yang_f:
#         dill.dump(yang, yang_f)
# #     with open("yang_pipeline_ubuntu_100minsamples.dill", "rb") as yang_f:
# #         yang=dill.load(yang_f)
#  
# # #     from sklearn.preprocessing.data import StandardScaler
# # # #     from sklearn.svm.classes import LinearSVC
# # #     from sklearn.linear_model import LogisticRegression
# # #     from sklearn.pipeline import Pipeline
# # #     dumb_classifier = Pipeline([("count vector", CountVectorizer()),
# # # #                                 ("scale", StandardScaler()),
# # #                                 ("logistic regression", LogisticRegression())
# # #                                 ])
# # #     dumb_classifier.fit(X,y)
# # #     print('fitted')
# # #     with open("bow_lr.dill", "wb") as bow_f:
# # #         dill.dump(dumb_classifier, bow_f)
# #     with open("bow_lr.dill", "rb") as bow_f:
# #         dumb_classifier = dill.load(bow_f)
        
        
    oneliners = []
    with open(oneliners_loc, "r", encoding="latin-1") as oneliners_f:
        for line in oneliners_f:
#             words = nltk.word_tokenize(line.lower())
#             words = [w for w in words if w.isalpha()]
              
            oneliners.append(line.strip())
            if len(oneliners) >= 10:
                break
    
#     class StatParserWrapper():
#         def __init__(self,parser):
#             self.parser=parser
#         
#         def parse(self, document):
#             return self.parser.parse(" ".join(document))

#     from stat_parser import Parser
    from parser.bllip_wrapper import BllipParser
    bllip = BllipParser()

    
    anchor_extractor = YangHumourAnchorExtractor(bllip.parse, yang, 3)
#     count = 0
    for oneliner in oneliners:
#         if label==1:
        anchors= anchor_extractor.find_humour_anchors(oneliner)
        print(oneliner)
        print(anchors)
        print("\n")
#         count+=1
#         if count >9:
#             break
    
    
    #TODO: classifier vs regressor for anchor extractor
    