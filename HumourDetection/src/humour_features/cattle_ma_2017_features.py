'''
Created on Feb 3, 2018

@author: Andrew Cattle <acattle@connect.ust.hk>

This module contains methods for exacting the features used in Cattle and Ma
(2017)

For more details see:

    Cattle, A., & Ma, X. (2017). SRHR at SemEval-2017 Task 6: Word Associations
    for Humour Recognition. In Proceedings of the 11th International Workshop on
    Semantic Evaluation (SemEval-2017) (pp. 401-406).
'''
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from humour_features.utils.common_features import get_interword_score_features
import numpy as np
from util.loggers import LoggerMixin
import logging
from sklearn.pipeline import Pipeline
from util.text_processing import get_word_pairs
from util.misc import mean
from itertools import product
from math import log


class CattleMaHumourFeatureExtractor(TransformerMixin, LoggerMixin):
    """
        A class for implementing the features used in Cattle and Ma (2017) as a
        scikit-learn transformer, suitable for use in scikit-learn pipelines.
    """
    
    def __init__(self, w2v_model_getter, eat_scorer, usf_scorer, perplexity_scorer, humour_anchor_identifier=None, verbose=False,swow_scorer=None):
        """
            Configure Cattle and Ma (2017) feature extraction options including
            Word2Vec model and Wilson et al. (2005) subjectivity lexicon
            locations.
            
                Theresa Wilson, Janyce Wiebe and Paul Hoffmann (2005).
                Recognizing Contextual Polarity in Phrase-Level Sentiment
                Analysis. Proceedings of HLT/EMNLP 2005, Vancouver, Canada.
            
            :param w2v_model_getter: function for retrieving Word2Vec model
            :type w2v_model_getter: Callable[[], GensimVectorModel]
            :param eat_scorer: method for getting EAT association strengths between two words
            :type eat_scorer: Callable[[str, str], float]
            :param usf_scorer: method for getting USF association strengths between two words
            :type usf_scorer: Callable[[str, str], float]
            :param perplexity_scorer: method for getting perplexity of a document
            :type perplexity_scorer: Callable[Iterable[Iterable[str], float]
            :param verbose: whether we should use verbose mode or not
            :type verbose: bool
        """
        
        #TODO: make eat,usf,swow a list of label/scorer tuples
        
        #TODO: specify multiple perplexity scorers for various levels of ngram
        self.get_w2v = w2v_model_getter
        self.get_eat_scores = eat_scorer
        self.get_usf_scores = usf_scorer
        self.get_perplexities = perplexity_scorer
        self._get_humour_anchors = humour_anchor_identifier
        #TODO: add CountVectorizer configurations
        self.count_vectorizer = CountVectorizer(tokenizer=lambda x:  x, preprocessor=lambda x: x)
        
        if verbose:
            self.logger.setLevel(logging.DEBUG)
            
        self.get_swow_scores = swow_scorer
    
#     def get_humour_anchors(self, documents):
#         """
#             Get the humour anchors for all documents.
#             
#             Since multiple functions extract humour anchors, it's more efficient
#             to extract them one time in advance
#             
#             :param documents: documents to be processed. Each document should be a sequence of tokens
#             :type documents: Iterable[str]
#             
#             :returns: each document's humour anchors
#             :rtype: List[List[str]]
#         """
#         
#         #default to all words being humour anchors
#         humour_anchors = documents
#         if self._get_humour_anchors:
#             humour_anchors = self._get_humour_anchors(documents)
#         
#         return humour_anchors

    def _make_anchor_bow(self,humour_anchors):
        anchor_bows = []
        for anchors, _, _ in humour_anchors:
            anchor_bow=[]
            for anchor in anchors:
                if len(anchor)==7 and type(anchor[1])!=str:
                    anchor= anchor[0]
                
                anchor_bow.extend(anchor)
            anchor_bows.append(anchor_bow)
        
        return anchor_bows

    def _make_anchor_pairs(self, humour_anchors):
        """
        Method for generating humour anchor pairs that respect setup/punchline structure
        """
        
        doc_intra_anchor_pairs = []
        doc_inter_anchor_pairs = []
        
#         for anchors, baseline_scor, decrement in humour_anchors:
        for anchors, _, _ in humour_anchors:
            intra_anchor_pairs = []
            for anchor in anchors:
                if len(anchor)==7 and type(anchor[1])!=str:
                    anchor= anchor[0]
                
                intra_anchor_pairs.extend(combinations(anchor,2))
                
            doc_intra_anchor_pairs.append(intra_anchor_pairs)
            
            inter_anchor_pairs = []
            for anchor1, anchor2 in combinations(anchors,2):
                if len(anchor1)==7 and type(anchor1[1])!=str:
                    anchor1= anchor1[0]
                if len(anchor2)==7 and type(anchor2[1])!=str:
                    anchor2= anchor2[0]
                inter_anchor_pairs.extend(product(anchor1,anchor2))
            
            doc_inter_anchor_pairs.append(inter_anchor_pairs)
        
        return doc_intra_anchor_pairs, doc_inter_anchor_pairs
    
    def _make_punchline_setup_pairs(self, humour_anchors, only_funny=False):
        """
        Method for generating humour anchor pairs that respect setup/punchline structure
        """
        
        doc_inter_pairs = []
        doc_intra_pairs = []
        
#         for anchors, baseline_scor, decrement in humour_anchors:
        for anchors, baseline_score, decrement in humour_anchors:
            inter_pairs=[]
            intra_pairs =[]
            
            if (not only_funny) or (baseline_score > log(0.5)):
                #if the document is funny or if we don't care if it's funny or not
            
#             if decrement and decrement > 0:
                #TODO: check decrement
                setup_words =[]
                punch_words=[] 
                
                for anchor in anchors:
                    sent_index=0
                    num_sents=1
                    beginning_index=0
                    sent_len=1
                    abs_index=0
                    total_len=1
                    if len(anchor)==7 and type(anchor[1])!=str:
                        anchor, sent_index, num_sents, beginning_index, sent_len, abs_index, total_len = anchor
                    if abs_index < total_len/2:
                        setup_words.extend(anchor)
                    else:
                        punch_words.extend(anchor)
                
                inter_pairs=list(product(setup_words,punch_words))
                intra_pairs=list(combinations(setup_words,2))
                intra_pairs.extend(combinations(punch_words,2))
            
            doc_inter_pairs.append(inter_pairs)
            doc_intra_pairs.append(intra_pairs)
        
        return doc_intra_pairs, doc_inter_pairs
    
    def get_association_strengths(self, documents, association_scorer, skip_humour_anchors=False, dataset="", skip_pairs=False):
        """
            Calculate various association strengths. This includes min, average,
            and max scores for forward, backward, word-level difference, and
            document-level difference.
            
            :param documents: documents to be processed. Each document should be a sequence of tokens
            :type documents: Iterable[Iterable[str]]
            :param association_scorer: function for getting association strengths
            :type association_scorer: Callable[[Iterable[Tuple[str, str]]], Iterable[float]]
            :param skip_humour_anchors: whether humour anchors extraction should skipped. If humour anchors are extracted manually in advance, this should be set to True
            :type skip_humour_anchors: bool
            
            :return: A matrix representing the extracted association strengths in the form (forward min, forward avg, forward max, backward min, backward avg, backward max, word-level diff min, word-level diff avg, word-level diff max, doc-level diff min, doc-level diff avg, doc-level diff max) x # of documents
            :rtype: numpy.array
        """
        
        #If we want to extract humour anchors, use the anchor extractor method specified in __init__()
        #If we don't, use None to tell get_interword_score_features() to skip extracting them
        anchor_identifier = self._get_humour_anchors if not skip_humour_anchors else None
        if not skip_pairs:
            documents = get_word_pairs(documents, anchor_identifier)
        
        #get unique word pairs (for computing forward strengths)
        unique_pairs = set()
        for doc in documents:
            unique_pairs.update(doc)
        #add reverse pairs (for computing backward strengths
        unique_pairs.update([(b,a) for a,b in unique_pairs])
        
        unique_pairs = list(unique_pairs) #sets are unordered. Use list to ensure order doesn't change
        
        print(f"{len(unique_pairs)} pairs")
        
        #get association strengths  and create lookup table
        strengths_map = dict(zip(unique_pairs, association_scorer(unique_pairs)))
        
#         import dill
#         with open(f"{dataset} strength map potd pretok.dill", "wb") as f:
#             dill.dump(strengths_map, f)
#         with open(f"{dataset} strength map potd.dill", "rb") as f:
#             strengths_map=dill.load(f)
        
        feature_vects = []
        for doc in documents:
            forward_strenghts = np.array([strengths_map[word_pair] for word_pair in doc])
            backward_strengths = np.array([strengths_map[(b,a)] for a, b in doc])
            word_level_differences = forward_strenghts - backward_strengths
            
            if forward_strenghts.shape[0] == 0:
                forward_strenghts=np.array([0])
            if  backward_strengths.shape[0] == 0:
                backward_strengths=np.array([0])
            if  word_level_differences.shape[0] == 0:
                word_level_differences=np.array([0])
            
            min_f = min(forward_strenghts)
            avg_f = mean(forward_strenghts)
            max_f = max(forward_strenghts)
            
            min_b = min(backward_strengths)
            avg_b = mean(backward_strengths)
            max_b = max(backward_strengths)
            
            min_word_diff = min(word_level_differences)
            avg_word_diff = mean(word_level_differences)
            max_word_diff = max(word_level_differences)
            
            min_doc_diff = min_f - min_b
            avg_doc_diff = avg_f - avg_b
            max_doc_diff = max_f - max_b
            
            feature_vects.append((min_f, avg_f, max_f, min_b, avg_b, max_b, min_word_diff, avg_word_diff, max_word_diff, min_doc_diff, avg_doc_diff, max_doc_diff))
        
        return np.vstack(feature_vects)
    
    def get_w2v_sims(self, documents, skip_humour_anchors=False, skip_pairs=False):
        """
            Calculates the min, max, and average Word2Vec similarities between
            word pairs.
            
            :param documents: documents to be processed. Each document should be a sequence of tokens
            :type documents: Iterable[Iterable[str]]
            :param skip_humour_anchors: whether humour anchors extraction should skipped. If humour anchors are extracted manually in advance, this should be set to True
            :type skip_humour_anchors: bool
            
            :return: A matrix representing the extracted Word2Vec similarity features in the form (min, avg, max) x # of documents
            :rtype: numpy.array
        """
        
        #If we want to extract humour anchors, use the anchor extractor method specified in __init__()
        #If we don't, use None to tell get_interword_score_features() to skip extracting them
        anchor_identifier = self._get_humour_anchors if not skip_humour_anchors else None
        pair_generator = (lambda x,y : x) if skip_pairs else None #either use dummy pair generator or default
        
        w2v_scorer = self.get_w2v().get_similarity
        return get_interword_score_features(documents, w2v_scorer, token_filter=anchor_identifier, pair_generator=pair_generator)
    
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
    
    def get_interpersonal_features(self, documents):
        """
            Calculates the interpersonal effect features described in Section
            4.3 of Yang et al. (2015)
            
            :param documents: documents to be processed. Each document shoudl be a sequence of tokens
            :type documents: Iterable[Iterable[str]]
            
            :return: A matrix of the extracted interpersonal features where eat row is the form (neg_polarity, pos_polarity, weak_subjectivity, strong_subjectivity)
            :rtype: numpy.array
        """
        
        strongsubj=[]
        weaksubj=[]
        negative=[]
        positive=[]
        with open("D:/datasets/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff", "r") as wilson_f:
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

        strongsubj, weaksubj, positive, negative = set(strongsubj), set(weaksubj), set(positive), set(negative)
        
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
    
    def fit(self,X,y):
        """
            Trains the count vectorizer
            
            :returns: self (for compatibility with sklearn pipelines)
            :rtype: TransformerMixin
        """
        
        self.count_vectorizer.fit(X)
        #TODO: implement fit_transform?
        
        return self
    
    def transform(self, X):
        """
            Takes in a  series of tokenized documents and extracts a set of
            features equivalent to the highest performing model in Yang et al.
            (2015)
            
            :param X: pre-tokenized documents
            :type X: Iterable[Iterable[str]]
            
            :return: The features described in Cattle and Ma (2017)
            :rtype: numpy.array
        """
        
        X_tok = default_preprocessing_and_tokenization(X, stopwords=[],pos_to_ignore=[], flatten_sents=True)
#         X_tok = [word_tokenize(doc.lower()) for doc in X]
        
        features = []
#         self.logger.debug("Adding ngram features")
#         #TODO: ensemble with the ngram features?
#         features.append(self.count_vectorizer.transform(X))
        self.logger.debug("Adding perplexities")
        features.append(self.get_perplexities(X_tok))
        self.logger.debug("Adding Word2Vec similarities (full doc)")
        features.append(self.get_w2v_sims(X_tok, skip_humour_anchors=True))
        purge_all_gensim_vector_models()
        
        self.logger.debug("Adding EAT association strengths (full doc)")
        features.append(self.get_association_strengths(X_tok, self.get_eat_scores, skip_humour_anchors=True, dataset="eat"))
        self.logger.debug("Adding USF association strengths (full doc)")
        features.append(self.get_association_strengths(X_tok, self.get_usf_scores, skip_humour_anchors=True, dataset="usf"))
        features.append(self.get_association_strengths(X_tok, self.get_swow_scores, skip_humour_anchors=True))
        features.append(get_interword_score_features(X_tok, get_en_conceptnet_numberbatch().get_similarity, token_filter=None))
        purge_all_gensim_vector_models()
        #TODO: do I need the full dataset, not just en?
        cnrv_forward,cnrv_backward = get_relation_count_vectors(X_tok, english_relations, token_filter=None, include_backward=True)
        features.append(cnrv_forward)
        features.append(cnrv_forward+cnrv_backward)
        features.append(np.hstack([cnrv_forward,cnrv_backward]))
        
        
        if self._get_humour_anchors:
            self.logger.debug("Extracting humour anchors")
            #Since multiple functions take advantage of humour anchor extraction, makes sense to extract them only once
            humour_anchors = self._get_humour_anchors(X)
#         humour_anchors = self.get_humour_anchors(X)

#             humour_anchors = []
#             for anchors, _,_ in self._get_humour_anchors(X):
#                 doc_bow = []
#                 for anchor in anchors:
#                     if len(anchor)==7 and type(anchor[1])!=str:
#                         anchor= anchor[0]
#                      
#                     doc_bow.extend(anchor)
#                  
#                 humour_anchors.append(doc_bow)
#              
            ha_bow = self._make_anchor_bow(humour_anchors)
            features.append(self.get_w2v_sims(ha_bow, skip_humour_anchors=True))
            purge_all_gensim_vector_models()
            
            features.append(self.get_association_strengths(ha_bow, self.get_eat_scores, skip_humour_anchors=True, dataset="eat"))
            features.append(self.get_association_strengths(ha_bow, self.get_usf_scores, skip_humour_anchors=True, dataset="usf"))
            features.append(self.get_association_strengths(ha_bow, self.get_swow_scores, skip_humour_anchors=True))
            features.append(get_interword_score_features(ha_bow, get_en_conceptnet_numberbatch().get_similarity, token_filter=None))
            purge_all_gensim_vector_models()
            cnrv_bow_forward,cnrv_bow_backward = get_relation_count_vectors(ha_bow, english_relations, token_filter=None, include_backward=True)
            features.append(cnrv_bow_forward)
            features.append(cnrv_bow_forward+cnrv_bow_backward)
            features.append(np.hstack([cnrv_bow_forward,cnrv_bow_backward]))

            def interlace_matrix(a,b):
                c=np.empty((a.shape[0], a.shape[1]+b.shape[1]), dtype=a.dtype)
                c[:,0::6] = a[:,0::3]
                c[:,1::6] = a[:,1::3]
                c[:,2::6] = a[:,2::3]
                c[:,3::6] = b[:,0::3]
                c[:,4::6] = b[:,1::3]
                c[:,5::6] = b[:,2::3]
                
                return c
            
#             intra_pairs, inter_pairs = self._make_anchor_pairs(humour_anchors)
#             intra_pairs, inter_pairs = self._make_punchline_setup_pairs(humour_anchors)
            for intra_pairs, inter_pairs in (self._make_anchor_pairs(humour_anchors), self._make_punchline_setup_pairs(humour_anchors, False), self._make_punchline_setup_pairs(humour_anchors, True)):
                #intra
                self.logger.debug("Adding Word2Vec similarities (humour anchors)")
                w2v_intra = self.get_w2v_sims(intra_pairs, skip_humour_anchors=True, skip_pairs=True)
                self.logger.debug("Adding Word2Vec similarities (humour anchors)")
                w2v_inter = self.get_w2v_sims(inter_pairs, skip_humour_anchors=True, skip_pairs=True)
                purge_all_gensim_vector_models()
                features.append(np.hstack([w2v_intra, w2v_inter]))
                
    #             features.append(self.get_w2v_sims(intra_pairs, skip_humour_anchors=True, skip_pairs=True))
                self.logger.debug("Adding EAT association strengths (humour anchors)")
                eat_intra = self.get_association_strengths(intra_pairs, self.get_eat_scores, skip_humour_anchors=True, dataset="eat", skip_pairs=True)
                self.logger.debug("Adding EAT association strengths (humour anchors)")
                eat_inter = self.get_association_strengths(inter_pairs, self.get_eat_scores, skip_humour_anchors=True, dataset="eat", skip_pairs=True)
                features.append(interlace_matrix(eat_intra, eat_inter))
                
    #             features.append(self.get_association_strengths(intra_pairs, self.get_eat_scores, skip_humour_anchors=True, dataset="eat", skip_pairs=True))
                self.logger.debug("Adding USF association strengths (humour anchors)")
                usf_intra = self.get_association_strengths(intra_pairs, self.get_usf_scores, skip_humour_anchors=True, dataset="usf", skip_pairs=True)
                self.logger.debug("Adding USF association strengths (humour anchors)")
                usf_inter = self.get_association_strengths(inter_pairs, self.get_usf_scores, skip_humour_anchors=True, dataset="usf", skip_pairs=True)
                features.append(interlace_matrix(usf_intra, usf_inter))
    #             features.append(self.get_association_strengths(intra_pairs, self.get_usf_scores, skip_humour_anchors=True, dataset="usf", skip_pairs=True))
                
                
                swow_ha_intra = self.get_association_strengths(intra_pairs, self.get_swow_scores, skip_humour_anchors=True, skip_pairs=True)
                swow_ha_inter = self.get_association_strengths(inter_pairs, self.get_swow_scores, skip_humour_anchors=True, skip_pairs=True)
                features.append(interlace_matrix(swow_ha_intra, swow_ha_inter))
                
                cnnb_intra = get_interword_score_features(intra_pairs, get_en_conceptnet_numberbatch().get_similarity, token_filter=None, pair_generator = lambda x,y : x)
                cnnb_inter = get_interword_score_features(inter_pairs, get_en_conceptnet_numberbatch().get_similarity, token_filter=None, pair_generator = lambda x,y : x)
                purge_all_gensim_vector_models()
                features.append(np.hstack([cnnb_intra, cnnb_inter]))
                
                cnrvec_intra_forward, cnrvec_intra_back = get_relation_count_vectors(intra_pairs, english_relations, token_filter=None, pair_generator=lambda x,y: x, include_backward=True)
                cnrvec_inter_forward, cnrvec_inter_back = get_relation_count_vectors(inter_pairs, english_relations, token_filter=None, pair_generator=lambda x,y: x, include_backward=True)
                features.append(np.hstack([cnrvec_intra_forward, cnrvec_inter_forward]))
                features.append(np.hstack([cnrvec_intra_forward + cnrvec_intra_back, cnrvec_inter_forward + cnrvec_inter_back]))
                features.append(np.hstack([cnrvec_intra_forward, cnrvec_intra_back, cnrvec_inter_forward, cnrvec_inter_back]))
                
                   
                #inter
    #             features.append(self.get_w2v_sims(inter_pairs, skip_humour_anchors=True, skip_pairs=True))
    #             features.append(self.get_association_strengths(inter_pairs, self.get_eat_scores, skip_humour_anchors=True, dataset="eat", skip_pairs=True))
    #             features.append(self.get_association_strengths(inter_pairs, self.get_usf_scores, skip_humour_anchors=True, dataset="usf", skip_pairs=True))
     
                
                
                
            
            
            
#             features.append(swow_ha_intra)
#             features.append(swow_ha_inter)
            
#             _, pairs = self._make_punchline_setup_pairs(humour_anchors)
#              
#             #intra
#             self.logger.debug("Adding Word2Vec similarities (humour anchors)")
#             features.append(self.get_w2v_sims(pairs, skip_humour_anchors=True, skip_pairs=True))
#             self.logger.debug("Adding EAT association strengths (humour anchors)")
#             features.append(self.get_association_strengths(pairs, self.get_eat_scores, skip_humour_anchors=True, dataset="eat", skip_pairs=True))
#             self.logger.debug("Adding USF association strengths (humour anchors)")
#             features.append(self.get_association_strengths(pairs, self.get_usf_scores, skip_humour_anchors=True, dataset="usf", skip_pairs=True))
            
        else:
            self.logger.debug("No humour anchor extractor specified. Defaulting to all 0s")
            features.append(np.zeros((len(X),54))) #3 + 12 + 12 (54 for inter/intra)
        
        
#         self.logger.debug("Adding allit rhyme")
#         features.append(self.get_phonetic_features(X_tok))
#         self.logger.debug("Adding interper")
#         features.append(self.get_interpersonal_features(X_tok))
        
        
#         return scipy.sparse.hstack(features)
#         return np.hstack(features)
        return features

if __name__ == "__main__":
    
    from nltk import word_tokenize
    import re
    from string import punctuation
    from sklearn.ensemble.forest import RandomForestClassifier
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    from sklearn.svm import LinearSVC, SVC
    from sklearn.preprocessing import StandardScaler,LabelBinarizer
    from sklearn.pipeline import Pipeline
    from util.dataset_readers.potd_reader import read_raw_potd_data#, read_potd_data
    from util.dataset_readers.oneliner_reader import read_16000_oneliner_data
    
    potd_loc = "D:/datasets/pun of the day/puns_pos_neg_data.csv"
    oneliners_pos = "D:/datasets/16000 oneliners/Jokes16000.txt"
    oneliners_neg = "D:/datasets/16000 oneliners/MIX16000.txt"
    potd_pos = "D:/datasets/pun of the day/puns_of_day.csv"
    potd_neg = "D:/datasets/pun of the day/new_select.txt"
    proverbs = "D:/datasets/pun of the day/proverbs.txt"
    
    potd_docs_and_labels= read_raw_potd_data(potd_pos, potd_neg, proverbs)
#     potd2 = read_potd_data(potd_loc)  
       
    oneliner_docs_and_labels = read_16000_oneliner_data(oneliners_pos, oneliners_neg)
       
       
#     for r, t in zip(potd_docs_and_labels[:len(potd2)], potd2):
#         if r[1] == -1:
#             print("neg")
#         print(r)
#         print(t)
#         print()
#     
#     for d in potd_docs_and_labels[len(potd2):]:
#         print(d)
       
       
    import random
    random.seed(10)
    random.shuffle(potd_docs_and_labels)
    random.shuffle(oneliner_docs_and_labels)
    
    import pickle
#     with open("potd_raw_kfold_redo.pkl", "rb") as df:
#         potd_docs_and_labels = pickle.load(df)
    with open("oneliner_raw_kfold_redo.pkl", "rb") as df:
        oneliner_docs_and_labels = pickle.load(df)
    bom=re.compile("\ufeff")
    oneliner_docs_and_labels = [(bom.sub("", doc), label) for doc, label in oneliner_docs_and_labels]

    with open("oneliner_anchor_map_with_metadata.pkl", "rb") as af:
            ol_anchor_map = pickle.load(af)
    with open("potd_anchor_map_with_metadata.pkl", "rb") as af:
            potd_anchor_map = pickle.load(af)
            
    #flatten
#     anchors = [[ t for a in doc_anchors for t in a] for doc_anchors in anchors]
#     anchor_map = {doc:anchor for (doc, _), anchor in zip(docs_and_labels,anchors)}








#TODO: uncomment
#     for k,v in anchor_map.items():
#         anchor_map[k] = tuple(t for a in v for t in a) #TODO: in a[0]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     for k in anchor_map:
#         if "Monde" in k:
#             print(k)
    
    docs, labels = zip(*potd_docs_and_labels)
    from util.text_processing import default_preprocessing_and_tokenization
#     docs = default_preprocessing_and_tokenization(docs) #Remove stopwords
#     docs = default_preprocessing_and_tokenization(docs, stopwords=[])
#     from util.text_processing import default_preprocessing_and_tokenization
#     docs = default_preprocessing_and_tokenization(docs)
#     #TODO: this also gets rid of negation words. Negation might be a helpful feature
#     from nltk import word_tokenize
#     docs = [word_tokenize(doc) for doc in docs]
     
    test_size = round(len(docs) * 0.1) #90/10 training test split
#     test_size=0
    test_X, test_y = docs[:test_size], labels[:test_size]
    train_X, train_y = docs[test_size:], labels[test_size:]
    
     
#     lda_loc="c:/vectors/lda_prep_no_lemma/no_lemma.101.lda"
#     wordids_loc="c:/vectors/lda_prep_no_lemma/lda_no_lemma_wordids.txt.bz2"
#     tfidf_loc="c:/vectors/lda_prep_no_lemma/lda_no_lemma.tfidf_model"
#     w2v_loc="c:/vectors/GoogleNews-vectors-negative300.bin"
#     glove_loc="c:/vectors/glove.840B.300d.withheader.bin"
#     w2g_vocab_loc="c:/vectors/wiki.moreselective.gz"
#     w2g_model_loc="c:/vectors/wiki.hyperparam.selectivevocab.w2g"
#     autoex_loc = "c:/vectors/autoextend.word2vecformat.bin"
    lesk_loc = "d:/git/PyWordNetSimilarity/PyWordNetSimilarity/src/lesk-relation.dat"
    betweenness_pkl="d:/git/HumourDetection/HumourDetection/src/word_associations/wordnet_betweenness.pkl"
    load_pkl="d:/git/HumourDetection/HumourDetection/src/word_associations/wordnet_load.pkl"
     
#     import pickle
#     from word_associations.strength_prediction import train_cattle_ma_2017_association_pipeline
    from util.model_wrappers.common_models import get_wikipedia_lda, get_google_word2vec,\
        get_google_autoextend, get_stanford_glove, get_wikipedia_word2gauss, get_conceptnet_numberbatch,\
        get_en_conceptnet_numberbatch
    from util.conceptnet_utils import get_numberbatch_sims, get_relation_count_vectors
        
#     print("Running Yang et al. (2015) as baseline")
    from humour_features.yang_2015_features import *
#     wilson_lexicon_loc = "D:/datasets/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff"
# #     random.shuffle(oneliner_docs_and_labels)
# #     ol_X, ol_y = zip(*oneliner_docs_and_labels)
# #     train_X_tok = default_preprocessing_and_tokenization(train_X, stopwords=[])
# #     test_X_tok = default_preprocessing_and_tokenization(test_X, stopwords=[])
# #     run_yang_et_al_2015_baseline((train_X_tok,train_y),(test_X_tok,test_y),get_google_word2vec, wilson_lexicon_loc)
# #     run_yang_et_al_2015_baseline((train_X,train_y),(test_X,test_y),get_google_word2vec, wilson_lexicon_loc, n_estimators=10)
#     baseline_extractor = YangHumourFeatureExtractor(get_google_word2vec, wilson_lexicon_loc, verbose=True)
#     train_X_baseline = baseline_extractor.fit_transform(train_X, train_y)
#     test_X_baseline = baseline_extractor.transform(test_X)
#     print("raw-stop 10")
#     clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, verbose=1)
#     clf.fit(train_X_baseline,train_y)
#     pred_y = clf.predict(test_X_baseline)
#         
#     p,r,f,_ = precision_recall_fscore_support(test_y, pred_y, average="binary")
#     a = accuracy_score(test_y, pred_y)
#     print(f"Accuracy: {a}")
#     print(f"Precision: {p}")
#     print(f"Recall: {r}")
#     print(f"F-Score: {f}")
#     print("\nraw-stop 100")
#     clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=1)
#     clf.fit(train_X_baseline,train_y)
#     pred_y = clf.predict(test_X_baseline)
#         
#     p,r,f,_ = precision_recall_fscore_support(test_y, pred_y, average="binary")
#     a = accuracy_score(test_y, pred_y)
#     print(f"Accuracy: {a}")
#     print(f"Precision: {p}")
#     print(f"Recall: {r}")
#     print(f"F-Score: {f}")

#     yang = train_yang_et_al_2015_pipeline(train_X_tok,ol_y,get_google_word2vec, wilson_lexicon_loc)
#     print("saving")
    import dill
#     with open("yang_16000_mihalcea.dill", "wb") as f:
#     with open("yang_pipeline_potd.dill", "rb") as f:
# #     with open("yang_pipeline_potd.dill", "rb") as f:
#         yang = dill.load(f)
        
#     from nltk.parse.stanford import StanfordParser
#     parser = StanfordParser()
#     hae = YangHumourAnchorExtractor(lambda x: next(parser.parse(x)), yang, 3)
    def map_anchors(X, anchor_map):
        anchors = []
        for doc in X:
            anchors.append(anchor_map[doc])
        return anchors
    
#     def train_association_predictor(assoc_graph):
#         stim, resp, strengths = zip(*assoc_graph.get_all_associations())
#         word_pairs = zip(stim, resp)
#         strengths = np.array(strengths) * 100
#         
#         return train_cattle_ma_2017_association_pipeline(word_pairs, strengths,
#                                                          lda_model_getter=get_wikipedia_lda,
#                                                          w2v_model_getter=get_google_word2vec,
#                                                          autoex_model_getter=get_google_autoextend,
#                                                          betweenness_loc=betweenness_pkl,
#                                                          load_loc=load_pkl,
#                                                          glove_model_getter=get_stanford_glove,
#                                                          w2g_model_getter=get_wikipedia_word2gauss,
#                                                          lesk_relations=lesk_loc,
#                                                          verbose=True)
#     
#     from word_associations.association_readers.networkx_readers import EATNetworkx, USFNetworkx
#     usf = USFNetworkx("../Data/PairsFSG2.net")
#     eat = EATNetworkx("../Data/eat/pajek/EATnew2.net")

    from functools import partial
# #     from word_associations.association_readers.query_igraph import get_strengths_wsl
# #     eat_func = partial(get_strengths_wsl, dataset="eat", pajek_loc="/mnt/d/git/HumourDetection/HumourDetection/src/Data/eat/pajek/EATnew2.net", tmpdir="d:/temp")
# #     usf_func = partial(get_strengths_wsl, dataset="usf", pajek_loc="/mnt/d/git/HumourDetection/HumourDetection/src/Data/PairsFSG2.net", tmpdir="d:/temp")
# #     swow_all_func = partial(get_strengths_wsl, dataset="usf", pajek_loc="/mnt/d/datasets/SWoW/swow_all.net", tmpdir="d:/temp") #swow and usf are the same pajek format (edge weights are proportions, not counts)
# #     swow_100_func = partial(get_strengths_wsl, dataset="usf", pajek_loc="/mnt/d/datasets/SWoW/swow_100.net", tmpdir="d:/temp")
#     
#     from word_associations.association_readers.igraph_readers import USFIGraph, EATIGraph
#     eat_func = EATIGraph("D:/git/HumourDetection/HumourDetection/src/Data/eat/pajek/EATnew2.net").get_association_strengths
#     usf_func = USFIGraph("D:/git/HumourDetection/HumourDetection/src/Data/PairsFSG2.net").get_association_strengths
#     swow_all_func = USFIGraph("D:/datasets/SWoW/swow_all.net").get_association_strengths #swow and usf are the same pajek format (edge weights are proportions, not counts)
#     swow_100_func = USFIGraph("D:/datasets/SWoW/swow_100.net").get_association_strengths
#     swow_stren_func = USFIGraph("D:/datasets/SWoW/swow_stren.net").get_association_strengths
#     from nltk.corpus import stopwords
#     ENGLISH_STOPWORDS = set(stopwords.words('english'))
#     ENGLISH_STOPWORDS.add("n't")
#     punc_re = re.compile(f'[{re.escape(punctuation)}]')
#     
#     def no_stopwords(document):
#         sanitized_doc = []
#         for token in document:
#             if token.lower() not in ENGLISH_STOPWORDS:
#                 sanitized_doc.append(token)
#         
#         return sanitized_doc
#     
#     def tokenizer(document):
#         
#         tokens = word_tokenize(document.lower()) #Tokenize and lowercase document
#         #TODO: keep sentence information?
#         
#         processed_tokens = []
#         for token in tokens:
#             if token not in ENGLISH_STOPWORDS:
#                 token = punc_re.sub("", token) #remove punctuation
#                 #https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
#                 
#                 if token: #if this isn't an empty string
#                     processed_tokens.append(token)
#         
#         return processed_tokens
        
#     from util.keras_pipeline_persistance import save_keras_pipeline, load_keras_pipeline
#     from util.keras_metrics import r2_score, pearsonr #for compatibility with saved pipelines
#     from word_associations.strength_prediction import _create_mlp
# #     eat_predictor = train_association_predictor(eat)
# #     save_keras_pipeline("d:/datasets/trained models/eat", eat_predictor)
# #     eat_predictor = load_keras_pipeline("d:/datasets/trained models/eat")
# #     eat_predictor = load_keras_pipeline("d:/git/HumourDetection/HumourDetection/src/word_associations/models/eat-all")#,custom_objects={"r2_score":r2_score, "pearsonr":pearsonr})
# #     usf_predictor = load_keras_pipeline("d:/git/HumourDetection/HumourDetection/src/word_associations/models/usf-all")#,custom_objects={"r2_score":r2_score, "pearsonr":pearsonr})
# #     eat_predictor = load_keras_pipeline("d:/git/HumourDetection/HumourDetection/src/word_associations/models/eat-all")#,custom_objects={"r2_score":r2_score, "pearsonr":pearsonr})
#     swow_comp_predictor = load_keras_pipeline("d:/git/HumourDetection/HumourDetection/src/word_associations/models/swow all-all",custom_objects={"r2_score":r2_score, "pearsonr":pearsonr})
#     swow_100_predictor = load_keras_pipeline("d:/git/HumourDetection/HumourDetection/src/word_associations/models/swow 100-all",custom_objects={"r2_score":r2_score, "pearsonr":pearsonr})
#     swow_stren_predictor = load_keras_pipeline("d:/git/HumourDetection/HumourDetection/src/word_associations/models/swow stren-all",custom_objects={"r2_score":r2_score, "pearsonr":pearsonr})
#         
#     import logging
#     for p in (swow_comp_predictor, swow_100_predictor):
#         p.named_steps["extract features"].logger.setLevel(logging.DEBUG)
        
#     unique_pairs = set()
#     datasets = [potd_docs_and_labels, oneliner_docs_and_labels]
#     for dataset in datasets:
#         docs, _ = zip(*dataset)
#         docs = default_preprocessing_and_tokenization(docs, stopwords=[],pos_to_ignore=[], flatten_sents=True)
#         docs = get_word_pairs(docs, None)
#         for doc in docs:
#             unique_pairs.update(doc)
#         #add reverse pairs (for computing backward strengths
#         unique_pairs.update([(b,a) for a,b in unique_pairs])
#         
#     unique_pairs = list(unique_pairs) #sets are unordered. Use list to ensure order doesn't change
        
    #get association strengths  and create lookup table
#     eat_strengths_map = dict(zip(unique_pairs, eat_predictor.predict(unique_pairs)))
#     with open("eat_prediction_map.pkl", "wb") as f:
#         pickle.dump(eat_strengths_map, f)
#     usf_strengths_map = dict(zip(unique_pairs, usf_predictor.predict(unique_pairs)))
#     with open("usf_prediction_map.pkl", "wb") as f:
#         pickle.dump(usf_strengths_map, f)
#     eat_strengths_map = dict(zip(unique_pairs, eat_func(unique_pairs)))
#     with open("eat_graph_map.pkl", "wb") as f:
#         pickle.dump(eat_strengths_map, f)
#     usf_strengths_map = dict(zip(unique_pairs, usf_func(unique_pairs)))
#     with open("usf_graph_map.pkl", "wb") as f:
#         pickle.dump(usf_strengths_map, f)
#     swow_comp_strengths_map = dict(zip(unique_pairs, swow_comp_predictor.predict(unique_pairs)))
#     with open("swow_all_prediction_map.pkl", "wb") as f:
#         pickle.dump(swow_comp_strengths_map, f)
#     swow_100_strengths_map = dict(zip(unique_pairs, swow_100_predictor.predict(unique_pairs)))
#     with open("swow_100_prediction_map.pkl", "wb") as f:
#         pickle.dump(swow_100_strengths_map, f)
#     swow_stren_strengths_map = dict(zip(unique_pairs, swow_stren_predictor.predict(unique_pairs)))
#     with open("swow_stren_prediction_map.pkl", "wb") as f:
#         pickle.dump(swow_stren_strengths_map, f)
#     swow_all_strengths_map = dict(zip(unique_pairs, swow_all_func(unique_pairs)))
#     with open("swow_all_graph_map.pkl", "wb") as f:
#         pickle.dump(swow_all_strengths_map, f)
#     swow_100_strengths_map = dict(zip(unique_pairs, swow_100_func(unique_pairs)))
#     with open("swow_100_graph_map.pkl", "wb") as f:
#         pickle.dump(swow_100_strengths_map, f)
#     swow_stren_strengths_map = dict(zip(unique_pairs, swow_stren_func(unique_pairs)))
#     with open("swow_stren_graph_map.pkl", "wb") as f:
#         pickle.dump(swow_stren_strengths_map, f)
    
    punc_re = re.compile(f'[{re.escape(punctuation)}]')
    def map_strengths(pairs,strength_map=None):
        s = []
        for a,b in pairs:
#             print(a,b)
            p = (punc_re.sub("", a), punc_re.sub("", b))
            try:
                s.append(strength_map[p])
            except KeyError:
                s.append(0)
                print(f"{p} not found")
        return s
    with open("eat_prediction_map.pkl", "rb") as f:
        eat_pred_map = pickle.load(f)
    with open("usf_prediction_map.pkl", "rb") as f:
        usf_pred_map = pickle.load(f)
    with open("eat_graph_map.pkl", "rb") as f:
        eat_g_map = pickle.load(f)
    with open("usf_graph_map.pkl", "rb") as f:
        usf_g_map = pickle.load(f)
    with open("swow_all_prediction_map.pkl", "rb") as f:
        swow_all_pred_map = pickle.load(f)
#     with open("swow_100_prediction_map.pkl", "rb") as f:
#         swow_100_pred_map = pickle.load(f)
    with open("swow_stren_prediction_map.pkl", "rb") as f:
        swow_stren_pred_map = pickle.load(f)
#     with open("swow_all_graph_map.pkl", "rb") as f:
#         swow_all_g_map = pickle.load(f)
#     with open("swow_100_graph_map.pkl", "rb") as f:
#         swow_100_g_map = pickle.load(f)
#     with open("swow_stren_graph_map.pkl", "rb") as f:
#         swow_stren_g_map = pickle.load(f)
    eat_predictor_map = partial(map_strengths, strength_map=eat_pred_map)
    usf_predictor_map = partial(map_strengths, strength_map=usf_pred_map)
    eat_graph_map = partial(map_strengths, strength_map=eat_g_map)
    usf_graph_map = partial(map_strengths, strength_map=usf_g_map)
#     swow_all_predictor_map = partial(map_strengths, strength_map=swow_all_pred_map)
#     swow_100_predictor_map = partial(map_strengths, strength_map=swow_100_pred_map)
    swow_stren_predictor_map = partial(map_strengths, strength_map=swow_stren_pred_map)
#     swow_all_graph_map = partial(map_strengths, strength_map=swow_all_g_map)
#     swow_100_graph_map = partial(map_strengths, strength_map=swow_100_g_map)
#     swow_stren_graph_map = partial(map_strengths, strength_map=swow_stren_g_map)
       
       
    from perplexity.kenlm_model import KenLMSubprocessWrapper
    kenlm_query_loc = "/mnt/d/git/kenlm-stable/build/bin/query"
    kenlm_model_loc = "/mnt/d/datasets/news-discuss_3.bin"
    klm = KenLMSubprocessWrapper(kenlm_model_loc, kenlm_query_loc, "d:/temp")
        
    null_hae = partial(default_preprocessing_and_tokenization, stopwords=[], pos_to_ignore=[])
        
        
#     print(usf.get_association_strengths([("dog", "cat"), ("car", "bus")]))
          
        
#     humour_pipeline = Pipeline([("feature extraction", CattleMaHumourFeatureExtractor(get_google_word2vec, eat.get_association_strengths, usf.get_association_strengths, klm.get_perplexities, None, True)),
#                                 ("estimator", RandomForestClassifier(n_estimators=100, min_samples_leaf=100, n_jobs=-1, verbose=1))])
#        
#     humour_pipeline.fit(train_X, train_y)
#     pred_y = humour_pipeline.predict(test_X)
    
    print("starting feature extract")
    #eat_predictor_map, usf_predictor_map
    #eat_graph_map, usf_graph_map
    #eat_func, usf_func
    #eat_predictor.predict, usf_predictor.predict
    
    #get_conceptnet_numberbatch
    with open("D:/datasets/conceptnet_relations_en_only.pkl", "rb") as f:
        english_relations=pickle.load(f)
    result_strs=[]
    for name, dataset, anchor_func in [("potd", potd_docs_and_labels, partial(map_anchors, anchor_map=potd_anchor_map)),("ol", oneliner_docs_and_labels,partial(map_anchors, anchor_map=ol_anchor_map))]:
        print(f"\n\n\n\n{name}")
        
        docs, labels = zip(*dataset)
        print(f"{len(docs)}\t{len(labels)}")
        
        fe = CattleMaHumourFeatureExtractor(get_google_word2vec, eat_graph_map, usf_graph_map, klm.get_perplexities, anchor_func, True, swow_stren_predictor_map)
    #     train_feat = fe.fit_transform(train_X, train_y)
    #     test_feat = fe.transform(test_X)
        from util.model_wrappers.gensim_wrappers.gensim_vector_models import purge_all_gensim_vector_models
    #     purge_all_gensim_vector_models()
    #                  
    # #     np.save("ol_raw_ml_redundant_hae_setup_punch_ol_train_X",train_feat)
    # #     np.save("ol_raw_ml_redundant_hae_setup_punch_ol_test_X",test_feat)
    # #     np.save("ol_raw_ml_redundant_hae_setup_punch_ol_train_y", train_y)
    # #     np.save("ol_raw_ml_redundant_hae_setup_punch_ol_test_y", test_y)
    # #     train_feat=np.load("ol_raw_ml_redundant_hae_ol_train_X.npy")
    # #     test_feat=np.load("ol_raw_ml_redundant_hae_ol_test_X.npy")
    # #     train_y=np.load("ol_raw_ml_redundant_hae_ol_train_y.npy")
    # #     test_y=np.load("ol_raw_ml_redundant_hae_ol_test_y.npy")
    #     train_feat = np.vstack((train_feat,test_feat))
    #     train_y = np.concatenate((train_y,test_y))
    #     print(train_feat.shape[1])
    #  
    #   
    # #     train_feat = fe.fit_transform(train_X, train_y)
    # #     np.save("ol_raw_graph_kfold_hae_X",train_feat)
    # #     np.save("ol_raw_graph_kfold_hae_y",train_y)
    # #     train_feat = np.load("ol_raw_ml_kfold_hae_X.npy")
    # #     train_y = np.load("ol_raw_ml_kfold_hae_y.npy")
    #     test_y = train_y

#         perplex, w2v, eat, usf, swow, cnnb, cnrvec_f_only, cnrvec_comb, cnrvec_fb, w2v_ha_bow, eat_ha_bow, usf_ha_bow, swow_ha_bow, cnnb_ha_bow, cnrvec_ha_bow_f_only, cnrvec_ha_bow_comb, cnrvec_ha_bow_fb, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, swow_ha_intrainter, cnnb_ha_intrainter, cnrvec_ha_intrainter_f_only, cnrvec_ha_intrainter_comb, cnrvec_ha_intrainter_fb, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, swow_ha_setuppunch, cnnb_ha_setuppunch, cnrvec_ha_setuppunch_f_only, cnrvec_ha_setuppunch_comb, cnrvec_ha_setuppunch_fb, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, swow_ha_funnyonly, cnnb_ha_funnyonly, cnrvec_ha_funnyonly_f_only, cnrvec_ha_funnyonly_comb, cnrvec_ha_funnyonly_fb =  fe.fit_transform(docs, labels)
#         purge_all_gensim_vector_models()
#           
#         import os
#         if not os.path.isdir(f"{name}"):
#             os.mkdir(f"{name}")
#         np.save(f"{name}/perplex.npy",perplex)
#         np.save(f"{name}/w2v.npy",w2v)
#         np.save(f"{name}/eat.npy",eat)
#         np.save(f"{name}/usf.npy",usf)
#         np.save(f"{name}/swow.npy",swow)
#         np.save(f"{name}/cnnb.npy",cnnb)
#         np.save(f"{name}/cnrvec_f_only.npy",cnrvec_f_only)
#         np.save(f"{name}/cnrvec_comb.npy",cnrvec_comb)
#         np.save(f"{name}/cnrvec_fb.npy",cnrvec_fb)
#         np.save(f"{name}/w2v_ha_bow.npy",w2v_ha_bow)
#         np.save(f"{name}/eat_ha_bow.npy",eat_ha_bow)
#         np.save(f"{name}/usf_ha_bow.npy",usf_ha_bow)
#         np.save(f"{name}/swow_ha_bow.npy",swow_ha_bow)
#         np.save(f"{name}/cnnb_ha_bow.npy",cnnb_ha_bow)
#         np.save(f"{name}/cnrvec_ha_bow_f_only.npy",cnrvec_ha_bow_f_only)
#         np.save(f"{name}/cnrvec_ha_bow_comb.npy",cnrvec_ha_bow_comb)
#         np.save(f"{name}/cnrvec_ha_bow_fb.npy",cnrvec_ha_bow_fb)
#         np.save(f"{name}/w2v_ha_intrainter.npy",w2v_ha_intrainter)
#         np.save(f"{name}/eat_ha_intrainter.npy",eat_ha_intrainter)
#         np.save(f"{name}/usf_ha_intrainter.npy",usf_ha_intrainter)
#         np.save(f"{name}/swow_ha_intrainter.npy",swow_ha_intrainter)
#         np.save(f"{name}/cnnb_ha_intrainter.npy",cnnb_ha_intrainter)
#         np.save(f"{name}/cnrvec_ha_intrainter_f_only.npy",cnrvec_ha_intrainter_f_only)
#         np.save(f"{name}/cnrvec_ha_intrainter_comb.npy",cnrvec_ha_intrainter_comb)
#         np.save(f"{name}/cnrvec_ha_intrainter_fb.npy",cnrvec_ha_intrainter_fb)
#         np.save(f"{name}/w2v_ha_setuppunch.npy",w2v_ha_setuppunch)
#         np.save(f"{name}/eat_ha_setuppunch.npy",eat_ha_setuppunch)
#         np.save(f"{name}/usf_ha_setuppunch.npy",usf_ha_setuppunch)
#         np.save(f"{name}/swow_ha_setuppunch.npy",swow_ha_setuppunch)
#         np.save(f"{name}/cnnb_ha_setuppunch.npy",cnnb_ha_setuppunch)
#         np.save(f"{name}/cnrvec_ha_setuppunch_f_only.npy",cnrvec_ha_setuppunch_f_only)
#         np.save(f"{name}/cnrvec_ha_setuppunch_comb.npy",cnrvec_ha_setuppunch_comb)
#         np.save(f"{name}/cnrvec_ha_setuppunch_fb.npy",cnrvec_ha_setuppunch_fb)
#         np.save(f"{name}/w2v_ha_funnyonly.npy",w2v_ha_funnyonly)
#         np.save(f"{name}/eat_ha_funnyonly.npy",eat_ha_funnyonly)
#         np.save(f"{name}/usf_ha_funnyonly.npy",usf_ha_funnyonly)
#         np.save(f"{name}/swow_ha_funnyonly.npy",swow_ha_funnyonly)
#         np.save(f"{name}/cnnb_ha_funnyonly.npy",cnnb_ha_funnyonly)
#         np.save(f"{name}/cnrvec_ha_funnyonly_f_only.npy",cnrvec_ha_funnyonly_f_only)
#         np.save(f"{name}/cnrvec_ha_funnyonly_comb.npy",cnrvec_ha_funnyonly_comb)
#         np.save(f"{name}/cnrvec_ha_funnyonly_fb.npy",cnrvec_ha_funnyonly_fb)

#         np.save(f"{name}/eat_graph.npy", eat)
#         np.save(f"{name}/usf_graph.npy", usf)
#         np.save(f"{name}/eat_graph_ha_bow.npy",eat_ha_bow)
#         np.save(f"{name}/usf_graph_ha_bow.npy",usf_ha_bow)
#         np.save(f"{name}/eat_graph_ha_intrainter.npy",eat_ha_intrainter)
#         np.save(f"{name}/usf_graph_ha_intrainter.npy",usf_ha_intrainter)
#         np.save(f"{name}/eat_graph_ha_setuppunch.npy",eat_ha_setuppunch)
#         np.save(f"{name}/usf_graph_ha_setuppunch.npy",usf_ha_setuppunch)
#         np.save(f"{name}/eat_graph_ha_funnyonly.npy",eat_ha_funnyonly)
#         np.save(f"{name}/usf_graph_ha_funnyonly.npy",usf_ha_funnyonly)
        

        perplex = np.load(f"{name}/perplex.npy")
        w2v = np.load(f"{name}/w2v.npy")
#         eat = np.load(f"{name}/eat.npy")
#         usf = np.load(f"{name}/usf.npy")
        swow = np.load(f"{name}/swow.npy")
        cnnb = np.load(f"{name}/cnnb.npy")
        cnrvec_f_only = np.load(f"{name}/cnrvec_f_only.npy")
        cnrvec_comb = np.load(f"{name}/cnrvec_comb.npy")
        cnrvec_fb = np.load(f"{name}/cnrvec_fb.npy")
        w2v_ha_bow = np.load(f"{name}/w2v_ha_bow.npy")
#         eat_ha_bow = np.load(f"{name}/eat_ha_bow.npy")
#         usf_ha_bow = np.load(f"{name}/usf_ha_bow.npy")
        swow_ha_bow = np.load(f"{name}/swow_ha_bow.npy")
        cnnb_ha_bow = np.load(f"{name}/cnnb_ha_bow.npy")
        cnrvec_ha_bow_f_only = np.load(f"{name}/cnrvec_ha_bow_f_only.npy")
        cnrvec_ha_bow_comb = np.load(f"{name}/cnrvec_ha_bow_comb.npy")
        cnrvec_ha_bow_fb = np.load(f"{name}/cnrvec_ha_bow_fb.npy")
        w2v_ha_intrainter = np.load(f"{name}/w2v_ha_intrainter.npy")
#         eat_ha_intrainter = np.load(f"{name}/eat_ha_intrainter.npy")
#         usf_ha_intrainter = np.load(f"{name}/usf_ha_intrainter.npy")
        swow_ha_intrainter = np.load(f"{name}/swow_ha_intrainter.npy")
        cnnb_ha_intrainter = np.load(f"{name}/cnnb_ha_intrainter.npy")
        cnrvec_ha_intrainter_f_only = np.load(f"{name}/cnrvec_ha_intrainter_f_only.npy")
        cnrvec_ha_intrainter_comb = np.load(f"{name}/cnrvec_ha_intrainter_comb.npy")
        cnrvec_ha_intrainter_fb = np.load(f"{name}/cnrvec_ha_intrainter_fb.npy")
        w2v_ha_setuppunch = np.load(f"{name}/w2v_ha_setuppunch.npy")
#         eat_ha_setuppunch = np.load(f"{name}/eat_ha_setuppunch.npy")
#         usf_ha_setuppunch = np.load(f"{name}/usf_ha_setuppunch.npy")
        swow_ha_setuppunch = np.load(f"{name}/swow_ha_setuppunch.npy")
        cnnb_ha_setuppunch = np.load(f"{name}/cnnb_ha_setuppunch.npy")
        cnrvec_ha_setuppunch_f_only = np.load(f"{name}/cnrvec_ha_setuppunch_f_only.npy")
        cnrvec_ha_setuppunch_comb = np.load(f"{name}/cnrvec_ha_setuppunch_comb.npy")
        cnrvec_ha_setuppunch_fb = np.load(f"{name}/cnrvec_ha_setuppunch_fb.npy")
        w2v_ha_funnyonly = np.load(f"{name}/w2v_ha_funnyonly.npy")
#         eat_ha_funnyonly = np.load(f"{name}/eat_ha_funnyonly.npy")
#         usf_ha_funnyonly = np.load(f"{name}/usf_ha_funnyonly.npy")
        swow_ha_funnyonly = np.load(f"{name}/swow_ha_funnyonly.npy")
        cnnb_ha_funnyonly = np.load(f"{name}/cnnb_ha_funnyonly.npy")
        cnrvec_ha_funnyonly_f_only = np.load(f"{name}/cnrvec_ha_funnyonly_f_only.npy")
        cnrvec_ha_funnyonly_comb = np.load(f"{name}/cnrvec_ha_funnyonly_comb.npy")
        cnrvec_ha_funnyonly_fb = np.load(f"{name}/cnrvec_ha_funnyonly_fb.npy")

        eat = np.load(f"{name}/eat_graph.npy")
        usf = np.load(f"{name}/usf_graph.npy")
        eat_ha_bow = np.load(f"{name}/eat_graph_ha_bow.npy")
        usf_ha_bow = np.load(f"{name}/usf_graph_ha_bow.npy")
        eat_ha_intrainter = np.load(f"{name}/eat_graph_ha_intrainter.npy")
        usf_ha_intrainter = np.load(f"{name}/usf_graph_ha_intrainter.npy")
        eat_ha_setuppunch = np.load(f"{name}/eat_graph_ha_setuppunch.npy")
        usf_ha_setuppunch = np.load(f"{name}/usf_graph_ha_setuppunch.npy")
        eat_ha_funnyonly = np.load(f"{name}/eat_graph_ha_funnyonly.npy")
        usf_ha_funnyonly = np.load(f"{name}/usf_graph_ha_funnyonly.npy")
        
        
#         with open(f"{name}_feats.pkl", "rb") as f:
#             perplex, w2v, eat, usf, swow, cnnb, cnrvec, w2v_ha_bow, eat_ha_bow, usf_ha_bow, swow_ha_bow, cnnb_ha_bow, cnrvec_ha_bow, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, swow_ha_intrainter, cnnb_ha_intrainter, cnrvec_ha_intrainter, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, swow_ha_setuppunch, cnnb_ha_setuppunch, cnrvec_ha_setuppunch, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, swow_ha_funnyonly, cnnb_ha_funnyonly, cnrvec_ha_funnyonly = pickle.load(f)
#         print("\t".join(str(f.shape[1]) for f in (perplex, w2v, eat, usf, swow, cnnb, cnrvec, w2v_ha_bow, eat_ha_bow, usf_ha_bow, swow_ha_bow, cnnb_ha_bow, cnrvec_ha_bow, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, swow_ha_intrainter, cnnb_ha_intrainter, cnrvec_ha_intrainter, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, swow_ha_setuppunch, cnnb_ha_setuppunch, cnrvec_ha_setuppunch, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, swow_ha_funnyonly, cnnb_ha_funnyonly, cnrvec_ha_funnyonly)))
          
        #(test label, columns to keep)
    #     test_setups = [("all (no hae)", list(range(0,28))),
    #                    ("all (only hae)", [0]+list(range(28,55))),
    #                    ("all (full + hae)", list(range(55))),
    #                    ("perplex only", [0]),
    #                    ("word2vec only (no hae)", list(range(1,4))),
    #                    ("word2vec only (only hae)", list(range(28,31))),
    #                    ("word2vec only (full + hae)", list(range(1,4)) + list(range(28,31))),
    #                    ("all word assoc (no hae)", list(range(4,28))),
    #                    ("all word assoc (only hae)", list(range(31,55))),
    #                    ("all word assoc (full + hae)", list(range(4,28))+list(range(31,55))),
    #                    ("eat only (no hae)", list(range(4,16))),
    #                    ("eat only (only hae)", list(range(31,43))),
    #                    ("eat only (full + hae)", list(range(4,16)) + list(range(31,43))),
    #                    ("eat forward only (no hae)", list(range(4,7))),
    #                    ("eat forward only (only hae)", list(range(31,34))),
    #                    ("eat forward only (full + hae)", list(range(4,7))+list(range(31,34))),
    #                    ("eat backward only (no hae)", list(range(7,10))),
    #                    ("eat backward only (only hae)", list(range(34,37))),
    #                    ("eat backward only (full + hae)", list(range(7,10))+list(range(34,37))),
    #                    ("eat diff only (no hae)", list(range(10,16))),
    #                    ("eat diff only (only hae)", list(range(37,43))),
    #                    ("eat diff only (full + hae)", list(range(10,16))+list(range(37,43))),
    #                    ("eat micro diff only (no hae)", list(range(10,13))),
    #                    ("eat micro diff only (only hae)", list(range(37,40))),
    #                    ("eat micro diff only (full + hae)", list(range(10,13))+list(range(37,40))),
    #                    ("eat macro diff only (no hae)", list(range(13,16))),
    #                    ("eat macro diff only (only hae)", list(range(40,43))),
    #                    ("eat macro diff only (full + hae)", list(range(13,16))+list(range(40,43))),
    #                    ("usf only (no hae)", list(range(16,28))),
    #                    ("usf only (only hae)", list(range(43,55))),
    #                    ("usf only (full + hae)", list(range(16,28))+list(range(43,55))),
    #                    ("usf forward only (no hae)", list(range(16,19))),
    #                    ("usf forward only (only hae)", list(range(43,46))),
    #                    ("usf forward only (full + hae)", list(range(16,19))+list(range(43,46))),
    #                    ("usf backward only (no hae)", list(range(19,22))),
    #                    ("usf backward only (only hae)", list(range(46,49))),
    #                    ("usf backward only (full + hae)", list(range(19,22))+list(range(46,49))),
    #                    ("usf diff only (no hae)", list(range(22,28))),
    #                    ("usf diff only (only hae)", list(range(49,55))),
    #                    ("usf diff only (full + hae)", list(range(22,28))+list(range(49,55))),
    #                    ("usf micro diff only (no hae)", list(range(22,25))),
    #                    ("usf micro diff only (only hae)", list(range(49,52))),
    #                    ("usf micro diff only (full + hae)", list(range(22,25))+list(range(49,52))),
    #                    ("usf macro diff only (no hae)", list(range(25,28))),
    #                    ("usf macro diff only (only hae)", list(range(52,55))),
    #                    ("usf macro diff only (full + hae)", list(range(25,28))+list(range(52,55))),
    #                     
    #                     ("all + swow (no hae)", list(range(0,28))+list(range(55,67))),
    #                     ("all + swow (only hae)", [0]+list(range(67,79))),
    #                     ("all + swow (full + hae)", list(range(train_feat.shape[1]))),
    #                     ("swow only (no hae)", list(range(55,67))),
    #                     ("swow only (only hae)", list(range(67,79))),
    #                     ("swow only (full + hae)", list(range(55,79))),
    #                     ("swow forward only (no hae)", list(range(55,58))),
    #                     ("swow forward only (only hae)", list(range(67,70))),
    #                     ("swow forward only (full + hae)", list(range(55,58))+list(range(67,70))),
    #                     ("swow backward only (no hae)", list(range(58,61))),
    #                     ("swow backward only (only hae)", list(range(70,73))),
    #                     ("swow backward only (full + hae)", list(range(58,61))+list(range(70,73))),
    #                     ("swow diff only (no hae)", list(range(61,67))),
    #                     ("swow diff only (only hae)", list(range(73,79))),
    #                     ("swow diff only (full + hae)", list(range(61,67))+list(range(73,79))),
    #                     ("swow micro diff only (no hae)", list(range(61,64))),
    #                     ("swow micro diff only (only hae)", list(range(73,76))),
    #                     ("swow micro diff only (full + hae)", list(range(61,64))+list(range(73,76))),
    #                     ("swow macro diff only (no hae)", list(range(64,67))),
    #                     ("swow macro diff only (only hae)", list(range(76,79))),
    #                     ("swow macro diff only (full + hae)", list(range(64,67))+list(range(76,79)))
    # #                    ("allit only", [28,29,30,31]),
    # #                    ("allit only", [28,29,30,31]),
    # #                    ("interpresonal only", [32,33,34,35])
    #                    ]
         
    #     test_setups = [("all (no hae)", list(range(0,28))),
    #                    ("all (only hae)", [0]+list(range(28,82))),
    #                    ("all (full + hae)", list(range(82))),
    #                    ("perplex only", [0]),
    #                    ("word2vec only (no hae)", list(range(1,4))),
    #                    ("word2vec only (only hae)", list(range(28,31)) + list(range(55,58))),
    #                    ("word2vec only (full + hae)", list(range(1,4)) + list(range(28,31)) + list(range(55,58))),
    #                    ("all word assoc (no hae)", list(range(4,28))),
    #                    ("all word assoc (only hae)", list(range(31,55)) + list(range(58,82))),
    #                    ("all word assoc (full + hae)", list(range(4,28))+list(range(31,55)) + list(range(58,82))),
    #                    ("eat only (no hae)", list(range(4,16))),
    #                    ("eat only (only hae)", list(range(31,43)) + list(range(58,70))),
    #                    ("eat only (full + hae)", list(range(4,16)) + list(range(31,43)) + list(range(58,70))),
    #                    ("eat forward only (no hae)", list(range(4,7))),
    #                    ("eat forward only (only hae)", list(range(31,34)) + list(range(58,61))),
    #                    ("eat forward only (full + hae)", list(range(4,7))+list(range(31,34)) + list(range(58,61))),
    #                    ("eat backward only (no hae)", list(range(7,10))),
    #                    ("eat backward only (only hae)", list(range(34,37)) + list(range(61,64))),
    #                    ("eat backward only (full + hae)", list(range(7,10))+list(range(34,37)) + list(range(61,64))),
    #                    ("eat diff only (no hae)", list(range(10,16))),
    #                    ("eat diff only (only hae)", list(range(37,43)) + list(range(64,70))),
    #                    ("eat diff only (full + hae)", list(range(10,16))+list(range(37,43)) + list(range(64,70))),
    #                    ("eat micro diff only (no hae)", list(range(10,13))),
    #                    ("eat micro diff only (only hae)", list(range(37,40)) + list(range(64,67))),
    #                    ("eat micro diff only (full + hae)", list(range(10,13))+list(range(37,40)) + list(range(64,67))),
    #                    ("eat macro diff only (no hae)", list(range(13,16))),
    #                    ("eat macro diff only (only hae)", list(range(40,43)) + list(range(67,70))),
    #                    ("eat macro diff only (full + hae)", list(range(13,16))+list(range(40,43)) + list(range(67,70))),
    #                    ("usf only (no hae)", list(range(16,28))),
    #                    ("usf only (only hae)", list(range(43,55)) + list(range(70,82))),
    #                    ("usf only (full + hae)", list(range(16,28))+list(range(43,55)) + list(range(70,82))),
    #                    ("usf forward only (no hae)", list(range(16,19))),
    #                    ("usf forward only (only hae)", list(range(43,46)) + list(range(70,73))),
    #                    ("usf forward only (full + hae)", list(range(16,19))+list(range(43,46)) + list(range(70,73))),
    #                    ("usf backward only (no hae)", list(range(19,22))),
    #                    ("usf backward only (only hae)", list(range(46,49)) + list(range(73,76))),
    #                    ("usf backward only (full + hae)", list(range(19,22))+list(range(46,49)) + list(range(73,76))),
    #                    ("usf diff only (no hae)", list(range(22,28))),
    #                    ("usf diff only (only hae)", list(range(49,55)) + list(range(76,82))),
    #                    ("usf diff only (full + hae)", list(range(22,28))+list(range(49,55)) + list(range(76,82))),
    #                    ("usf micro diff only (no hae)", list(range(22,25))),
    #                    ("usf micro diff only (only hae)", list(range(49,52)) + list(range(76,79))),
    #                    ("usf micro diff only (full + hae)", list(range(22,25))+list(range(49,52)) + list(range(76,79))),
    #                    ("usf macro diff only (no hae)", list(range(25,28))),
    #                    ("usf macro diff only (only hae)", list(range(52,55)) + list(range(79,82))),
    #                    ("usf macro diff only (full + hae)", list(range(25,28))+list(range(52,55)) + list(range(79,82))),
    #                      
    #                    ("all + swow (no hae)", list(range(0,28))+list(range(55,67))),
    #                    ("all + swow (only hae)", [0]+list(range(67,79))+list(range(94,118))),
    #                    ("all + swow (full + hae)", list(range(train_feat.shape[1]))),
    #                    ("swow only (no hae)", list(range(82, 94))),
    #                    ("swow only (only hae)", list(range(94, 118))),
    #                    ("swow only (full + hae)", list(range(82,118))),
    #                    ("swow forward only (no hae)", list(range(82,85))),
    #                    ("swow forward only (only hae)", list(range(94,97))+list(range(106,109))),
    #                    ("swow forward only (full + hae)", list(range(82,85))+list(range(94,97))+list(range(106,109))),
    #                    ("swow backward only (no hae)", list(range(85,88))),
    #                    ("swow backward only (only hae)", list(range(97,100))+list(range(109,112))),
    #                    ("swow backward only (full + hae)", list(range(85,88))+list(range(97,100))+list(range(109,112))),
    #                    ("swow diff only (no hae)", list(range(88,94))),
    #                    ("swow diff only (only hae)", list(range(100,106))+list(range(112,118))),
    #                    ("swow diff only (full + hae)", list(range(88,94))+list(range(100,106))+list(range(112,118))),
    #                    ("swow micro diff only (no hae)", list(range(88,91))),
    #                    ("swow micro diff only (only hae)", list(range(100,103))+list(range(112,115))),
    #                    ("swow micro diff only (full + hae)", list(range(88,91))+list(range(100,103))+list(range(112,115))),
    #                    ("swow macro diff only (no hae)", list(range(91,94))),
    #                    ("swow macro diff only (only hae)", list(range(103,106))+list(range(115,118))),
    #                    ("swow macro diff only (full + hae)", list(range(91,94))+list(range(103,106))+list(range(115,118)))
    # #                    ("allit only", [28,29,30,31]),
    # #                    ("interpresonal only", [32,33,34,35])
    #                    ]
    
        test_setups = [("all (no hae)", (perplex, w2v, eat, usf)),
#                        ("all (only bow hae)", (perplex, w2v_ha_bow, eat_ha_bow, usf_ha_bow)),
#                        ("all (only intra/inter hae)", (perplex, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter)),
#                        ("all (only setup/punch hae)", (perplex, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch)),
#                        ("all (only funny only hae)", (perplex, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly)),
#                        ("all (full + bow hae)", (perplex, w2v, eat, usf, w2v_ha_bow, eat_ha_bow, usf_ha_bow)),
#                        ("all (full + intra/inter hae)", (perplex, w2v, eat, usf, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter)),
#                        ("all (full + setup/punch hae)", (perplex, w2v, eat, usf, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch)),
#                        ("all (full + funny only hae)", (perplex, w2v, eat, usf, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly)),
                       
#                        ("perplex only", (perplex,)),
#                        
#                        ("word2vec only (no hae)", (w2v,)),
#                        ("word2vec only (only bow hae)", (w2v_ha_bow,)),
#                        ("word2vec only (only intra/inter hae)", (w2v_ha_intrainter,)),
#                        ("word2vec only (only setup/punch hae)", (w2v_ha_setuppunch,)),
#                        ("word2vec only (only funny only hae)", (w2v_ha_funnyonly,)),
#                        ("word2vec only (full + bow hae)", (w2v, w2v_ha_bow)),
#                        ("word2vec only (full + intra/inter hae)", (w2v, w2v_ha_intrainter)),
#                        ("word2vec only (full + setup/punch hae)", (w2v, w2v_ha_setuppunch)),
#                        ("word2vec only (full + funny only hae)", (w2v, w2v_ha_funnyonly)),
#                        
#                        ("all word assoc (no hae)", (eat,usf)),
#                        ("all word assoc (only bow hae)", (eat_ha_bow, usf_ha_bow)),
#                        ("all word assoc (only intra/inter hae)", (eat_ha_intrainter, usf_ha_intrainter)),
#                        ("all word assoc (only setup/punch hae)", (eat_ha_setuppunch, usf_ha_setuppunch)),
#                        ("all word assoc (only funny only hae)", (eat_ha_funnyonly, usf_ha_funnyonly)),
#                        ("all word assoc (full + bow hae)", (eat, usf, eat_ha_bow, usf_ha_bow)),
#                        ("all word assoc (full + intra/inter hae)", (eat, usf, eat_ha_intrainter, usf_ha_intrainter)),
#                        ("all word assoc (full + setup/punch hae)", (eat, usf, eat_ha_setuppunch, usf_ha_setuppunch)),
#                        ("all word assoc (full + funny only hae)", (eat, usf, eat_ha_funnyonly, usf_ha_funnyonly)),
#                        
#                        ("eat only (no hae)", (eat,)),
#                        ("eat only (only bow hae)", (eat_ha_bow,)),
#                        ("eat only (only intra/inter hae)", (eat_ha_intrainter,)),
#                        ("eat only (only setup/punch hae)", (eat_ha_setuppunch,)),
#                        ("eat only (only funny only hae)", (eat_ha_funnyonly,)),
#                        ("eat only (full + bow hae)", (eat,  eat_ha_bow)),
#                        ("eat only (full + intra/inter hae)", (eat,  eat_ha_intrainter)),
#                        ("eat only (full + setup/punch hae)", (eat,  eat_ha_setuppunch)),
#                        ("eat only (full + funny only hae)", (eat,  eat_ha_funnyonly)),
#                        ("eat forward only (no hae)", (eat[:,0:3],)),
#                        ("eat forward only (only bow hae)", (eat_ha_bow[:,0:3],)),
#                        ("eat forward only (only intra/inter hae)", (eat_ha_intrainter[:,0:6],)),
#                        ("eat forward only (only setup/punch hae)", (eat_ha_setuppunch[:,0:6],)),
#                        ("eat forward only (only funny only hae)", (eat_ha_funnyonly[:,0:6],)),
#                        ("eat forward only (full + bow hae)", (eat[:,0:3], eat_ha_bow[:,0:3])),
#                        ("eat forward only (full + intra/inter hae)", (eat[:,0:3], eat_ha_intrainter[:,0:6])),
#                        ("eat forward only (full + setup/punch hae)", (eat[:,0:3], eat_ha_setuppunch[:,0:6])),
#                        ("eat forward only (full + funny only hae)", (eat[:,0:3], eat_ha_funnyonly[:,0:6])),
#                        ("eat backward only (no hae)", (eat[:,3:6],)),
#                        ("eat backward only (only bow hae)", (eat_ha_bow[:,3:6],)),
#                        ("eat backward only (only intra/inter hae)", (eat_ha_intrainter[:,6:12],)),
#                        ("eat backward only (only setup/punch hae)", (eat_ha_setuppunch[:,6:12],)),
#                        ("eat backward only (only funny only hae)", (eat_ha_funnyonly[:,6:12],)),
#                        ("eat backward only (full + bow hae)", (eat[:,3:6],eat_ha_bow[:,3:6])),
#                        ("eat backward only (full + intra/inter hae)", (eat[:,3:6],eat_ha_intrainter[:,6:12])),
#                        ("eat backward only (full + setup/punch hae)", (eat[:,3:6],eat_ha_setuppunch[:,6:12])),
#                        ("eat backward only (full + funny only hae)", (eat[:,3:6],eat_ha_funnyonly[:,6:12])),
#                        ("eat diff only (no hae)", (eat[:,6:12],)),
#                        ("eat diff only (only bow hae)", (eat_ha_bow[:,6:12],)),
#                        ("eat diff only (only intra/inter hae)", (eat_ha_intrainter[:,12:24],)),
#                        ("eat diff only (only setup/punch hae)", (eat_ha_setuppunch[:,12:24],)),
#                        ("eat diff only (only funny only hae)", (eat_ha_funnyonly[:,12:24],)),
#                        ("eat diff only (full + bow hae)", (eat[:,6:12], eat_ha_bow[:,6:12])),
#                        ("eat diff only (full + intra/inter hae)", (eat[:,6:12], eat_ha_intrainter[:,12:24])),
#                        ("eat diff only (full + setup/punch hae)", (eat[:,6:12], eat_ha_setuppunch[:,12:24])),
#                        ("eat diff only (full + funny only hae)", (eat[:,6:12], eat_ha_funnyonly[:,12:24])),
#                        ("eat micro diff only (no hae)", (eat[:,6:9],)),
#                        ("eat micro diff only (only bow hae)", (eat_ha_bow[:,6:9],)),
#                        ("eat micro diff only (only intra/inter hae)", (eat_ha_intrainter[:,12:18],)),
#                        ("eat micro diff only (only setup/punch hae)", (eat_ha_setuppunch[:,12:18],)),
#                        ("eat micro diff only (only funny only hae)", (eat_ha_funnyonly[:,12:18],)),
#                        ("eat micro diff only (full + bow hae)", (eat[:,6:9], eat_ha_bow[:,6:9])),
#                        ("eat micro diff only (full + intra/inter hae)", (eat[:,6:9], eat_ha_intrainter[:,12:18])),
#                        ("eat micro diff only (full + setup/punch hae)", (eat[:,6:9], eat_ha_setuppunch[:,12:18])),
#                        ("eat micro diff only (full + funny only hae)", (eat[:,6:9], eat_ha_funnyonly[:,12:18])),
#                        ("eat macro diff only (no hae)", (eat[:,9:12],)),
#                        ("eat macro diff only (only bow hae)", (eat_ha_bow[:,9:12],)),
#                        ("eat macro diff only (only intra/inter hae)", (eat_ha_intrainter[:,18:24],)),
#                        ("eat macro diff only (only setup/punch hae)", (eat_ha_setuppunch[:,18:24],)),
#                        ("eat macro diff only (only funny only hae)", (eat_ha_funnyonly[:,18:24],)),
#                        ("eat macro diff only (full + bow hae)", (eat[:,9:12], eat_ha_bow[:,9:12])),
#                        ("eat macro diff only (full + intra/inter hae)", (eat[:,9:12], eat_ha_intrainter[:,18:24])),
#                        ("eat macro diff only (full + setup/punch hae)", (eat[:,9:12], eat_ha_setuppunch[:,18:24])),
#                        ("eat macro diff only (full + funny only hae)", (eat[:,9:12], eat_ha_funnyonly[:,18:24])),
#                        
#                        ("usf only (no hae)", (usf,)),
#                        ("usf only (only bow hae)", (usf_ha_bow,)),
#                        ("usf only (only intra/inter hae)", (usf_ha_intrainter,)),
#                        ("usf only (only setup/punch hae)", (usf_ha_setuppunch,)),
#                        ("usf only (only funny only hae)", (usf_ha_funnyonly,)),
#                        ("usf only (full + bow hae)", (usf,  usf_ha_bow)),
#                        ("usf only (full + intra/inter hae)", (usf,  usf_ha_intrainter)),
#                        ("usf only (full + setup/punch hae)", (usf,  usf_ha_setuppunch)),
#                        ("usf only (full + funny only hae)", (usf,  usf_ha_funnyonly)),
#                        ("usf forward only (no hae)", (usf[:,0:3],)),
#                        ("usf forward only (only bow hae)", (usf_ha_bow[:,0:3],)),
#                        ("usf forward only (only intra/inter hae)", (usf_ha_intrainter[:,0:6],)),
#                        ("usf forward only (only setup/punch hae)", (usf_ha_setuppunch[:,0:6],)),
#                        ("usf forward only (only funny only hae)", (usf_ha_funnyonly[:,0:6],)),
#                        ("usf forward only (full + bow hae)", (usf[:,0:3], usf_ha_bow[:,0:3])),
#                        ("usf forward only (full + intra/inter hae)", (usf[:,0:3], usf_ha_intrainter[:,0:6])),
#                        ("usf forward only (full + setup/punch hae)", (usf[:,0:3], usf_ha_setuppunch[:,0:6])),
#                        ("usf forward only (full + funny only hae)", (usf[:,0:3], usf_ha_funnyonly[:,0:6])),
#                        ("usf backward only (no hae)", (usf[:,3:6],)),
#                        ("usf backward only (only bow hae)", (usf_ha_bow[:,3:6],)),
#                        ("usf backward only (only intra/inter hae)", (usf_ha_intrainter[:,6:12],)),
#                        ("usf backward only (only setup/punch hae)", (usf_ha_setuppunch[:,6:12],)),
#                        ("usf backward only (only funny only hae)", (usf_ha_funnyonly[:,6:12],)),
#                        ("usf backward only (full + bow hae)", (usf[:,3:6],usf_ha_bow[:,3:6])),
#                        ("usf backward only (full + intra/inter hae)", (usf[:,3:6],usf_ha_intrainter[:,6:12])),
#                        ("usf backward only (full + setup/punch hae)", (usf[:,3:6],usf_ha_setuppunch[:,6:12])),
#                        ("usf backward only (full + funny only hae)", (usf[:,3:6],usf_ha_funnyonly[:,6:12])),
#                        ("usf diff only (no hae)", (usf[:,6:12],)),
#                        ("usf diff only (only bow hae)", (usf_ha_bow[:,6:12],)),
#                        ("usf diff only (only intra/inter hae)", (usf_ha_intrainter[:,12:24],)),
#                        ("usf diff only (only setup/punch hae)", (usf_ha_setuppunch[:,12:24],)),
#                        ("usf diff only (only funny only hae)", (usf_ha_funnyonly[:,12:24],)),
#                        ("usf diff only (full + bow hae)", (usf[:,6:12], usf_ha_bow[:,6:12])),
#                        ("usf diff only (full + intra/inter hae)", (usf[:,6:12], usf_ha_intrainter[:,12:24])),
#                        ("usf diff only (full + setup/punch hae)", (usf[:,6:12], usf_ha_setuppunch[:,12:24])),
#                        ("usf diff only (full + funny only hae)", (usf[:,6:12], usf_ha_funnyonly[:,12:24])),
#                        ("usf micro diff only (no hae)", (usf[:,6:9],)),
#                        ("usf micro diff only (only bow hae)", (usf_ha_bow[:,6:9],)),
#                        ("usf micro diff only (only intra/inter hae)", (usf_ha_intrainter[:,12:18],)),
#                        ("usf micro diff only (only setup/punch hae)", (usf_ha_setuppunch[:,12:18],)),
#                        ("usf micro diff only (only funny only hae)", (usf_ha_funnyonly[:,12:18],)),
#                        ("usf micro diff only (full + bow hae)", (usf[:,6:9], usf_ha_bow[:,6:9])),
#                        ("usf micro diff only (full + intra/inter hae)", (usf[:,6:9], usf_ha_intrainter[:,12:18])),
#                        ("usf micro diff only (full + setup/punch hae)", (usf[:,6:9], usf_ha_setuppunch[:,12:18])),
#                        ("usf micro diff only (full + funny only hae)", (usf[:,6:9], usf_ha_funnyonly[:,12:18])),
#                        ("usf macro diff only (no hae)", (usf[:,9:12],)),
#                        ("usf macro diff only (only bow hae)", (usf_ha_bow[:,9:12],)),
#                        ("usf macro diff only (only intra/inter hae)", (usf_ha_intrainter[:,18:24],)),
#                        ("usf macro diff only (only setup/punch hae)", (usf_ha_setuppunch[:,18:24],)),
#                        ("usf macro diff only (only funny only hae)", (usf_ha_funnyonly[:,18:24],)),
#                        ("usf macro diff only (full + bow hae)", (usf[:,9:12], usf_ha_bow[:,9:12])),
#                        ("usf macro diff only (full + intra/inter hae)", (usf[:,9:12], usf_ha_intrainter[:,18:24])),
#                        ("usf macro diff only (full + setup/punch hae)", (usf[:,9:12], usf_ha_setuppunch[:,18:24])),
#                        ("usf macro diff only (full + funny only hae)", (usf[:,9:12], usf_ha_funnyonly[:,18:24])),
#                        
#                        ("all + swow (no hae)", (perplex, w2v, eat, usf,swow)),
#                        ("all + swow (only bow hae)", (perplex, w2v_ha_bow, eat_ha_bow, usf_ha_bow, swow_ha_bow)),
#                        ("all + swow (only intra/inter hae)", (perplex, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, swow_ha_intrainter)),
#                        ("all + swow (only setup/punch hae)", (perplex, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, swow_ha_setuppunch)),
#                        ("all + swow (only funny only hae)", (perplex, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, swow_ha_funnyonly)),
#                        ("all + swow (full + bow hae)", (perplex, w2v, eat, usf, swow, w2v_ha_bow, eat_ha_bow, usf_ha_bow, swow_ha_bow)),
#                        ("all + swow (full + intra/inter hae)", (perplex, w2v, eat, usf, swow, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, swow_ha_intrainter)),
#                        ("all + swow (full + setup/punch hae)", (perplex, w2v, eat, usf, swow, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, swow_ha_setuppunch)),
#                        ("all + swow (full + funny only hae)", (perplex, w2v, eat, usf, swow, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, swow_ha_funnyonly)),
#                        
#                        ("swow only (no hae)", (swow,)),
#                        ("swow only (only bow hae)", (swow_ha_bow,)),
#                        ("swow only (only intra/inter hae)", (swow_ha_intrainter,)),
#                        ("swow only (only setup/punch hae)", (swow_ha_setuppunch,)),
#                        ("swow only (only funny only hae)", (swow_ha_funnyonly,)),
#                        ("swow only (full + bow hae)", (swow,  swow_ha_bow)),
#                        ("swow only (full + intra/inter hae)", (swow,  swow_ha_intrainter)),
#                        ("swow only (full + setup/punch hae)", (swow,  swow_ha_setuppunch)),
#                        ("swow only (full + funny only hae)", (swow,  swow_ha_funnyonly)),
#                        ("swow forward only (no hae)", (swow[:,0:3],)),
#                        ("swow forward only (only bow hae)", (swow_ha_bow[:,0:3],)),
#                        ("swow forward only (only intra/inter hae)", (swow_ha_intrainter[:,0:6],)),
#                        ("swow forward only (only setup/punch hae)", (swow_ha_setuppunch[:,0:6],)),
#                        ("swow forward only (only funny only hae)", (swow_ha_funnyonly[:,0:6],)),
#                        ("swow forward only (full + bow hae)", (swow[:,0:3], swow_ha_bow[:,0:3])),
#                        ("swow forward only (full + intra/inter hae)", (swow[:,0:3], swow_ha_intrainter[:,0:6])),
#                        ("swow forward only (full + setup/punch hae)", (swow[:,0:3], swow_ha_setuppunch[:,0:6])),
#                        ("swow forward only (full + funny only hae)", (swow[:,0:3], swow_ha_funnyonly[:,0:6])),
#                        ("swow backward only (no hae)", (swow[:,3:6],)),
#                        ("swow backward only (only bow hae)", (swow_ha_bow[:,3:6],)),
#                        ("swow backward only (only intra/inter hae)", (swow_ha_intrainter[:,6:12],)),
#                        ("swow backward only (only setup/punch hae)", (swow_ha_setuppunch[:,6:12],)),
#                        ("swow backward only (only funny only hae)", (swow_ha_funnyonly[:,6:12],)),
#                        ("swow backward only (full + bow hae)", (swow[:,3:6],swow_ha_bow[:,3:6])),
#                        ("swow backward only (full + intra/inter hae)", (swow[:,3:6],swow_ha_intrainter[:,6:12])),
#                        ("swow backward only (full + setup/punch hae)", (swow[:,3:6],swow_ha_setuppunch[:,6:12])),
#                        ("swow backward only (full + funny only hae)", (swow[:,3:6],swow_ha_funnyonly[:,6:12])),
#                        ("swow diff only (no hae)", (swow[:,6:12],)),
#                        ("swow diff only (only bow hae)", (swow_ha_bow[:,6:12],)),
#                        ("swow diff only (only intra/inter hae)", (swow_ha_intrainter[:,12:24],)),
#                        ("swow diff only (only setup/punch hae)", (swow_ha_setuppunch[:,12:24],)),
#                        ("swow diff only (only funny only hae)", (swow_ha_funnyonly[:,12:24],)),
#                        ("swow diff only (full + bow hae)", (swow[:,6:12], swow_ha_bow[:,6:12])),
#                        ("swow diff only (full + intra/inter hae)", (swow[:,6:12], swow_ha_intrainter[:,12:24])),
#                        ("swow diff only (full + setup/punch hae)", (swow[:,6:12], swow_ha_setuppunch[:,12:24])),
#                        ("swow diff only (full + funny only hae)", (swow[:,6:12], swow_ha_funnyonly[:,12:24])),
#                        ("swow micro diff only (no hae)", (swow[:,6:9],)),
#                        ("swow micro diff only (only bow hae)", (swow_ha_bow[:,6:9],)),
#                        ("swow micro diff only (only intra/inter hae)", (swow_ha_intrainter[:,12:18],)),
#                        ("swow micro diff only (only setup/punch hae)", (swow_ha_setuppunch[:,12:18],)),
#                        ("swow micro diff only (only funny only hae)", (swow_ha_funnyonly[:,12:18],)),
#                        ("swow micro diff only (full + bow hae)", (swow[:,6:9], swow_ha_bow[:,6:9])),
#                        ("swow micro diff only (full + intra/inter hae)", (swow[:,6:9], swow_ha_intrainter[:,12:18])),
#                        ("swow micro diff only (full + setup/punch hae)", (swow[:,6:9], swow_ha_setuppunch[:,12:18])),
#                        ("swow micro diff only (full + funny only hae)", (swow[:,6:9], swow_ha_funnyonly[:,12:18])),
#                        ("swow macro diff only (no hae)", (swow[:,9:12],)),
#                        ("swow macro diff only (only bow hae)", (swow_ha_bow[:,9:12],)),
#                        ("swow macro diff only (only intra/inter hae)", (swow_ha_intrainter[:,18:24],)),
#                        ("swow macro diff only (only setup/punch hae)", (swow_ha_setuppunch[:,18:24],)),
#                        ("swow macro diff only (only funny only hae)", (swow_ha_funnyonly[:,18:24],)),
#                        ("swow macro diff only (full + bow hae)", (swow[:,9:12], swow_ha_bow[:,9:12])),
#                        ("swow macro diff only (full + intra/inter hae)", (swow[:,9:12], swow_ha_intrainter[:,18:24])),
#                        ("swow macro diff only (full + setup/punch hae)", (swow[:,9:12], swow_ha_setuppunch[:,18:24])),
#                        ("swow macro diff only (full + funny only hae)", (swow[:,9:12], swow_ha_funnyonly[:,18:24])),
#             
#                        ("all + swow + cnnb (no hae)", (perplex, w2v, eat, usf, swow, cnnb)),
#                        ("all + swow + cnnb (only bow hae)", (perplex, w2v_ha_bow, eat_ha_bow, usf_ha_bow, swow_ha_bow, cnnb_ha_bow)),
#                        ("all + swow + cnnb (only intra/inter hae)", (perplex, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, swow_ha_intrainter, cnnb_ha_intrainter)),
#                        ("all + swow + cnnb (only setup/punch hae)", (perplex, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, swow_ha_setuppunch, cnnb_ha_setuppunch)),
#                        ("all + swow + cnnb (only funny only hae)", (perplex, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, swow_ha_funnyonly, cnnb_ha_funnyonly)),
#                        ("all + swow + cnnb (full + bow hae)", (perplex, w2v, eat, usf, swow, cnnb, w2v_ha_bow, eat_ha_bow, usf_ha_bow, swow_ha_bow, cnnb_ha_bow)),
#                        ("all + swow + cnnb (full + intra/inter hae)", (perplex, w2v, eat, usf, swow, cnnb, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, swow_ha_intrainter, cnnb_ha_intrainter)),
#                        ("all + swow + cnnb (full + setup/punch hae)", (perplex, w2v, eat, usf, swow, cnnb, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, swow_ha_setuppunch, cnnb_ha_setuppunch)),
#                        ("all + swow + cnnb (full + funny only hae)", (perplex, w2v, eat, usf, swow, cnnb, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, swow_ha_funnyonly, cnnb_ha_funnyonly)),

#                         ("all + cnnb (no hae)", (perplex, w2v, eat, usf, cnnb)),
#                         ("all + cnnb (only bow hae)", (perplex, w2v_ha_bow, eat_ha_bow, usf_ha_bow, cnnb_ha_bow)),
#                         ("all + cnnb (only intra/inter hae)", (perplex, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, cnnb_ha_intrainter)),
#                         ("all + cnnb (only setup/punch hae)", (perplex, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, cnnb_ha_setuppunch)),
#                         ("all + cnnb (only funny only hae)", (perplex, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, cnnb_ha_funnyonly)),
#                         ("all + cnnb (full + bow hae)", (perplex, w2v, eat, usf, cnnb, w2v_ha_bow, eat_ha_bow, usf_ha_bow, cnnb_ha_bow)),
#                         ("all + cnnb (full + intra/inter hae)", (perplex, w2v, eat, usf, cnnb, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, cnnb_ha_intrainter)),
#                         ("all + cnnb (full + setup/punch hae)", (perplex, w2v, eat, usf, cnnb, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, cnnb_ha_setuppunch)),
#                         ("all + cnnb (full + funny only hae)", (perplex, w2v, eat, usf, cnnb, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, cnnb_ha_funnyonly)),
#                         
#                         ("cnnb only (no hae)", (cnnb,)),
#                         ("cnnb only (only bow hae)", (cnnb_ha_bow,)),
#                         ("cnnb only (only intra/inter hae)", (cnnb_ha_intrainter,)),
#                         ("cnnb only (only setup/punch hae)", (cnnb_ha_setuppunch,)),
#                         ("cnnb only (only funny only hae)", (cnnb_ha_funnyonly,)),
#                         ("cnnb only (full + bow hae)", (cnnb, cnnb_ha_bow)),
#                         ("cnnb only (full + intra/inter hae)", (cnnb, cnnb_ha_intrainter)),
#                         ("cnnb only (full + setup/punch hae)", (cnnb, cnnb_ha_setuppunch)),
#                         ("cnnb only (full + funny only hae)", (cnnb, cnnb_ha_funnyonly)),
#                        
#                         ("all + swow + cnrvec_f_only (no hae)", (perplex, w2v, eat, usf, swow, cnrvec_f_only)),
#                         ("all + swow + cnrvec_comb (no hae)", (perplex, w2v, eat, usf, swow, cnrvec_comb)),
#                         ("all + swow + cnrvec_fb (no hae)", (perplex, w2v, eat, usf, swow, cnrvec_fb)),
#                         ("all + swow + cnrvec_f_only (only bow hae)", (perplex, w2v_ha_bow, eat_ha_bow, usf_ha_bow, swow_ha_bow, cnrvec_ha_bow_f_only)),
#                         ("all + swow + cnrvec_comb (only bow hae)", (perplex, w2v_ha_bow, eat_ha_bow, usf_ha_bow, swow_ha_bow, cnrvec_ha_bow_comb)),
#                         ("all + swow + cnrvec_fb (only bow hae)", (perplex, w2v_ha_bow, eat_ha_bow, usf_ha_bow, swow_ha_bow, cnrvec_ha_bow_fb)),
#                         ("all + swow + cnrvec_f_only (only intra/inter hae)", (perplex, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, swow_ha_intrainter, cnrvec_ha_intrainter_f_only)),
#                         ("all + swow + cnrvec_comb (only intra/inter hae)", (perplex, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, swow_ha_intrainter, cnrvec_ha_intrainter_comb)),
#                         ("all + swow + cnrvec_fb (only intra/inter hae)", (perplex, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, swow_ha_intrainter, cnrvec_ha_intrainter_fb)),
#                         ("all + swow + cnrvec_f_only (only setup/punch hae)", (perplex, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, swow_ha_setuppunch, cnrvec_ha_setuppunch_f_only)),
#                         ("all + swow + cnrvec_comb (only setup/punch hae)", (perplex, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, swow_ha_setuppunch, cnrvec_ha_setuppunch_comb)),
#                         ("all + swow + cnrvec_fb (only setup/punch hae)", (perplex, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, swow_ha_setuppunch, cnrvec_ha_setuppunch_fb)),
#                         ("all + swow + cnrvec_f_only (only funny only hae)", (perplex, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, swow_ha_funnyonly, cnrvec_ha_funnyonly_f_only)),
#                         ("all + swow + cnrvec_comb (only funny only hae)", (perplex, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, swow_ha_funnyonly, cnrvec_ha_funnyonly_comb)),
#                         ("all + swow + cnrvec_fb (only funny only hae)", (perplex, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, swow_ha_funnyonly, cnrvec_ha_funnyonly_fb)),
#                         ("all + swow + cnrvec_f_only (full + bow hae)", (perplex, w2v, eat, usf, swow, cnrvec_f_only, w2v_ha_bow, eat_ha_bow, usf_ha_bow, swow_ha_bow, cnrvec_ha_bow_f_only)),
#                         ("all + swow + cnrvec_comb (full + bow hae)", (perplex, w2v, eat, usf, swow, cnrvec_comb, w2v_ha_bow, eat_ha_bow, usf_ha_bow, swow_ha_bow, cnrvec_ha_bow_comb)),
#                         ("all + swow + cnrvec_fb (full + bow hae)", (perplex, w2v, eat, usf, swow, cnrvec_fb, w2v_ha_bow, eat_ha_bow, usf_ha_bow, swow_ha_bow, cnrvec_ha_bow_fb)),
#                         ("all + swow + cnrvec_f_only (full + intra/inter hae)", (perplex, w2v, eat, usf, swow, cnrvec_f_only, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, swow_ha_intrainter, cnrvec_ha_intrainter_f_only)),
#                         ("all + swow + cnrvec_comb (full + intra/inter hae)", (perplex, w2v, eat, usf, swow, cnrvec_comb, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, swow_ha_intrainter, cnrvec_ha_intrainter_comb)),
#                         ("all + swow + cnrvec_fb (full + intra/inter hae)", (perplex, w2v, eat, usf, swow, cnrvec_fb, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, swow_ha_intrainter, cnrvec_ha_intrainter_fb)),
#                         ("all + swow + cnrvec_f_only (full + setup/punch hae)", (perplex, w2v, eat, usf, swow, cnrvec_f_only, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, swow_ha_setuppunch, cnrvec_ha_setuppunch_f_only)),
#                         ("all + swow + cnrvec_comb (full + setup/punch hae)", (perplex, w2v, eat, usf, swow, cnrvec_comb, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, swow_ha_setuppunch, cnrvec_ha_setuppunch_comb)),
#                         ("all + swow + cnrvec_fb (full + setup/punch hae)", (perplex, w2v, eat, usf, swow, cnrvec_fb, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, swow_ha_setuppunch, cnrvec_ha_setuppunch_fb)),
#                         ("all + swow + cnrvec_f_only (full + funny only hae)", (perplex, w2v, eat, usf, swow, cnrvec_f_only, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, swow_ha_funnyonly, cnrvec_ha_funnyonly_f_only)),
#                         ("all + swow + cnrvec_comb (full + funny only hae)", (perplex, w2v, eat, usf, swow, cnrvec_comb, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, swow_ha_funnyonly, cnrvec_ha_funnyonly_comb)),
#                         ("all + swow + cnrvec_fb (full + funny only hae)", (perplex, w2v, eat, usf, swow, cnrvec_fb, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, swow_ha_funnyonly, cnrvec_ha_funnyonly_fb)),

#                         ("all + cnrvec_fb (no hae)", (perplex, w2v, eat, usf, cnrvec_fb)),
#                         ("all + cnrvec_fb (only bow hae)", (perplex, w2v_ha_bow, eat_ha_bow, usf_ha_bow, cnrvec_ha_bow_fb)),
#                         ("all + cnrvec_fb (only intra/inter hae)", (perplex, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, cnrvec_ha_intrainter_fb)),
#                         ("all + cnrvec_fb (only setup/punch hae)", (perplex, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, cnrvec_ha_setuppunch_fb)),
#                         ("all + cnrvec_fb (only funny only hae)", (perplex, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, cnrvec_ha_funnyonly_fb)),
#                         ("all + cnrvec_fb (full + bow hae)", (perplex, w2v, eat, usf, cnrvec_fb, w2v_ha_bow, eat_ha_bow, usf_ha_bow, cnrvec_ha_bow_fb)),
#                         ("all + cnrvec_fb (full + intra/inter hae)", (perplex, w2v, eat, usf, cnrvec_fb, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, cnrvec_ha_intrainter_fb)),
#                         ("all + cnrvec_fb (full + setup/punch hae)", (perplex, w2v, eat, usf, cnrvec_fb, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, cnrvec_ha_setuppunch_fb)),
#                         ("all + cnrvec_fb (full + funny only hae)", (perplex, w2v, eat, usf, cnrvec_fb, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, cnrvec_ha_funnyonly_fb)),

# #                         ("cnrvec_f_only only (no hae)", (cnrvec_f_only,)),
# #                         ("cnrvec_comb only (no hae)", (cnrvec_comb,)),
#                         ("cnrvec_fb only (no hae)", (cnrvec_fb,)),
# #                         ("cnrvec_f_only only (only bow hae)", (cnrvec_ha_bow_f_only,)),
# #                         ("cnrvec_comb only (only bow hae)", (cnrvec_ha_bow_comb,)),
#                         ("cnrvec_fb only (only bow hae)", (cnrvec_ha_bow_fb,)),
# #                         ("cnrvec_f_only only (only intra/inter hae)", (cnrvec_ha_intrainter_f_only,)),
# #                         ("cnrvec_comb only (only intra/inter hae)", (cnrvec_ha_intrainter_comb,)),
#                         ("cnrvec_fb only (only intra/inter hae)", (cnrvec_ha_intrainter_fb,)),
# #                         ("cnrvec_f_only only (only setup/punch hae)", (cnrvec_ha_setuppunch_f_only,)),
# #                         ("cnrvec_comb only (only setup/punch hae)", (cnrvec_ha_setuppunch_comb,)),
#                         ("cnrvec_fb only (only setup/punch hae)", (cnrvec_ha_setuppunch_fb,)),
# #                         ("cnrvec_f_only only (only funny only hae)", (cnrvec_ha_funnyonly_f_only,)),
# #                         ("cnrvec_comb only (only funny only hae)", (cnrvec_ha_funnyonly_comb,)),
#                         ("cnrvec_fb only (only funny only hae)", (cnrvec_ha_funnyonly_fb,)),
# #                         ("cnrvec_f_only only (full + bow hae)", (cnrvec_f_only, cnrvec_ha_bow_f_only)),
# #                         ("cnrvec_comb only (full + bow hae)", (cnrvec_comb, cnrvec_ha_bow_comb)),
#                         ("cnrvec_fb only (full + bow hae)", (cnrvec_fb, cnrvec_ha_bow_fb)),
# #                         ("cnrvec_f_only only (full + intra/inter hae)", (cnrvec_f_only, cnrvec_ha_intrainter_f_only)),
# #                         ("cnrvec_comb only (full + intra/inter hae)", (cnrvec_comb, cnrvec_ha_intrainter_comb)),
#                         ("cnrvec_fb only (full + intra/inter hae)", (cnrvec_fb, cnrvec_ha_intrainter_fb)),
# #                         ("cnrvec_f_only only (full + setup/punch hae)", (cnrvec_f_only, cnrvec_ha_setuppunch_f_only)),
# #                         ("cnrvec_comb only (full + setup/punch hae)", (cnrvec_comb, cnrvec_ha_setuppunch_comb)),
#                         ("cnrvec_fb only (full + setup/punch hae)", (cnrvec_fb, cnrvec_ha_setuppunch_fb)),
# #                         ("cnrvec_f_only only (full + funny only hae)", (cnrvec_f_only, cnrvec_ha_funnyonly_f_only)),
# #                         ("cnrvec_comb only (full + funny only hae)", (cnrvec_comb, cnrvec_ha_funnyonly_comb)),
#                         ("cnrvec_fb only (full + funny only hae)", (cnrvec_fb, cnrvec_ha_funnyonly_fb)),
                        
#                         ("all + swow + cnnb + cnrvec_fb (no hae)", (perplex, w2v, eat, usf, swow, cnnb, cnrvec_fb)),
#                         ("all + swow + cnnb + cnrvec_fb (only bow hae)", (perplex, w2v_ha_bow, eat_ha_bow, usf_ha_bow, swow_ha_bow, cnrvec_ha_bow_fb)),
#                         ("all + swow + cnnb + cnrvec_fb (only intra/inter hae)", (perplex, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, swow_ha_intrainter, cnnb_ha_bow, cnrvec_ha_intrainter_fb)),
#                         ("all + swow + cnnb + cnrvec_fb (only setup/punch hae)", (perplex, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, swow_ha_setuppunch, cnnb_ha_intrainter, cnrvec_ha_setuppunch_fb)),
#                         ("all + swow + cnnb + cnrvec_fb (only funny only hae)", (perplex, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, swow_ha_funnyonly, cnnb_ha_funnyonly, cnrvec_ha_funnyonly_fb)),
#                         ("all + swow + cnnb + cnrvec_fb (full + bow hae)", (perplex, w2v, eat, usf, swow, cnnb, cnrvec_fb, w2v_ha_bow, eat_ha_bow, usf_ha_bow, swow_ha_bow, cnnb_ha_bow, cnrvec_ha_bow_fb)),
#                         ("all + swow + cnnb + cnrvec_fb (full + intra/inter hae)", (perplex, w2v, eat, usf, swow, cnnb, cnrvec_fb, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, swow_ha_intrainter, cnnb_ha_intrainter, cnrvec_ha_intrainter_fb)),
#                         ("all + swow + cnnb + cnrvec_fb (full + setup/punch hae)", (perplex, w2v, eat, usf, swow, cnnb, cnrvec_fb, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, swow_ha_setuppunch, cnnb_ha_setuppunch, cnrvec_ha_setuppunch_fb)),
#                         ("all + swow + cnnb + cnrvec_fb (full + funny only hae)", (perplex, w2v, eat, usf, swow, cnnb, cnrvec_fb, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, swow_ha_funnyonly, cnnb_ha_funnyonly, cnrvec_ha_funnyonly_fb)),

#                         ("all + cnnb + cnrvec_fb (no hae)", (perplex, w2v, eat, usf, cnnb, cnrvec_fb)),
#                         ("all + cnnb + cnrvec_fb (only bow hae)", (perplex, w2v_ha_bow, eat_ha_bow, usf_ha_bow, cnrvec_ha_bow_fb)),
#                         ("all + cnnb + cnrvec_fb (only intra/inter hae)", (perplex, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, cnnb_ha_bow, cnrvec_ha_intrainter_fb)),
#                         ("all + cnnb + cnrvec_fb (only setup/punch hae)", (perplex, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, cnnb_ha_intrainter, cnrvec_ha_setuppunch_fb)),
#                         ("all + cnnb + cnrvec_fb (only funny only hae)", (perplex, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, cnnb_ha_funnyonly, cnrvec_ha_funnyonly_fb)),
#                         ("all + cnnb + cnrvec_fb (full + bow hae)", (perplex, w2v, eat, usf, cnnb, cnrvec_fb, w2v_ha_bow, eat_ha_bow, usf_ha_bow, cnnb_ha_bow, cnrvec_ha_bow_fb)),
#                         ("all + cnnb + cnrvec_fb (full + intra/inter hae)", (perplex, w2v, eat, usf, cnnb, cnrvec_fb, w2v_ha_intrainter, eat_ha_intrainter, usf_ha_intrainter, cnnb_ha_intrainter, cnrvec_ha_intrainter_fb)),
#                         ("all + cnnb + cnrvec_fb (full + setup/punch hae)", (perplex, w2v, eat, usf, cnnb, cnrvec_fb, w2v_ha_setuppunch, eat_ha_setuppunch, usf_ha_setuppunch, cnnb_ha_setuppunch, cnrvec_ha_setuppunch_fb)),
#                         ("all + cnnb + cnrvec_fb (full + funny only hae)", (perplex, w2v, eat, usf, cnnb, cnrvec_fb, w2v_ha_funnyonly, eat_ha_funnyonly, usf_ha_funnyonly, cnnb_ha_funnyonly, cnrvec_ha_funnyonly_fb)),
                       
                       ]
        
        def run_test(features, splits):
            clf = RandomForestClassifier(n_estimators=100, n_jobs=1) #min_samples_leaf=100,
#             clf = LinearSVC(C=0.1, dual=False)
#             clf=SVC(kernel="poly")
            train_subset = np.hstack(features)
#             clf = Pipeline((("scale", StandardScaler()),
#                             ("predict", clf),
#                             )
#                            )
    #         test_subset = np.delete(test_feat, [i for i in range(test_feat.shape[1]) if i not in columns_to_keep], 1)
    #              
    #         clf.fit(train_subset,train_y)
    #         pred_y = clf.predict(test_subset)
               
#             pred_y = cross_val_predict(clf, train_subset, train_y, cv=splits, n_jobs=3)
#             pred_y = cross_val_predict(clf, train_subset, labels, cv=splits, n_jobs=3)
            pred_y = cross_val_predict(clf, train_subset, labels, cv=10, n_jobs=3)
                      
    #         print(time.time()-start)
    
            split_results = ([], [], [], [])
            for _, test_i in splits:
                test_label, test_pred = [labels[k] for k in test_i], [pred_y[k] for k in test_i]
                split_p,split_r,split_f,_ = precision_recall_fscore_support(test_label, test_pred, average="binary")
                split_a = accuracy_score(test_label, test_pred)
                for i, score in enumerate((split_a,split_p,split_r,split_f)):
                    split_results[i].append(score)
                    
            p,r,f,_ = precision_recall_fscore_support(labels, pred_y, average="binary")
            a = accuracy_score(labels, pred_y)
             
            score_str = f"{a}\t{p}\t{r}\t{f}"
            
            return score_str, split_results
        
        print("starting test")
        results = []
        test_labels=[]
          
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        from sklearn.model_selection import StratifiedKFold
        from scipy.stats import ttest_rel, ttest_ind
          
#         with open("oneliner_splits_redo.pkl", "rb") as f:
#             splits=pickle.load(f)

        #TODO: Does this fix SVM's issues?
        labels = LabelBinarizer().fit_transform(labels)
        labels=labels.ravel()
        
        kfolds = StratifiedKFold(10, random_state=30) #random_state makes splits repeatable
        splits = list(kfolds.split(docs, labels))
        
        
        #get baseline scores
        test_labels.append(test_setups[0][0])
        score_str, baseline_split_results = run_test(test_setups[0][1], splits)
        results.append(score_str)
        
        
        for i, (test_label, features) in enumerate(test_setups[1:]):
            test_labels.append(test_label)
#     #         print(test_label)
#     #         import time
#     #         start = time.time()
#     #         print(i)
#             clf = RandomForestClassifier(n_estimators=100, n_jobs=1, verbose=1) #min_samples_leaf=100,
#             train_subset = np.hstack(features)
#     #         test_subset = np.delete(test_feat, [i for i in range(test_feat.shape[1]) if i not in columns_to_keep], 1)
#     #              
#     #         clf.fit(train_subset,train_y)
#     #         pred_y = clf.predict(test_subset)
#                
#     #         pred_y = cross_val_predict(clf, train_subset, train_y, cv=splits, n_jobs=3)
#             pred_y = cross_val_predict(clf, train_subset, labels, cv=10, n_jobs=3)
#                       
#     #         print(time.time()-start)
#                       
#             p,r,f,_ = precision_recall_fscore_support(labels, pred_y, average="binary")
#             a = accuracy_score(labels, pred_y)
#              
#             results.append(f"{a}\t{p}\t{r}\t{f}")
            score_str, split_results = run_test(features, splits)
#             results.append(score_str)
            
            sig_results = []
            for bline,ha in zip(baseline_split_results,split_results):
#                 t_value, p_value = ttest_rel(ha,bline) #order matters for the t value. positive t for (b,a) means b is higher than a
                t_value, p_value = ttest_ind(ha,bline) #order matters for the t value. positive t for (b,a) means b is higher than a
                sig_str = ""
                if (t_value > 0):
                    if p_value/2 < 0.05:
                        if p_value/2 < 0.005:
                            sig_str = "**"
                        else:
                            sig_str = "*"
                sig_results.append(sig_str)
            
            sig_results_str="\t".join(sig_results)
             
            results.append(f"{score_str}\t|\t{sig_results_str}")
            
            
            print(f"{test_label} done")
    #         print(f"{test_label}\t\t{a}\t{p")
    #         print(f"Accuracy: {a}")
    #         print(f"Precision: {p}")
    #         print(f"Recall: {r}")
    #         print(f"F-Score: {f}")
        result_strs.append((name, "\n".join(test_labels), "\n".join(results)))
#         print()
    
    for n, tl, r in result_strs:
        print(f"\n\n{n}")
        print(tl)
        print(r)