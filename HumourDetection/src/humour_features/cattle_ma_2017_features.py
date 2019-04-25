'''
Created on Feb 3, 2018

@author: Andrew Cattle <acattle@cse.ust.hk>

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
import scipy


class CattleMaHumourFeatureExtractor(TransformerMixin, LoggerMixin):
    """
        A class for implementing the features used in Cattle and Ma (2017) as a
        scikit-learn transformer, suitable for use in scikit-learn pipelines.
    """
    
    def __init__(self, w2v_model_getter, eat_scorer, usf_scorer, perplexity_scorer, humour_anchor_identifier=None, verbose=False):
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
    
    def get_humour_anchors(self, documents):
        """
            Get the humour anchors for all documents.
            
            Since multiple functions extract humour anchors, it's more efficient
            to extract them one time in advance
            
            :param documents: documents to be processed. Each document should be a sequence of tokens
            :type documents: Iterable[Iterable[str]]
            
            :returns: each document's humour anchors
            :rtype: List[List[str]]
        """
        
        humour_anchors = []
        if self._get_humour_anchors:
            processed=0
            total = len(documents)
            for doc in documents:
                #TODO: would it be more efficient to perform this at documents level?
                humour_anchors.append(self._get_humour_anchors(doc))
                processed+=1
                if processed%500 == 0:
                    print(f"{processed}/{total}")
        else:
            #if no humour identifier specified, just return documents
            humour_anchors = documents
        
        return humour_anchors
    
    def get_association_strengths(self, documents, association_scorer, skip_humour_anchors=False, dataset=""):
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
    
    def get_w2v_sims(self, documents, skip_humour_anchors=False):
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
        
        w2v_scorer = self.get_w2v().get_similarity
        return get_interword_score_features(documents, w2v_scorer, token_filter=anchor_identifier)
    
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
        
#         from nltk import word_tokenize
#         X_tok = [word_tokenize(d) for d in X]
        
        features = []
#         self.logger.debug("Adding ngram features")
#         #TODO: ensemble with the ngram features?
#         features.append(self.count_vectorizer.transform(X))
#         self.logger.debug("Adding perplexities")
#         features.append(self.get_perplexities(X_tok))
         
        #Since multiple functions take advantage of humour anchor extraction, makes sense to extract them only once
        self.logger.debug("Extracting humour anchors")
        humour_anchors = self.get_humour_anchors(X)
         
        self.logger.debug("Adding Word2Vec similarities")
        features.append(self.get_w2v_sims(humour_anchors, skip_humour_anchors=True))
        self.logger.debug("Adding EAT association strengths")
        features.append(self.get_association_strengths(humour_anchors, self.get_eat_scores, skip_humour_anchors=True, dataset="eat"))
        self.logger.debug("Adding USF association strengths")
        features.append(self.get_association_strengths(humour_anchors, self.get_usf_scores, skip_humour_anchors=True, dataset="usf"))
        
        
#         return scipy.sparse.hstack(features)
        return np.hstack(features)

if __name__ == "__main__":
    
    from nltk import word_tokenize
    import re
    from string import punctuation
    from sklearn.ensemble.forest import RandomForestClassifier
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
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
    
    from util.text_processing import default_preprocessing_and_tokenization
#     docs = default_preprocessing_and_tokenization(docs) #Remove stopwords
#     docs = default_preprocessing_and_tokenization(docs, stopwords=[])
#     from util.text_processing import default_preprocessing_and_tokenization
#     docs = default_preprocessing_and_tokenization(docs)
#     #TODO: this also gets rid of negation words. Negation might be a helpful feature
#     from nltk import word_tokenize
#     docs = [word_tokenize(doc) for doc in docs]
     
    test_size = round(len(docs_and_labels) * 0.1) #90/10 training test split
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
        get_google_autoextend, get_stanford_glove, get_wikipedia_word2gauss
        
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
    with open("yang_pipeline_ol.dill", "rb") as f:
#     with open("yang_pipeline_potd.dill", "rb") as f:
        yang = dill.load(f)
    yang.named_steps[]
    from nltk.parse.stanford import StanfordParser
    parser = StanfordParser()
    hae = YangHumourAnchorExtractor(lambda x: next(parser.parse(x)), yang, 3)
    
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
    from word_associations.association_readers.query_igraph import get_strengths_wsl
    eat_func = partial(get_strengths_wsl, dataset="eat", pajek_loc="/mnt/d/git/HumourDetection/HumourDetection/src/Data/eat/pajek/EATnew2.net", tmpdir="d:/temp")
    usf_func = partial(get_strengths_wsl, dataset="usf", pajek_loc="/mnt/d/git/HumourDetection/HumourDetection/src/Data/PairsFSG2.net", tmpdir="d:/temp")
    
    from nltk.corpus import stopwords
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
    ENGLISH_STOPWORDS.add("n't")
    punc_re = re.compile(f'[{re.escape(punctuation)}]')
    
    def no_stopwords(document):
        sanitized_doc = []
        for token in document:
            if token.lower() not in ENGLISH_STOPWORDS:
                sanitized_doc.append(token)
        
        return sanitized_doc
    
    def tokenizer(document):
        
        tokens = word_tokenize(document.lower()) #Tokenize and lowercase document
        #TODO: keep sentence information?
        
        processed_tokens = []
        for token in tokens:
            if token not in ENGLISH_STOPWORDS:
                token = punc_re.sub("", token) #remove punctuation
                #https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
                
                if token: #if this isn't an empty string
                    processed_tokens.append(token)
        
        return processed_tokens
        
    from util.keras_pipeline_persistance import save_keras_pipeline, load_keras_pipeline
    from util.keras_metrics import r2_score, pearsonr #for compatibility with saved pipelines
    from word_associations.strength_prediction import _create_mlp
#     eat_predictor = train_association_predictor(eat)
#     save_keras_pipeline("d:/datasets/trained models/eat", eat_predictor)
#     eat_predictor = load_keras_pipeline("d:/datasets/trained models/eat")
    eat_predictor = load_keras_pipeline("d:/git/HumourDetection/HumourDetection/src/word_associations/models/eat-all")#,custom_objects={"r2_score":r2_score, "pearsonr":pearsonr})
    usf_predictor = load_keras_pipeline("d:/git/HumourDetection/HumourDetection/src/word_associations/models/usf-all")#,custom_objects={"r2_score":r2_score, "pearsonr":pearsonr})
    
    import logging
    for p in (eat_predictor, usf_predictor):
        p.named_steps["extract features"].logger.setLevel(logging.DEBUG)
    
    from perplexity.kenlm_model import KenLMSubprocessWrapper
    kenlm_query_loc = "/mnt/d/git/kenlm-stable/build/bin/query"
    kenlm_model_loc = "/mnt/d/datasets/news-discuss_3.bin"
    klm = KenLMSubprocessWrapper(kenlm_model_loc, kenlm_query_loc, "d:/temp")
    
    
#     print(usf.get_association_strengths([("dog", "cat"), ("car", "bus")]))
      
    
#     humour_pipeline = Pipeline([("feature extraction", CattleMaHumourFeatureExtractor(get_google_word2vec, eat.get_association_strengths, usf.get_association_strengths, klm.get_perplexities, None, True)),
#                                 ("estimator", RandomForestClassifier(n_estimators=100, min_samples_leaf=100, n_jobs=-1, verbose=1))])
#       
#     humour_pipeline.fit(train_X, train_y)
#     pred_y = humour_pipeline.predict(test_X)

    print("starting feature extract")
    fe = CattleMaHumourFeatureExtractor(get_google_word2vec, eat_func, usf_func, klm.get_perplexities, hae.find_humour_anchors, True)#eat_predictor.predict, usf_predictor.predict
    train_feat = fe.fit_transform(train_X, train_y)
    test_feat = fe.transform(test_X)
        
    np.save("potd_raw_hae_graph_train_X",train_feat)
    np.save("potd_raw_hae_graph_test_X",test_feat)
    np.save("potd_raw_hae_graph_train_y", train_y)
    np.save("potd_raw_hae_graph_test_y", test_y)
#     train_feat=np.load("potd_raw_no_stop_ml_train_X.npy")
#     test_feat=np.load("potd_raw_no_stop_ml_test_X.npy")
#     train_y=np.load("potd_raw_no_stop_ml_train_y.npy")
#     test_y=np.load("potd_raw_no_stop_ml_test_y.npy")
     
    #(test label, columns to keep)
    test_setups = [("all", list(range(train_feat.shape[1]))),
                   ("perplex only", [0]),
                   ("word2vec only", [1,2,3]),
                   ("all word assoc", [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]),
                   ("eat only", [4,5,6,7,8,9,10,11,12,13,14,15]),
                   ("eat forward only", [4,5,6]),
                   ("eat backward only", [7,8,9]),
                   ("eat diff only", [10,11,12,13,14,15]),
                   ("eat micro diff only", [10,11,12]),
                   ("eat macro diff only", [13,14,15]),
                   ("usf only", [16,17,18,19,20,21,22,23,24,25,26,27]),
                   ("usf forward only", [16,17,18]),
                   ("usf backward only", [19,20,21]),
                   ("usf diff only", [22,23,24,25,26,27]),
                   ("usf micro diff only", [22,23,24]),
                   ("usf macro diff only", [25,26,27])
                   ]
     
    print("starting test")
    for test_label, columns_to_keep in test_setups:
        clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, verbose=1) #min_samples_leaf=100,
        train_subset = np.delete(train_feat, [i for i in range(train_feat.shape[1]) if i not in columns_to_keep], 1)
        test_subset = np.delete(test_feat, [i for i in range(test_feat.shape[1]) if i not in columns_to_keep], 1)
          
        clf.fit(train_subset,train_y)
        pred_y = clf.predict(test_subset)
            
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        p,r,f,_ = precision_recall_fscore_support(test_y, pred_y, average="binary")
        a = accuracy_score(test_y, pred_y)
        print(f"{test_label}")
        print(f"Accuracy: {a}")
        print(f"Precision: {p}")
        print(f"Recall: {r}")
        print(f"F-Score: {f}")
    
    
    