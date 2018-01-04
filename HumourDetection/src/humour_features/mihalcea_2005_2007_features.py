'''
    Created on Dec 19, 2017

    :author: Andrew Cattle <acattle@cse.ust.hk>

    This module implements the humour features described in Mihalcea and
    Strapparava (2005) and Mihalcea and Pulman (2007)

        Mihalcea, R., & Pulman, S. (2007). Characterizing humour: An exploration
        of features in humorous texts. Computational Linguistics and Intelligent
        Text Processing, 337-347.
        
        Mihalcea, R., & Strapparava, C. (2005, October). Making computers laugh:
        Investigations in automatic humor recognition. In Proceedings of the
        Conference on Human Language Technology and Empirical Methods in Natural
        Language Processing (pp. 531-538). Association for Computational
        Linguistics.
'''

import numpy as np
from nltk.corpus import wordnet as wn
from humour_features.utils.common_features import get_alliteration_and_rhyme_features
from humour_features.utils.wordnet_domains import WordNetDomains
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm.classes import LinearSVC
from sklearn.pipeline import Pipeline

def train_mihalcea_strapparava_2005_pipeline(X, y, wnd_loc, content_model="svm"):
    """
        Train a  humour classifier using features described in Mihalcea and
        Strapparava (2005)
        
        :param X: Training documents. Each document should be a sequence of tokens.
        :type X: list(list(str))
        :param y: training labels
        :type y: list(int)
        :param wnd_loc: location of WordNet Domains file
        :type wnd_loc: str
        :param content_model: type of content model to use for content features. Must be "nb" or "svm"
        :type content_model: str
        
        :return: A trained pipeline that takes in tokenized documents and outputs predictions
        :rtype: sklearn.pipeline.Pipeline
    """
    
    #Mihalcea and Strapparava (2005) uses TimBL, which seems to just be a modified KNN that remembers fewer datapoints.
    #Sklearn doesn't implement that, but it does implement KNN (and computers have come a long way in the lst 12 years)
    from sklearn.neighbors.classification import KNeighborsClassifier
    mihalcea2005_pipeline = Pipeline([("feature_extraction", MihalceaFeatureExtractor(wnd_loc, content_model)),
                                      ("knn_classifier", KNeighborsClassifier())
                                      ])
    
    mihalcea2005_pipeline.fit(X,y)
    
    return mihalcea2005_pipeline

def train_mihalcea_pulman_2007_pipeline(X,y):
    """
        Train a  humour classifier using features described in Mihalcea and
        Pulman (2007)
        
        :param X: Training documents. Each document should be a sequence of tokens.
        :type X: list(list(str))
        :param y: training labels
        :type y: list(int)
        
        :return: A trained pipeline that takes in tokenized documents and outputs predictions
        :rtype: sklearn.pipeline.Pipeline
    """
    pass

def run_mihalcea_strapparava_2005_baseline(train, test, wnd_loc, content_model="svm"):
    """
        Convenience method for running Mihalcea and Strapparava (2005) humour
        classification experiment on a specified dataset.
        
        :param train: A tuple containing training documents and labels. Each document should be a sequence of tokens.
        :type train: tuple(list(list(str)), list(int))
        :param test: A tuple containing training documents and labels. Each document should be a sequence of tokens.
        :type test: tuple(list(list(str)), list(int))
        :param wnd_loc: location of WordNet Domains file
        :type wnd_loc: str
        :param content_model: type of content model to use for content features. Must be "nb" or "svm"
        :type content_model: str
    """
    X, y = train
    mihalcea2005_pipeline = train_mihalcea_strapparava_2005_pipeline(X, y, wnd_loc, content_model)
    
    test_X, test_y = test
    pred_y = mihalcea2005_pipeline.predict(test_X)
    
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    p,r,f,_ = precision_recall_fscore_support(test_y, pred_y, average="binary")
    a = accuracy_score(test_y, pred_y)
    print("Mihalcea and Strapparava (2005) Humour Classification")
    print("Accuracy: {}".format(a))
    print("Precision: {}".format(p))
    print("Recall: {}".format(r))
    print("F-Score: {}".format(f))

def run_mihalcea_pulman_2007_baseline(train,test):
    """
        Convenience method for running Mihalcea and Pulman (2007) humour
        classification experiment on a specified dataset.
        
        :param train: A tuple containing training documents and labels. Each document should be a sequence of tokens.
        :type train: tuple(list(list(str)), list(int))
        :param test: A tuple containing training documents and labels. Each document should be a sequence of tokens.
        :type test: tuple(list(list(str)), list(int))
    """
    pass

class MihalceaFeatureExtractor(TransformerMixin):
    """
        A class for implementing the features used in Mihalcea and Strapparava
        (2005) and Mihalcea and Pulman (2007) as a scikit-learn transformer,
        suitable for use in scikit-learn pipelines.
    """
    
    def __init__(self, wnd_loc, content_model="svm"):
        """
            Configure Mihalcea-inspired feature extraction options
            
            :param wnd_loc: the location of the WordNet Domains file
            :type wnd_loc: str
            :param content_model: the type of classifier to train for content-based features. Must be "nb" or "svm"
            :param content_model: the type of classifier to train for content-based features. Must be "nb" (for Multinomial Naive Bayes) or "svm" (for Support Vector Machine)
            :type content_model: str
        """
        
        self.wnd_loc = wnd_loc
        self.wnd = None
        
        self.content_model_type = content_model
        self.content_model = None
    
    def _get_wordnet_domains(self):
        """
            Lazy-loads WordNet Domains if required. Returns WordNetDomains object
            
            :returns: WordNetDomains object
            :rtype: humour_features.utils.wordnet_domains.WordNetDomains
        """
        
        if self.wnd == None:
            self.wnd=WordNetDomains(self.wnd_loc)
        
        return self.wnd
    
    def get_alliteration_features(self, documents):
        """
            Calculates the alliteration and rhyme features described in
            Section 3.1.1 of Mihalcea and Strapparava (2005)
            
            :param documents: documents to be processed. Each document is a sequence of tokens
            :type documents: list(list(str))
            
            :return: a matrix where columns represent extracted phonetic style features in the form (alliteration_num, rhyme_num) and rows are documents
            :rtype: numpy.array
        """
        
        allit_features = get_alliteration_and_rhyme_features(documents)
        
        #Mihalcea and Strapparava (2005) only care about number of allit/rhyme chains, not length
        #Therefore we should delete columns (axis=1) 1 and 3 (obj=(1,3))
        return np.delete(allit_features, (1,3), 1)
    
    def get_antonymy_features(self, documents):
        """
            Calculates the antonymy features described in Section 3.1.2 of
            Mihalcea and Strapparava (2005)
            
            :param documents: documents to be processed. Each document is a sequence of tokens
            :type documents: list(list(str))
            
            :return: a matrix where columns represent extracted phonetic style features in the form (alliteration_num, rhyme_num) and rows are documents
            :rtype: numpy.array
        """
        #TODO: include "similar-to" for abjectives (as per Mihalcea 2005 paper)?
        #TODO: how to speed up?
        
        feature_vects = []
        for document in documents:
            encountered_lemmas = set()
            encountered_antonyms = set()
            antonym_count=0
            for word in document:
                #TODO: POS tag?
                word_lemmas = set(wn.lemmas(word))
                word_antonyms = set()
                for lemma in word_lemmas:
                    word_antonyms.update(lemma.antonyms())                    
                
                #https://stackoverflow.com/questions/3170055/test-if-lists-share-any-items-in-python
                if (not word_lemmas.isdisjoint(encountered_antonyms)) or (not word_antonyms.isdisjoint(encountered_lemmas)):
                    #if we're a previous word's antonym or if a previous word is our antonym
                    antonym_count += 1
                    #TODO: Is this correct?
                    #"big small tiny" would return two antonyms ((big, small) and (big, tiny)) but "big large tiny" would only return only (big, tiny)
                    #alternatively, we could store lemmas and antonyms by word then get combinations of indexes
                
                encountered_lemmas.update(word_lemmas)
                encountered_antonyms.update(word_antonyms)
                    
            feature_vects.append(antonym_count)
        
        return np.vstack(feature_vects)
    
    def get_adult_slang_features(self, documents):
        """
            Calculates the adult slang features described in Section 3.1.3 of
            Mihalcea and Strapparava (2005)
            
            :param documents: documents to be processed. Each document is a sequence of tokens
            :type documents: list(list(str))
            
            :return: a matrix where columns represent extracted phonetic style features in the form (alliteration_num, rhyme_num) and rows are documents
            :rtype: numpy.array
        """
        
        wnd = self._get_wordnet_domains()
        
        feature_vects = []
        for document in documents:
            adult_slang_count = 0
            for word in document:
                synsets = wn.synsets(word) #TODO: filter by POS?
                
                if len(synsets) < 4: #filter out high polysemy words
                    #TODO: user specified?
                    domains = set()
                    for synset in synsets:
                        domains.update(self._get_wordnet_domains().get_domains(synset.name()))
                    
                    if "sexuality" in domains:
                        adult_slang_count += 1
            
            feature_vects.append(adult_slang_count)
        
        return np.vstack(feature_vects)
    
    def get_content_features(self, documents):
        """
            Calculates the adult slang features described in Section 3.1.3 of
            Mihalcea and Strapparava (2005)
            
            :param documents: documents to be processed. Each document is a sequence of tokens
            :type documents: list(list(str))
            
            :return: a matrix where columns represent extracted phonetic style features in the form (alliteration_num, rhyme_num) and rows are documents
            :rtype: numpy.array
            
            :raises NotFittedError: If content model hasn't been initialized
        """
        
        if self.content_model == None:
            raise NotFittedError("Must fit MihalceaFeatureExtractor before content features can be extracted.")
        
        return self.content_model.predict(documents).reshape((-1,1)) #reshape from vector to Nx1 matrix
    
    def fit(self,X,y):
        """
            Fits content model so that content features can be extracted
            
            :returns: self (for compatibility with sklearn pipelines)
            :rtype: TransformerMixin
            
            :raises ValueError: if specified content model type is not "nb" or "svm"
        """
        
        steps =[("count_vector", CountVectorizer(tokenizer=lambda x:  x, preprocessor=lambda x: x))] #skip tokenization and preporcessing
        
        if self.content_model_type == "nb":
            steps.append(("naive_bayes", MultinomialNB())) #Multinomal, as specified in Mihalcea and Strapparava (2005)
            
        elif self.content_model_type == "svm":
            steps.append(("svm", LinearSVC()))
            
        else:
            raise ValueError("Unknown content model type '{}'. Must be 'nb' or 'svm'.".format(self.content_model_type))
        
        self.content_model = Pipeline(steps)
        
        self.content_model.fit(X, y)
        
        return self

    def transform(self, X, y=None):
        """
            Takes in a  series of tokenized documents and extracts a set of
            features equivalent to the highest performing model in Mihalcea
            and Strapparava (2005)
            
            :param X: pre-tokenized documents
            :type X: list(list(str))
            
            :return: highest performing Mihalcea and Strapparava (2005) features as a numpy array
            :rtype: numpy.array
        """
        
        features = []
        
        features.append(self.get_alliteration_features(X))
        features.append(self.get_antonymy_features(X))
        features.append(self.get_adult_slang_features(X))
        features.append(self.get_content_features(X))

        return np.hstack(features)

if __name__ == "__main__":
    potd_loc = "D:/datasets/pun of the day/puns_pos_neg_data.csv"
    oneliners_loc = "D:/datasets/16000 oneliners/Jokes16000.txt"
    wnd_loc = "C:/vectors/lifted-wordnet-domains-develop/wordnet-domains-3.2-wordnet-3.0.txt"
    
#     potd_loc = "/mnt/d/datasets/pun of the day/puns_pos_neg_data.csv"
#     oneliners_loc = "/mnt/d/datasets/16000 oneliners/Jokes16000.txt"
#     wnd_loc = "/mnt/c/vectors/lifted-wordnet-domains-develop/wordnet-domains-3.2-wordnet-3.0.txt"
    
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
     
    test_size = round(len(docs_and_labels)*0.1) #hold out 10% as test
     
    test = zip(*docs_and_labels[:test_size]) #unzip the documents and labels
    train = zip(*docs_and_labels[test_size:])
      
      
    run_mihalcea_strapparava_2005_baseline(train, test, wnd_loc)