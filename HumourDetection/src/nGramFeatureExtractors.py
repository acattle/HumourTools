'''
Created on 2013-08-18

@author: Andrew
'''
import nltk
from numpy import mean, std
from math import pow, sqrt
#import csv
#from sklearn import metrics, cross_validation
#from sklearn.feature_extraction.dict_vectorizer import DictVectorizer

def escapePOS(pos):
    """A simple function to escape POS tags for situations where the
    POS tag is the same as the word itself (i.e. symbols and
    punctuations). This is important since words and POS tags are held
    held in a single dictionary, thus the keys should be unique.
    
    @var pos: the POS tag to be escaped
    @type pos: str
    
    @return : the escaped POS tag
    @rtype : str
    """
    
    return u"{} {}".format(POS_TAG, pos)

def cleanupEnds(text):
    oldText=text
    newText=""
    while newText != oldText:
        oldText = newText
        newText = hashtagEndPattern.sub("", oldText).strip()
        newText = urlEndPattern.sub("", newText).strip()
        newText = toUserEndPattern.sub("", newText).strip()
        
    return newText

def sanitizeText(text):
    sanitizedText = text.lower()
    
#    sanitizedText = cleanupEnds(sanitizedText)
    
    sanitizedText = toUserPattern.sub(u"TOUSER", sanitizedText)
    sanitizedText = urlPattern.sub(u"URL", sanitizedText)
    #remove user tags
    sanitizedText = hyperbolicPattern.sub(u"", sanitizedText)
    sanitizedText = understatementPattern.sub(u"", sanitizedText)
    sanitizedText = rhetoricalPattern.sub(u"", sanitizedText)
    sanitizedText = sarcasticPattern.sub(u"", sanitizedText)
    #escape hashtags
    sanitizedText = hashtagPattern.sub(ur"HASH_\1", sanitizedText)
    #correct ellipsis for better NLTK performance
    sanitizedText = ellipsisPattern.sub(u"...", sanitizedText)
    return sanitizedText.strip()

def getNGram(words, i, n):
    """Takes a list of (word, pos) pairs and generates features for both
    the word-based ngram and pos-based ngram
    
    @var words: the list of word, POS tuples
    @type words: [(str, str)]
    @var i: the index to start the ngram from
    @type i: int
    @var n: the size of the ngram (i.e. n=1 for unigrams, n=2 for bigrams, etc.)
    @type n: int
    
    @return the word-based ngram and the pos based n-gram in that order
    @rtype (str, str)
    """
    ngram = words[i:i+n] #+ because we want n items
    wordNGram = u" ".join(word[0] for word in ngram)
    posNGram = u" ".join(word[1] for word in ngram)
    
    return (wordNGram, posNGram)

#def getNGramFeatures(text, ngrams=[1,2], sanitize=sanitizeText):
def getNGramFeatures(text, ngrams=[1,2], fullText=True, rawTokens=True):
    """This method converts a document into ngram-based features. These features
    are returned as a dictionary in the form {n : ngram feature dict} in order to
    increase efficiency, allowing multiple combinations of ngrams to be built out
    of only a single call to this method (i.e. testing unigrams, bigrams, and
    unigrams+bigrams)
    
    @var text: the text to be converted into features
    @type text: str
    @var ngrams: list containing all the desired nGrams
    @type ngrams: [int]
    @var sanitize: the function for sanitizing the text before feature extraction
    @type sanitize: func(str) :: str
    
    @return: dictionary where the key is the ngram size and the value is the ngram features as their own dictionary
    @rtype: {int : dict} 
    """
    ngramFeatures = {} #holds
    
    #initialize ngram dicts
    for n in ngrams:
        ngramFeatures[n] = {} 
    
#    if sanitize: #if sanitization method specified
#        text = sanitize(text)
    text = sanitizeText(text)
    
    #if we don't break text into sentences first, word_tokenize produces strange results
    words = []
    for sent in nltk.sent_tokenize(text):
        words.extend(nltk.word_tokenize(sent))
    wordsPOS = nltk.pos_tag(words)
    
    #count ngrams
    for i in xrange(0, len(wordsPOS)):
        for n in ngrams:
            nInt = int(n)
            #for each type of ngram we want to collect
            if (i + nInt - 1) < len(wordsPOS):
                #check if we have enough items to build the ngram (i.e. is the index of the final item in the ngram still inside the list) 
                ngramWord, ngramPOS = getNGram(wordsPOS, i, nInt)
                ngramFeatures[n][ngramWord] = ngramFeatures[n].get(ngramWord, 0) + 1 #count the word
                ngramFeatures[n][escapePOS(ngramPOS)] = ngramFeatures[n].get(escapePOS(ngramPOS), 0) + 1 #count the POS        
    
    if fullText: #if we want to add the full text
        ngramFeatures[FULL_TEXT] = text
    if rawTokens: #if we want to add the raw tokens
        ngramFeatures[RAW_TOKENS] = wordsPOS
    
    return ngramFeatures

#def getFeaturesByN(documents, ngrams=[1,2], sanitize=None):
def getFeaturesByN(documents, ngrams=[1,2], fullText=True, rawTokens=True, minCount=0):
    """This method extracts features from a list of tagged documents but keeps
    them separated by ngram types for more efficient feature extraction when
    testing multiple ngram combinations (i.e. unigrams, bigrams, and
    unigrams+bigrams) 
    
    @var documents: list of tuples containing the document's ID number, text, and tag (in that order)
    @type text: [(int, str, str)]
    @var ngrams: list containing all the desired nGrams
    @type ngrams: [int]
    @var sanitize: the function for sanitizing text before feature extraction
    @type sanitize: func(str) :: str
    
    @return: dictionary where the key is the document's ID and the value is that document's feature dictionary separated by ngram type and it's tag
    @rtype: {int : (dict, str)} 
    """
    features = {}
    
    for document in documents:
        docID, text, tag = document
        
#        ngramFeatures = getNGramFeatures(text, ngrams, sanitize)
        ngramFeatures = getNGramFeatures(text, ngrams, fullText, rawTokens)
        
        features[docID] = (ngramFeatures, tag)
    
    if minCount: #if we want to filter by minimum count
        features = filterLowOccurances(features, ngrams, minCount)
    
    return features

#def getFeatures(documents, ngrams=[1,2], sanitize=None):
def getFeatures(documents, ngrams=[1,2]):
    """This method extracts features from a list of tagged documents
    
    @var documents: list of tuples containing the document's ID number, text, and tag (in that order)
    @type text: [(int, str, str)]
    @var ngrams: list containing all the desired nGrams
    @type ngrams: [int]
    @var sanitize: the function for sanitizing text before feature extraction
    @type sanitize: func(str) :: str
    
    @return: dictionary where the key is the document's ID and the value is that document's feature dictionary and it's tag
    @rtype: {int : (dict, str)} 
    """
    features = {}
    
    for document in documents:
        documentFeatures = {}
        docID, text, tag = document
        
#        ngramFeatures = getNGramFeatures(text, ngrams, sanitize)
        ngramFeatures = getNGramFeatures(text, ngrams)
        
        #combine the ngram dicts into a single dict
        for n in ngrams:
            documentFeatures = dict(documentFeatures.items() + ngramFeatures[n].items())
        
        features[docID] = (documentFeatures, tag)
    
    return features

def dataFormatter(featuresByN, ngrams=[1,2]):
    features = {}
    labels = {}
    for docID in featuresByN:
        ngramFeatures, tag = featuresByN[docID]
        combinedDict = {}
        for n in ngrams:
            combinedDict = dict(combinedDict.items() + ngramFeatures[n].items())
        features[docID] = combinedDict
        labels[docID] = tag
    
    return features, labels

def trainAndTestClassifierCrossValidateNGrams(docIDs, featuresByN, ngrams=[1,2], targetTag=None, classifierType=LinearSVC, folds=10, rowLabel="", csvWriter=None, baseline=None,spacedRows=False): 
   
    combinedFeatures, labels = dataFormatter(featuresByN, ngrams)
    
    kF = KFold(len(docIDs), n_folds=folds)
    
    precisionList = []
    recallList = []
    f_scoreList = []
    for trainIndexes, testIndexes in kF:
        trainData = []
        trainLabels = []
        testData = []
        testLabels = []
        for i in trainIndexes:
            trainData.append(combinedFeatures[docIDs[i]])
            trainLabels.append(labels[docIDs[i]])
        for i in testIndexes:
            testData.append(combinedFeatures[docIDs[i]])
            testLabels.append(labels[docIDs[i]])
            
        precision, recall, f_score = trainAndTestClassifier(trainData, trainLabels, testData, testLabels, targetTag, classifierType)
        
        precisionList.append(precision)
        recallList.append(recall)
        f_scoreList.append(f_score)
        
        #TODO: delete
        break
    
    precisionMean = mean(precisionList)
    precisionSD = std(precisionList)
    recallMean = mean(recallList)
    recallSD = std(recallList)
    fScoreMean = mean(f_scoreList)
    fScoreSD = std(f_scoreList)
    
    if baseline:
        precisionT, precisionP = ttest_rel(precisionList, baseline[0])
        recallT,recallP = ttest_rel(recallList, baseline[1])
        fScoreT,fScoreP = ttest_rel(f_scoreList, baseline[2])
        
        if csvWriter:
            csvWriter.writerow([rowLabel, precisionMean,precisionSD,precisionT,precisionP, recallMean, recallSD,recallT,recallP, fScoreMean, fScoreSD,fScoreT,fScoreP])
        else:
            print "{}-Fold Cross-validated".format(folds)
            print "Precision: {} (sd={}, t={}, p={})".format(precisionMean,precisionSD,precisionT,precisionP)
            print "Recall: {} (sd={}, t={}, p={})".format(recallMean, recallSD,recallT,recallP)
            print "F-Score: {} (sd={}, t={}, p={})".format(fScoreMean, fScoreSD,fScoreT,fScoreP)
    else:    
        if csvWriter:
            if spacedRows:
                csvWriter.writerow([rowLabel, precisionMean,precisionSD,"","", recallMean, recallSD,"","", fScoreMean, fScoreSD, "",""])
            else:
                csvWriter.writerow([rowLabel, precisionMean,precisionSD, recallMean, recallSD, fScoreMean, fScoreSD])
        else:
            print "{}-Fold Cross-validated".format(folds)
            print "Precision: {} (sd={})".format(precisionMean,precisionSD)
            print "Recall: {} (sd={})".format(recallMean, recallSD)
            print "F-Score: {} (sd={})".format(fScoreMean, fScoreSD)
    
    return (precisionList, recallList, f_scoreList)

def trainAndTestClassifierCrossValidateNGramsPerTag(docIDs, featuresByN, ngrams=[1,2], targetTags=None, classifierType=LinearSVC, folds=10, rowLabel="", csvWriter=None, baseline=None,spacedRows=False): 
   
    combinedFeatures, labels = dataFormatter(featuresByN, ngrams)
    
    kF = KFold(len(docIDs), n_folds=folds)
    
    precisions = {}
    recalls = {}
    f_scores = {}
    for trainIndexes, testIndexes in kF:
        trainData = []
        trainLabels = []
        testData = []
        testLabels = []
        for i in trainIndexes:
            trainData.append(combinedFeatures[docIDs[i]])
            trainLabels.append(labels[docIDs[i]])
        for i in testIndexes:
            testData.append(combinedFeatures[docIDs[i]])
            testLabels.append(labels[docIDs[i]])
            
        precision, recall, f_score = trainAndTestClassifierPerTag(trainData, trainLabels, testData, testLabels, targetTags, classifierType)
        
        for tag in precision.keys():
            if precisions.get(tag) is None:
                precisions[tag] = [] #initialize the tag is necessary
            if recalls.get(tag) is None:
                recalls[tag] = [] #initialize the tag is necessary
            if f_scores.get(tag) is None:
                f_scores[tag] = [] #initialize the tag is necessary
            precisions[tag].append(precision[tag])
            recalls[tag].append(recall[tag])
            f_scores[tag].append(f_score[tag])
    
    
    precisionMeans = {}
    precisionSDs = {}
    recallMeans = {}
    recallSDs = {}
    fScoreMeans = {}
    fScoreSDs = {}
    
    for tag in precisions.keys():
        precisionMeans[tag] = mean(precisions[tag])
        precisionSDs[tag] = std(precisions[tag])
        recallMeans[tag] = mean(recalls[tag])
        recallSDs[tag] = std(recalls[tag])
        fScoreMeans[tag] = mean(f_scores[tag])
        fScoreSDs[tag] = std(f_scores[tag])
   
    if csvWriter:
        for tag in precisionMeans.keys():
            if spacedRows:
                csvWriter.writerow(["{} - {}".format(rowLabel, tag), precisionMeans[tag],precisionSDs[tag],"","", recallMeans[tag], recallSDs[tag],"","", fScoreMeans[tag], fScoreSDs[tag], "",""])
            else:
                csvWriter.writerow(["{} - {}".format(rowLabel, tag), precisionMeans[tag],precisionSDs[tag], recallMeans[tag], recallSDs[tag],fScoreMeans[tag], fScoreSDs[tag]])
    else:
        print "{}-Fold Cross-validated".format(folds)
        print "Precision: {} (sd={})".format(precisionMeans,precisionSDs)
        print "Recall: {} (sd={})".format(recallMeans, recallSDs)
        print "F-Score: {} (sd={})".format(fScoreMeans, fScoreSDs)
    
    return (precisions, recalls, f_scores)

def filterLowOccurancesByN(featuresByN, ngrams=[1,2], minimumCount=3):
    docCounts = {}
    for docID in featuresByN: #for each document
        features, tag = featuresByN[docID]
        for n in ngrams:
            for feature in features[n]: #for each feature in the document
                count = 0
                docs = []
                if feature in  docCounts:
                    count, docs = docCounts[feature]
                count = count + 1 #increment the count
                docs.append(features[n]) #note which document it appears in
                docCounts[feature] = (count, docs)
    
    for feature in docCounts: #for each feature in the corpus
        count, docs = docCounts[feature]
        if count < minimumCount: #if the feature doesn't occur in enough documents
            for doc in docs: #for each document it occures in
                doc.pop(feature, None)
    
    return featuresByN

def filterLowOccurances(featuresByN, minimumCount=3):
    docCounts = {}
    for docID in featuresByN: #for each document
        features, tag = featuresByN[docID]
        for feature in features: #for each feature in the document
            count = 0
            docs = []
            if feature in  docCounts:
                count, docs = docCounts[feature]
            count = count + 1 #increment the count
            docs.append(features) #note which document it appears in
            docCounts[feature] = (count, docs)
    
    for feature in docCounts: #for each feature in the corpus
        count, docs = docCounts[feature]
        if count < minimumCount: #if the feature doesn't occur in enough documents
            for doc in docs: #for each document it occures in
                doc.pop(feature, None)
    
    return featuresByN

def lengthNormalize(featuresByN, ngrams=[1,2]):
    for n in ngrams:
        for docID in featuresByN:
            features, tag = featuresByN[docID]
            
            docSqSum = 0
            for count in features[n].values(): #for each ngram feature
                docSqSum = pow(count, 2) #sum the squares
            docLength = sqrt(docSqSum) #get the square root
            
            for feature in features[n]:
                features[n][feature] = float(features[n][feature])/docLength
    
    return featuresByN