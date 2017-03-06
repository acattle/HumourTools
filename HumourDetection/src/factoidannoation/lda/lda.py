'''
Created on Aug 29, 2016

@author: Andrew

Following tutorial from https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html
'''
import codecs
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora
from gensim.models.ldamodel import LdaModel

class LDATopic:
    def __init__(self, texts):
        '''
        Constructor
        '''
        engStopWords = stopwords.words("english")
        porter = PorterStemmer()
        
        tokenizedTexts = []
        for text in texts:
            tokens = []
            for sentence in sent_tokenize(text):
                for word in word_tokenize(sentence):
                    word = word.lower()
                    if word not in engStopWords:
                        stem = porter.stem(word)
                        tokens.append(stem)
            tokenizedTexts.append(tokens) #add the document to the list of tokenized documents
    
        #create document term matrix
        dictionary = corpora.Dictionary(tokenizedTexts)
        corpus = [dictionary.doc2bow(text) for text in tokenizedTexts]
        
        self.ldamodel = LdaModel(corpus, num_topics=20, id2word = dictionary, passes=20)

if __name__ == "__main__":
#     client = MongoClient()
#     gentlerSongs = client.tweets.GentlerSongs.find()
#      
#     texts = []
#     for tweet in gentlerSongs:
#         text = tweet["text"]
# #         text = re.sub("@midnight", "", text, flags=re.I)
#         text = re.sub(r"@\S*", "", text, flags=re.I)
# #         text = re.sub("#GentlerSongs", "", text, flags=re.I)
#         text = re.sub(r"#\S*", "", text, flags=re.I)
#         texts.append(text)
    texts = []
    with codecs.open("C:\\Users\\Andrew\\Desktop\\radev -caption corpus\\data\\445.data", "r", "utf-8") as hybridCarSubmissions:
        hybridCarSubmissions.readline() #we can ignore the first line
        for line in hybridCarSubmissions:
            caption = line.split("\t")[1] #get the second item in the tab delimited file
            texts.append(caption)
     
    ldaClass = LDATopic(texts)
    print ldaClass
        