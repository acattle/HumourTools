'''
Created on May 23, 2016

@author: Andrew
'''
from pymongo import MongoClient
import re

if __name__ == '__main__':
    client = MongoClient()
    cols = [("GentlerSongs", "gentle"), ("OlympicSongs", "olympic"), ("BoringBlockbusters", "boring"), ("OceanMovies", "ocean")]
    
    atMentions = re.compile(ur"@\w+", flags=re.I|re.U)
    atMidnight = re.compile(u"@midnight", flags=re.I|re.U)
    hashtag = re.compile(ur"#\w+", flags=re.I|re.U)
#     link = re.compile(ur"https?://[a-zA-Z0-9/\.\-]+", flags=re.I|re.U)
    
    for col, setup in cols:
#         for l in [3, 5, 7]:
#             count = client.tweets[col].find({"total likes" : {"$gte" : l}}).count()
#             print "{} - {} - {} done".format(col, l, count)
    
#         colHashtag = re.compile(u"#{}".format(col), flags=re.I|re.U)
#         result = client.tweets[col].delete_many({"text": {"$regex" : link}})
#         print u"{} : {}".format(col, result.deleted_count)

#         results = client.tweets[col].find({"$and" : [{"total likes" : {"$gte" : 7 }}, {"punch words" : {"$exists" : False}}]})
        results = client.tweets[col].find({"$and" : [{"total likes" : {"$gte" : 7}}, {"punch words" : {"$ne" : []}}]})
        for tweet in results:
            try:
                tweet["punch words"]
                #if punch word exists we've already annotated this tweet and can skip it
                continue
            except KeyError:
                pass
              
            mentions = atMentions.findall(tweet["text"])
              
            if len(mentions) > 1: #if more than 1 person is mentione
                continue
            elif len(mentions) == 1:
                if not atMidnight.match(mentions[0]): #if the mention someone other than @midngiht
                    continue
              
            if len(hashtag.findall(tweet["text"])) > 1: #if there's more than 1 hashtag
                continue
              
            try:
                print u"\n\n{}\n".format(tweet["text"])
            except UnicodeError:
                continue
              
            punchWordStr = raw_input("punch words: ") #"" means not applicable
               
            punchWords = []
            for punchWord in punchWordStr.split(","):
                punchWords.append(punchWord.strip())
               
            if punchWords == [""]:
                punchWords = []
               
            tweet["setup word"] = setup
            tweet["punch words"] = punchWords
               
            client.tweets[col].update({"_id" : tweet["_id"]}, tweet)