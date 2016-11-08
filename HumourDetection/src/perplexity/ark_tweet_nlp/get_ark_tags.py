'''
Created on Oct 26, 2016

@author: Andrew
'''
from pymongo import  MongoClient
import re
from CMUTweetTagger import runtagger_parse

if __name__ == '__main__':
    client = MongoClient()
    collections = ["GentlerSongs", "OlympicSongs", "BoringBlockbusters", "OceanMovies"]
    likes="total likes"
    minLikes=7
    punch_words = "punch words"
    text = "text"
    atMentions = re.compile(ur"@\w+", flags=re.I|re.U)
    atMidnight = re.compile(u"@midnight", flags=re.I|re.U)
    hashtag = re.compile(ur"#\w+", flags=re.I|re.U)
    whitespace = re.compile(ur"\s+", flags=re.U)
    
    tweet_texts = []
    col_and_ids = []
    for col in collections:
        for tweet in client.tweets[col].find({likes: {"$gte" : minLikes}}):
            #only looking at annotated tweets
            if punch_words not in tweet:
                continue
            if (tweet[punch_words] == None) or (tweet[punch_words] == []):
                continue
            for word in tweet[punch_words]:
                if word == "None":
                    continue
                if not word:
                    continue
            
            tweet_text = tweet[text]
            #filter out extra mentions or extra hashtags
            mentions = atMentions.findall(tweet_text)
            if len(mentions) > 1: #if more than 1 person is mentioned
                continue
            elif len(mentions) == 1:
                if not atMidnight.match(mentions[0]): #if the mention someone other than @midnight
                    continue
            if len(hashtag.findall(tweet_text)) > 1: #if there's more than 1 hashtag
                continue
            
            
#             tagged_tweets = runtagger_parse([tweet_text])
#             if len(tagged_tweets) > 1:
#                 print tweet_text
#                 raise Exception("greater than 1")
#             
#             tokens=[]
#             tags=[]
#             for token, tag, _ in tagged_tweets[0]:
#                 tokens.append(token)
#                 tags.append(tag)
#             token_str = " ".join(tokens)
#             tag_str = " ".join(tags)
#             client.tweets[col].update_one({"_id" : tweet["_id"]}, {"$set" : {"ark token" : token_str, "ark pos" : tag_str}})
            
            
            tweet_text = whitespace.sub(" ", tweet_text)
            tweet_texts.append(tweet_text)
            col_and_ids.append((col, tweet["_id"]))
     
    tagged_tweets = runtagger_parse(tweet_texts)
     
    print len(tweet_texts)
    print len(col_and_ids)
    print len(tagged_tweets)
     
    for i in xrange(len(tagged_tweets)):
        tweet_tags = tagged_tweets[i]
        col, _id = col_and_ids[i]
           
        tokens = []
        tags = []
        for token, tag, prob in tweet_tags:
            tokens.append(token)
            tags.append(tag)
           
        token_str = " ".join(tokens)
        tag_str = " ".join(tags)
           
        client.tweets[col].update_one({"_id" : _id}, {"$set" : {"ark token" : token_str, "ark pos" : tag_str}})