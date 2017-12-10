'''
Created on Sep 19, 2016

@author: Andrew
'''
import re
import pymongo

atMentions = re.compile(ur"@\w+", flags=re.I|re.U)
atMidnight = re.compile(u"@midnight", flags=re.I|re.U)
hashtag = re.compile(ur"#\w+", flags=re.I|re.U)

c = pymongo.MongoClient()

cols = ["GentlerSongs", "OlympicSongs", "OceanMovies", "BoringBlockbusters"]
for col in cols:
    tweets = []
    tweets7 = []
    tweets4 = []
    tweetsAn = []
    for tweet in c.tweets[col].find():
        mentions = atMentions.findall(tweet["text"])
        if len(mentions) > 1:
            continue
        elif len(mentions) == 1:
            if not atMidnight.match(mentions[0]):
                continue
        if len(hashtag.findall(tweet["text"])) > 1:
            continue
        tweets.append(tweet)
        if (tweet["total likes"] >= 7):
            tweets7.append(tweet)
        if (tweet["total likes"] >= 7) or (tweet["retweets"] >= 4) or (tweet["favorites"] >= 4):
            tweets4.append(tweet)
            if "punch words" in tweet:
                tweetsAn.append(tweet)
    print col
    print len(tweets)
    print len(tweets4)
    print len(tweets7)
    print len(tweetsAn)
    print "\n"