'''
Created on Jun 10, 2016

@author: Andrew
'''
from pymongo import MongoClient
from matplotlib import pylab, style, rcParams, pyplot, mlab
import pandas
import numpy as np
import re
import math

style.use("ggplot")
pylab.ion()

client = MongoClient()
atMentions = re.compile(ur"@\w+", flags=re.I|re.U)
atMidnight = re.compile(u"@midnight", flags=re.I|re.U)
hashtag = re.compile(ur"#\w+", flags=re.I|re.U)


# tweets = client.tweets.GentlerSongs.find({"$and" : [{"ngd" : {"$exists" : True}}, {"$and" : [{"w2v" : {"$gt" : 0}}, {"w2v" : {"$lt" : 1}}]}]})
# tweets = client.tweets.GentlerSongs.find({"$and" : [{"$and" : [{"usf fwa backward" : {"$gt" : 0}}, {"usf fwa backward" : {"$lt" : 0.01}}]}, {"$and" : [{"w2v" : {"$gt" : 0}}, {"w2v" : {"$lt" : 1}}]}]})
# tweets = client.tweets.GentlerSongs.find({"$and" : [{"usf fwa forward" : {"$gt" : 0}}, {"$or" : [{"favorites" : {"$gt" : 0}}, {"retweets" : {"$gt" : 0}}]}]})
# tweets = client.tweets.GentlerSongs.find({"$and" : [{"$and" : [{"w2v" : {"$exists" : True}}, {"w2v" : {"$gt" : 0}}]}, {"$or" : [{"favorites" : {"$gt" : 0}}, {"retweets" : {"$gt" : 0}}]}]})
# tweets = client.tweets.GentlerSongs.find({"$and" : [{"$and" : [{"usf fwa backward" : {"$gt" : 0}}, {"usf fwa backward" : {"$lt" : 1}}]}, {"$or" : [{"favorites" : {"$gt" : 0}}, {"retweets" : {"$gt" : 0}}]}]})
# features = ["usf fwa forward", "usf fwa backward", "ngd", "w2v", "perplexity 2", "perplexity 3", "perplexity 4", "pos perplexity 2", "pos perplexity 3", "pos perplexity 4"]
# feature = "usf fwa difference least"
# feature = "usf fwa backward average"
feature = "ngd furthest"
tweets = []
cols = ["GentlerSongs", "OlympicSongs", "OceanMovies", "BoringBlockbusters"]
for col in cols:
    for tweet in client.tweets[col].find({"$and" : [{"total likes" : {"$gte" : 7}}, {feature : {"$exists" : True}}]}):
        mentions = atMentions.findall(tweet["text"])
        if len(mentions) > 1: #if more than 1 person is mentione
            continue
        elif len(mentions) == 1:
            if not atMidnight.match(mentions[0]): #if the mention someone other than @midngiht
                continue
        if len(hashtag.findall(tweet["text"])) > 1: #if there's more than 1 hashtag
            continue
        if tweet[feature] > 0:
            tweets.append(tweet)
    
df = pandas.DataFrame(tweets)

# df['likes_quartiles'] = pandas.qcut(df['total likes'], 4, labels=['0-25%', '25-50%', '50-75%', '75-100%'])
# df.boxplot(column='ngd', by='likes_quartiles')

# _, breaks=np.histogram(df["ngd"],bins=5)
# df['Class']=(df["ngd"].values>breaks[..., np.newaxis]).sum(0)
# ax=df.boxplot(column='total likes',by='Class')
# ax.xaxis.set_ticklabels(['%s'%val for i, val in enumerate(breaks) if i in df.Class])

# df.plot("usf fwa backward", "w2v", kind="scatter")
# df.plot(feature, "total likes", kind="scatter")

# pyplot.figure(figsize=(float(8000)/96, float(6100)/96), dpi=96)
df = df[np.abs(df[feature]-df[feature].mean())<=(3*df[feature].std())] #exclude FWA outliers
df = df[np.abs(df["total likes"]-df["total likes"].mean())<=(3*df["total likes"].std())] #exclude total likes outliers
p=df.plot(feature, "total likes", kind="scatter", figsize=(float(1600)/100, float(900)/100), s=40)

# textkwargs = {"size" : "large"}
rcParams.update({'font.size': 15})
p.set_xlabel("$NGD$", size = 25)
# p.set_xlabel("$FWA_{backward}$", size = 25)
# p.set_xlabel("$FWA_{difference}$", size = 25)
p.set_ylabel("Total Likes", size = 25)
# p.set_title("$FWA_{backward}$ versus Total Likes", size=35, y=1.01)
# p.set_title("$FWA_{difference}$ versus Total Likes", size=35, y=1.01)
p.set_title("$NGD$ versus Total Likes", size=35, y=1.01)
p.set_ylim(bottom=0)
# p.set_xlim(left=-0.005, right=0.25000001)
# p.set_xlim(left=-0.150000001, right=0.10000001)
# pyplot.tight_layout()

# groups = df.groupby(pandas.cut(df["usf fwa forward"], 15))
# groups.mean().plot("usf fwa forward", "total likes")
# groups = df.groupby(pandas.cut(df["w2v"], 15))
# groups.mean().plot("usf fwa forward", "total likes")

mu = 0.9
variance = 0.2
sigma = math.sqrt(variance)
x = np.linspace(0, 1.8, 100)
p.plot(x,mlab.normpdf(x, mu, sigma)*185, color="#C05A4D", lw=2)

# p.plot([0, 0.21], [205, 5], "#C05A4D", lw=2)
# p.plot([-0.12, 0], [5, 210], "#C05A4D", lw=2)

p.set_xlabel("NGD", size = 'x-large')
df["likes/retweets"].value_counts().plot(kind="bar")

# Bin the data frame by "a" with 10 bins...
bins = np.linspace(0, df.ngd.max(), 16)
groups = df.groupby(np.digitize(df.ngd, bins))

groups["total likes"].mean().plot()
pylab.show()

# Get the mean of each bin:
pylab.plot(list(bins), list(groups["total likes"].mean())) # Also could do "groups.aggregate(np.mean)"

# Similarly, the median:
print groups.median()

# Apply some arbitrary function to aggregate binned data
print groups.aggregate(lambda x: np.mean(x[x > 0.5]))



