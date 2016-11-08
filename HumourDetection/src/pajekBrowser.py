'''
Created on Jun 21, 2016

@author: Andrew
'''
from igraph import Graph, drawing
from math import log, exp
from pymongo import MongoClient
from numpy import mean

def getSign(num):
    sign = float(10) ** -10 #get around later check for 0
    if num < 0:
        sign = -1
    elif num > 0:
        sign = 1
    return sign

class PajekGraph:
    def __init__(self, pajekFile):
        self.graph = Graph.Read_Pajek(pajekFile)
        self.graph.vs["name"] = self.graph.vs["id"] #work around for bug: https://github.com/igraph/python-igraph/issues/86
        self.words = set(self.graph.vs["name"])
        
        #Set the weights to the -log of their forward strength so we can use built-in Dijkstra
        logWeights = []
        for weight in self.graph.es["weight"]:
            logWeights.append(-log(weight))
        self.graph.es["weight"] = logWeights
        
    def getFowardStrength(self, fromWords, toWords, alpha=0.99):
        
        #check that all words are in the graph to avoid errors
        fWords=[]
        for word in fromWords:
            if word.upper() in self.words:
                fWords.append(word.upper())
        tWords=[]
        for word in toWords:
            if word.upper() in self.words:
                tWords.append(word.upper())
        
        if ((not fWords) or (not tWords)):
            #one of the sets is empty
            return None
        
        pathWeights =  self.graph.shortest_paths(fWords, tWords, weights=self.graph.es["weight"]) #returns weight
        
        #TODO: Remove cod efor making subgraphs
        path =  self.graph.get_shortest_paths(fWords[0], tWords, weights=self.graph.es["weight"], output="vpath") #returns path
        subvertices = set()
        for vertex in path[0]:
            subvertices.update(self.graph.neighbors(vertex, mode="OUT"))
#         subvs = self.graph.vs.select(subvertices)
#         subgraph = self.graph.subgraph(subvs)
        subgraph = self.graph.induced_subgraph(subvertices)
        subgraph.vs["label"] = subgraph.vs["name"]
        subgraph.es["label"] = subgraph.es["weight"] #https://stackoverflow.com/questions/21140853/labelling-the-edges-in-a-graph-with-python-igraph
        layout = subgraph.layout("kk")
        drawing.plot(subgraph, layout=layout)
        
        minWeight = float("inf")
        
        
        
        minI = None
        minJ = None
        for i in range(0, len(pathWeights)):
            for j in range(0, len(pathWeights[i])):
                if pathWeights[i][j] < minWeight:
                    minWeight=pathWeights[i][j]
                    minI = i
                    minJ = j
                    
        pathLength=0
        if not((minI == None) or (minJ == None)):
            shortestPath = self.graph.get_shortest_paths(fWords[i], to=tWords[j], weights=self.graph.es["weight"], output="vpath") #returns path
            pathLength = len(shortestPath[0])
        
        #convert back to probabilities
        prob = exp(-minWeight)
        #penalize path length
        if pathLength > 2:
            prob = prob*(alpha ** (pathLength - 2))
         
        return prob
        

if __name__ == "__main__":
    g = PajekGraph("Data/PairsFSG2.net")
    
    client = MongoClient()
    
    
    cols = [("GentlerSongs", "gentle"), ("OlympicSongs", "olympics"), ("BoringBlockbusters", "boring"), ("OceanMovies", "ocean")]
    for col, setup in cols:
#         tweets = client.tweets[col].find({"$and" : [{"punch words" : {"$exists" : True}}, {"usf fwa forward" : {"$exists" : False}}]})
        tweets = client.tweets[col].find({"punch words" : {"$exists" : True}})
        
        count = 0
        total = tweets.count()
        for tweet in tweets:
            mostF = (-float("inf"), "")
            leastF = (float("inf"), "")
            allF = []
            mostB = (-float("inf"), "")
            leastB = (float("inf"), "")
            allB = []
            
#             print (u"{} {}".format(col, tweet["punch words"]))
            if tweet["punch words"]:
                for word in tweet["punch words"]:
                    if word == "None":
                        continue
                    if not word:
                        continue
                    fwaForward = g.getFowardStrength([setup.upper()], [word.upper()])
                    if fwaForward == float("inf"):
                        print "how?"
                    if (fwaForward > 0):
                        if fwaForward > mostF[0]:
                            mostF = (fwaForward, word)
                        if fwaForward < leastF[0]:
                            leastF = (fwaForward, word)
                        allF.append(fwaForward)
                    fwaBackward = g.getFowardStrength([word.upper()], [setup.upper()])
                    if fwaBackward == float("inf"):
                        print "how?"
                    if fwaBackward > 0:
                        if fwaBackward > mostB[0]:
                            mostB = (fwaBackward, word)
                        if fwaBackward < leastB[0]:
                            leastB = (fwaBackward, word)
                        allB.append(fwaBackward)
                
#                 print "{} {}".format(bestF[0], bestB[0])
                if len(allF) > 0: #if at least 1 valid forward
                    tweet["usf fwa forward most"] = mostF[0]
                    tweet["usf fwa forward most word"] = mostF[1]
                    tweet["usf fwa forward least"] = leastF[0]
                    tweet["usf fwa forward least word"] = leastF[1]
                    tweet["usf fwa forward average"] = mean(allF)
                if len(allB) > 0: #if at least 1 valid backward
                    tweet["usf fwa backward most"] = mostB[0]
                    tweet["usf fwa backward most word"] = mostB[1]
                    tweet["usf fwa backward least"] = leastB[0]
                    tweet["usf fwa backward least word"] = leastB[1]
                    tweet["usf fwa backward average"] = mean(allB)
                if (len(allB) > 0) and (len(allF) > 0): #if we have valid readings for both
                    mostDiff = tweet["usf fwa forward most"] - tweet["usf fwa backward most"]
                    leastDiff = tweet["usf fwa forward least"] - tweet["usf fwa backward least"]
                    avgDiff = tweet["usf fwa forward average"] - tweet["usf fwa backward average"]
                    tweet["usf fwa difference most"] = mostDiff
                    tweet["usf fwa difference most sign"] = getSign(mostDiff)
                    tweet["usf fwa difference least"] = leastDiff
                    tweet["usf fwa difference least sign"] = getSign(leastDiff)
                    tweet["usf fwa difference average"] = avgDiff
                    tweet["usf fwa difference average sign"] = getSign(avgDiff)
                client.tweets[col].update({"_id" : tweet["_id"]}, tweet)
            
            else:
                pass
            
            count += 1
            print "{} of {} done".format(count, total)