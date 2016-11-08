'''
Created on May 26, 2016

@author: Andrew
'''
from selenium import webdriver
import re
from selenium.common.exceptions import NoSuchElementException
from time import sleep
from math import log
from pymongo import MongoClient
from numpy import mean

def getResultCount(query, driver=None):
    queryURL = "http://google.com/#q={}".format(query)
    
    if driver == None:
        driver = webdriver.Chrome("C:/chromedriver.exe")
    driver.get(queryURL)
    sleep(2)

    resultCount = 0
    try:
        resultStats = driver.find_element_by_id("resultStats").text
    
        countStr = re.search(r"[\d,]+", resultStats).group(0)
        countStr = countStr.replace(",", "")
        resultCount = float(countStr)
    except NoSuchElementException:
        resultCount = 0
    
    return resultCount

def getNormalizedGoogleDistance(x, y, driver=None, x_count=None):
    #https://en.wikipedia.org/wiki/Normalized_Google_distance
    #(max(log f(x), log f(y)) - log f(x,y))/(log N - min(log f(x), log f(y)))
    
    if driver == None:
        driver = webdriver.Chrome("C:/chromedriver.exe")
    
    ngd=None
    try:
        log_N = log(25270000000) #number of results for "the"
        if not x_count: #if we didn't specify x_count
            x_count = getResultCount(x, driver)
        log_fx = log(x_count)
    #     log_fx = log(204000000)
        y_count = getResultCount(y, driver)
        log_fy = log(y_count)
        xy_count = getResultCount(u"{} {}".format(x,y), driver)
        log_fxy = log(xy_count)
        
        ngd = (max(log_fx, log_fy) - log_fxy)/(log_N - min(log_fx, log_fy))
    except ValueError:
        #one of the counts was probably 0
        #NGD is undefined
        pass
    
    return ngd
    
    
if __name__ == '__main__':
    client = MongoClient()
    driver = webdriver.Chrome("C:/chromedriver.exe")
    
    cols = [("GentlerSongs", "gentle"), ("OlympicSongs", "olympic"), ("BoringBlockbusters", "boring"), ("OceanMovies", "ocean")]
    for col, setup in cols:
        setup_count = getResultCount(setup, driver)
        
        tweets = client.tweets[col].find({"$and" : [{"total likes" : {"$gte" : 7}}, {"punch words" : {"$exists" : True}}, {"ngd closest" : {"$exists" : False}}]})
    
        count = 0
        total = tweets.count()
        #https://stackoverflow.com/questions/24199729/pymongo-errors-cursornotfound-cursor-id-not-valid-at-server
        tweets.batch_size(10) # It's local host so it doesn't really matter if we do a ton of requests, I just don't want the cursor to time out
        for tweet in tweets:
            furthest = (-float("inf"), "")
            closest = (float("inf"), "")
            allVals = []
            
            if tweet["punch words"]:
                for word in tweet["punch words"]:
                    if word == "None":
                        continue
                    if not word:
                        continue
                    ngd = getNormalizedGoogleDistance(setup, word, driver, setup_count)
                    if ngd == None: #if ngd is undefined
                        continue
                    if ngd > furthest[0]:
                        furthest = [ngd, word]
                    if ngd < closest[0]:
                        closest = [ngd, word]
                    allVals.append(ngd)
                
                if len(allVals) > 0: #if at least one word has a defined NGD
                    tweet["ngd closest"] = closest[0]
                    tweet["ngd closest word"] = closest[1]
                    tweet["ngd furthest"] = furthest[0]
                    tweet["ngd furthest word"] = furthest[1]
                    tweet["ngd average"] = mean(allVals)
                    client.tweets[col].update({"_id" : tweet["_id"]}, tweet)
            
            else:
                pass
            
            count += 1
            print "{} of {} done".format(count, total)
        
    driver.close()