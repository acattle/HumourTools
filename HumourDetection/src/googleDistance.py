'''
Created on May 26, 2016

@author: Andrew
'''
from selenium import webdriver
import re
from selenium.common.exceptions import NoSuchElementException
from time import sleep, strftime
from math import log
from urllib import quote
# from pymongo import MongoClient
from numpy import mean
import os
import glob
import codecs
from nltk.corpus import stopwords
from multiprocessing.pool import Pool

def getResultCount(query, driver=None):
    def get_count_from_page(driver, query_url):
        if driver == None:
            driver = webdriver.Chrome("C:/chromedriver.exe")
        driver.get(queryURL)
        sleep(2)
        
        resultStats = driver.find_element_by_id("resultStats").text
        countStr = re.search(r"[\d,]+", resultStats).group(0)
        countStr = countStr.replace(",", "")
        resultCount = float(countStr)
        return resultCount
        
    query = quote(query.encode("utf-8")) #needed for googling hashtags
    #encoding because urllib can't handle unicode. google will fix the decoding issue for us.
    queryURL = u"http://google.com/#q={}".format(query)
    
    resultCount = 0
    tries = 0
    change_url = 5
    max_tries = 10
    try:
        while True:
            try:
                if driver == None:
                    driver = webdriver.Chrome("C:/chromedriver.exe")
                driver.get(queryURL)
                sleep(2)
            
                resultStats = driver.find_element_by_id("resultStats").text
                countStr = re.search(r"[\d,]+", resultStats).group(0)
                countStr = countStr.replace(",", "")
                resultCount = float(countStr)
                break
            except Exception, e:
                tries+=1
                if tries % change_url == 0: #if we've tried change_url number of times
                    queryURL=u"http://google.com/#q={}&start=10".format(query)
                    #start=10 forces the second page
                    #Searches like "ramsay's shows" were returning a banner with links ot his shows instead of the result count
                    #however, if a search returns no results, start=10 will omit the google suggestions, which we want
                    #so it's a last ditch effort
                if tries > max_tries:
                    raise e
                print "error getting {}. {} retires remaining".format(query, max_tries-tries)
    except NoSuchElementException:
        print "ERROE NoSuchElement for {} after {} tries. Assume it's a true 0".format(query, max_tries)
        resultCount = 0

    return resultCount

def getNormalizedGoogleDistance(x, y, driver=None, x_count=None, y_count=None):
    #https://en.wikipedia.org/wiki/Normalized_Google_distance
    #(max(log f(x), log f(y)) - log f(x,y))/(log N - min(log f(x), log f(y)))
    
    if driver == None:
        driver = webdriver.Chrome("C:/chromedriver.exe")
    
    ngd=None
    try:
        log_N = log(25270000000) #number of results for "the"
        if x_count == None: #if we didn't specify x_count
            x_count = getResultCount(x, driver)
        log_fx = log(x_count)
    #     log_fx = log(204000000)
        if y_count == None:
            y_count = getResultCount(y, driver)
        log_fy = log(y_count)
        xy_count = getResultCount(u"{} {}".format(x,y), driver)
        log_fxy = log(xy_count)
        
        ngd = (max(log_fx, log_fy) - log_fxy)/(log_N - min(log_fx, log_fy))
    except ValueError,e:
        print e
        #one of the counts was probably 0
        #NGD is undefined
        pass
    
    return ngd

seen_words = {}
def get_count_from_cache(word, driver):
    word = word.lower()
    count =None
    if word in seen_words:
        #no need to get it again from the web
        count = seen_words[word]
    else:
        #get it from the web
        count = getResultCount(word, driver)
        #add it to seen words
        seen_words[word]=count
    return count
seen_comb = {}
def get_ngd_from_cache(word1, word2,driver):
    word1=word1.lower()
    word2=word2.lower()
    ngd = None
    comb = u" ".join(sorted([word1, word2])) #sorted for consistent word order
    if comb in seen_comb:
        #no need to get it again
        ngd = seen_comb[comb]
    else:
        #get it from the web
        ngd = getNormalizedGoogleDistance(word1, word2, driver, get_count_from_cache(word1,driver), get_count_from_cache(word2,driver))
        seen_comb[comb] = ngd
    
    return ngd

english_stopwords = stopwords.words("english")
pos_to_ignore = ["D","P","X","Y", "T", "&", "~", ",", "!", "U", "E"]
def process_file(filename):
    driver=None
    try:
        driver = webdriver.Firefox(executable_path="C:/geckodriver.exe")
#         driver = webdriver.Chrome("C:/chromedriver.exe")
        sleep(3)
        
        name = os.path.splitext(os.path.basename(filename))[0]
        hashtag = "#{}".format(re.sub("_", "", name.lower()))
        hashtag_words = name.split("_")        
        #remove swords that don't give you some idea of the domain
        hashtag_words = [word.lower() for word in hashtag_words if word.lower() not in english_stopwords]
        #the next 3 are to catch "<blank>In#Words" type hashtags
        hashtag_words = [word for word in hashtag_words if word != "in"]
        hashtag_words = [word for word in hashtag_words if not ((len(word) == 1) and (word.isdigit()))]
        hashtag_words = [word for word in hashtag_words if word != "words"]
        
        print "{}\tprocessing {}".format(strftime("%y-%m-%d_%H:%M:%S"),name)
        tweet_ids = []
        tweet_tokens = []
        tweet_pos = []
        with codecs.open(filename, "r", encoding="utf-8") as tweet_file:
            for line in tweet_file:
                line=line.strip()
                if line == "":
                    continue
                line_split = line.split("\t")
                tweet_tokens.append(line_split[0].split())
                tweet_pos.append(line_split[1].split())
                tweet_ids.append(line_split[3])
        
        already_collected = set()
        lines_to_rewrite = []
        #check if file exists to avoid redoing a lot of effort
        ngd_fileloc = "{}.ngd".format(filename)
        if os.path.isfile(ngd_fileloc): #if the file exists
            with codecs.open(ngd_fileloc, "r", encoding="utf-8") as resume_file:
                header = resume_file.readline().strip()
                if header.split() == hashtag_words: #only if the header matches what we've extracted
                    for line in resume_file:
                        line_split = line.split("\t")
                        if len(line_split) != (len(hashtag_words) +2): #if we don't have enough columns
                            print u"ERROR - previously collected tweet is incomplet: {}".format(line)
                            continue
                        
                        tweet_id = line_split[0]
                        max_val = line_split[1].split()[2]
                        
                        if max_val == "inf":
                            print u"Tweet {} has an inf value. Will retry".format(tweet_id)
                            continue
                        
                        already_collected.add(tweet_id)
                        lines_to_rewrite.append(line)
        
        done = 0
        with codecs.open(ngd_fileloc, "w", encoding="utf-8") as out_file:
            out_file.write(u"{}\n".format(u" ".join(hashtag_words)))
            for line in lines_to_rewrite:
                out_file.write(line)
            
            for tokens, pos, tweet_id in zip(tweet_tokens,tweet_pos, tweet_ids):
                if tweet_id in already_collected: #skip it if we already have a valid reading
                    done+=1
                    continue
                
                ngd_results_by_word = []
                for word in hashtag_words:
                    ngds_by_hashtag_word=[]
                    for token, tag in zip(tokens, pos):
                        token=token.lower()
                        if (tag in pos_to_ignore) or (token in english_stopwords):
                            continue
                        if (token == "@midnight") or (token == hashtag): #if it's the @midnight account of the game's hashtag
                                continue #we don't want to process it
                        
                        ngd = get_ngd_from_cache(word, token, driver)
                        if ngd == None:
                            ngd = float("inf")
                        ngds_by_hashtag_word.append(ngd)
                        
                    if len(ngds_by_hashtag_word) == 0:
                        print u"ERRORL no valid tokens\t{}".format(u" ".join(tokens))
                        ngds_by_hashtag_word = [float("inf")]
                    
                    
                    ngd_results_by_word.append((min(ngds_by_hashtag_word), mean(ngds_by_hashtag_word), max(ngds_by_hashtag_word)))
                
                mins, avgs, maxes = zip(*ngd_results_by_word) #separate out the columns
                
                overall = (min(mins), mean(avgs), max(maxes))
            
                per_word_ngds = u"\t".join([u"{} {} {}".format(*res) for res in ngd_results_by_word])
                overall_ngds = u"{} {} {}".format(*overall)
                line = u"{}\t{}\t{}\n".format(tweet_id, overall_ngds, per_word_ngds) 
                out_file.write(line)
                done+=1
                if done % 20 == 0:
                    print "{}\t{}\t{} of {} completed".format(strftime("%y-%m-%d_%H:%M:%S"), name, done, len(tweet_ids))
        print "{}\tfinished {}".format(strftime("%y-%m-%d_%H:%M:%S"),name)
        return True
    finally:
        if driver != None:
            driver.quit()
    
if __name__ == '__main__':
#     client = MongoClient()
#     driver = webdriver.Chrome("C:/chromedriver.exe")
#     driver = webdriver.Firefox()
#     sleep(3)
    
#     cols = [("GentlerSongs", "gentle"), ("OlympicSongs", "olympic"), ("BoringBlockbusters", "boring"), ("OceanMovies", "ocean")]
#     for col, setup in cols:
#         setup_count = getResultCount(setup, driver)
#         
#         tweets = client.tweets[col].find({"$and" : [{"total likes" : {"$gte" : 7}}, {"punch words" : {"$exists" : True}}, {"ngd closest" : {"$exists" : False}}]})
#     
#         count = 0
#         total = tweets.count()
#         #https://stackoverflow.com/questions/24199729/pymongo-errors-cursornotfound-cursor-id-not-valid-at-server
#         tweets.batch_size(10) # It's local host so it doesn't really matter if we do a ton of requests, I just don't want the cursor to time out
#         for tweet in tweets:
#             furthest = (-float("inf"), "")
#             closest = (float("inf"), "")
#             allVals = []
#             
#             if tweet["punch words"]:
#                 for word in tweet["punch words"]:
#                     if word == "None":
#                         continue
#                     if not word:
#                         continue
#                     ngd = getNormalizedGoogleDistance(setup, word, driver, setup_count)
#                     if ngd == None: #if ngd is undefined
#                         continue
#                     if ngd > furthest[0]:
#                         furthest = [ngd, word]
#                     if ngd < closest[0]:
#                         closest = [ngd, word]
#                     allVals.append(ngd)
#                 
#                 if len(allVals) > 0: #if at least one word has a defined NGD
#                     tweet["ngd closest"] = closest[0]
#                     tweet["ngd closest word"] = closest[1]
#                     tweet["ngd furthest"] = furthest[0]
#                     tweet["ngd furthest word"] = furthest[1]
#                     tweet["ngd average"] = mean(allVals)
#                     client.tweets[col].update({"_id" : tweet["_id"]}, tweet)
#             
#             else:
#                 pass
#             
#             count += 1
#             print "{} of {} done".format(count, total)

#     setup_count = getResultCount("OLYMPIC", driver)
#     print "NGD(ROW) = {}".format(getNormalizedGoogleDistance("OLYMPIC", "ROW", driver, setup_count))
#     print "NGD(SAIL) = {}".format(getNormalizedGoogleDistance("OLYMPIC", "SAIL", driver, setup_count))
#         
#     driver.close()


    semeval_dir = r"C:/Users/Andrew/Desktop/SemEval Data"
    dirs = [r"trial_dir/trial_data",
            r"train_dir/train_data",
            r"evaluation_dir/evaluation_data"]
#     dirs = ["train_dir/train_data"]
    tagged_dir = "tagged"
     
    filenames = []
    for d in dirs:
        os.chdir(os.path.join(semeval_dir, d, tagged_dir))
        for f in glob.glob("*.tsv"):
            filenames.append(os.path.join(semeval_dir, d, tagged_dir,f))
    
#     process_file(os.path.join(semeval_dir,r"trial_dir\trial_data",tagged_dir,"Fast_Food_Books.tsv" ))
    p=Pool(8)
    res = p.map(process_file, filenames)
     
    for r, f in zip(res,filenames):
        print "{} {}".format(f,r)
    
    