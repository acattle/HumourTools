'''
Created on Oct 25, 2016

@author: Andrew
'''
import re
import codecs
from unicodeCSV import UnicodeReader
import csv
from numpy.core.defchararray import index

"""=============== PATTERNS FOR FINDING COMMON TWEET ELEMENTS ==============="""
toUserPattern = re.compile(r"@\w+", flags=re.U|re.I)
urlPattern = re.compile(r"https?://\S+", flags=re.U|re.I)

no_query = "NO_QUERY"

if __name__ == '__main__':
#     rawLoc = "C:\\Users\\Andrew\\Desktop\\Senti140\\training.1600000.processed.noemoticon.csv"
#     tweetsOnlydLoc = "C:\\Users\\Andrew\\Desktop\\Senti140\\tweets_not_escaped.txt"
    taggedLoc = "C:\\Users\\Andrew\\Desktop\\Senti140\\tweets_tokenized_and_pos.txt"
    posOutLoc = "C:\\Users\\Andrew\\Desktop\\Senti140\\pos_tags.txt"
    tokenOutLoc = "C:\\Users\\Andrew\\Desktop\\Senti140\\tokens.txt"
    tokenSanitizedHashtagsOutLoc = "C:\\Users\\Andrew\\Desktop\\Senti140\\tokens_sanitized_hashtags.txt"
    
#     with open(rawLoc, 'r') as csvfile:
#         csvreader = csv.reader(csvfile)#, delimiter=',', quotechar='|')
#         
#         with open(tweetsOnlydLoc, "w") as outFile:
#             for row in csvreader:
#                 sanitized_text = row[5]
# #                 sanitized_text = toUserPattern.sub("AT_USER", sanitized_text)
# #                 sanitized_text = urlPattern.sub("HYPERLINK", sanitized_text)
#                 
#                 outFile.write("{}\n".format(sanitized_text))
    pos_tags = []
    sanitized_texts = []
    sanitized_texts_no_hashtags = []
    with open(taggedLoc, "r") as taggedFile:
        for line in taggedFile:
            if line.strip():
                tokens_str, tags_str, _, _ = line.split("\t") #ark output is tab delimited. Tokens \t POS tags \t probabilities \t original text
                  
                pos_tags.append(tags_str)
                  
                tags = tags_str.split()
                tokens = tokens_str.split()
                  
                sanitized_tokens = []
                sanitized_tokens_no_hashtags = []
                for i in xrange(len(tags)):
                    tag = tags[i]
                    token = tokens[i]
                      
                    if tag == "U":
                        token = "HYPER_LINK"
                    elif tag == "@": #if we have an at reply at this index
                        token = "AT_USERNAME"
                      
                    sanitized_tokens.append(token)
                          
                    if tag == "#":
                        sanitized_tokens_no_hashtags.append("HASH_HASHTAG")
                    else:
                        sanitized_tokens_no_hashtags.append(token)
                  
                sanitized_texts.append(" ".join(sanitized_tokens))
                sanitized_texts_no_hashtags.append(" ".join(sanitized_tokens_no_hashtags))
      
    with open(posOutLoc, "w") as posOutFile:
        pos_str = "\n".join(pos_tags)
        posOutFile.write(pos_str)
      
    with open(tokenOutLoc, "w") as tokenOutFile:
        token_str = "\n".join(sanitized_texts)
        tokenOutFile.write(token_str)
          
    with open(tokenSanitizedHashtagsOutLoc, "w") as tokenNoHashOutFile:
        token_no_at_no_hash_str = "\n".join(sanitized_texts_no_hashtags)
        tokenNoHashOutFile.write(token_no_at_no_hash_str)