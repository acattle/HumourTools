'''
Created on Sep 18, 2016

@author: Andrew Cattle <acattle@cse.ust.hk>
'''
import subprocess
from tempfile import TemporaryDirectory
import os
import re

class KenLMSubprocessWrapper():
    """
        Wrapper class for using KenLM as a Python subprocess.
        
        This is useful if you're using a Windows development environment since
        KenLM doesn't like compiling on Windows but works just fine with
        Windows Subsystem for Linux (WSL).
        
        For more details on WSL, see:
        https://docs.microsoft.com/en-us/windows/wsl/install-win10
    """
    
    def __init__(self, model_Loc, kenlm_query_loc):
        """
            Initialize a KenLMSubprocessWrapper
            
            :param model_loc: the location of the model which will be queried (WSL format, i.e. '/mnt/c/...')
            :type model_loc: str
            :param kenlm_query_loc: location of the KenLM query binary (WSL format, i.e. '/mnt/c/...')
            :type kenlm_query_loc: str
        """
        self.model_loc = model_Loc
        self.kenlm_query_loc = kenlm_query_loc

    def get_probabilities(self, documents, tmpdir=None):
        """
            Query multiple documents at the same time and extract their log
            probabilties. (Note that KenLM uses log base 10).
            
            Since KenLM Query is called as a subprocess, the LM is read from
            disk for each call. Therefore, querying multiple documents in a
            single call is much more efficient.
            
            Note that the contents of documents as well as the output of KenLM
            Query will be written to temporary files on disk. If documents is
            very large, and/or your OS's default temporary directory is on an
            SSD, dir can be used to specify temp files to be written on a
            larger, possible hard-disk drive.
            
            :param documents: tokenized documents to query
            :type documents: Iterable[Iterable[str]]
            :param tmpdir: the directory temporary files will be created in. If None, your OS's default temp directory will be used
            :type tmpdir: str
            
            :returns: A list containing the log probabilities of each document
            :rtype: List[float]
        """
        #TODO: accept a file as input, so we can skip writing documents to disk
        
        #This command will invoke KenLM's query binary with the specified model using Windows Subsystem for Linux
        query_str = f"wsl {self.kenlm_query_loc} {self.model_loc}"
        
        #It would be more straight forward to just use tempfile.TemporaryNamedFiles
        #but Windows won't let files to be open more than once (i.e. you can't read and write a file at the same time).
        #We could use NamedTemporaryFile(delete=False) and open the file twice sequentially (once to write, once to read)
        #but then we'd need to manually clean up the temporary files, removing the advantage of using tempfile in the first place.
        #By using TemporaryDirectory, any files we create in that tempory directory will be deleted along with it
        #BUT this means we need to make sure any files in the directory are closed before tempfile tries to delete it
        with TemporaryDirectory(dir=tmpdir) as td:
            doc_loc = os.path.join(td, "documents")
            out_loc = os.path.join(td, "kenlm_query_output")
            
            with open(doc_loc, "w") as df:
                for doc in documents:
                    df.write(" ".join(doc))
                    df.write("\n")
            
            with open(doc_loc, "r") as df, open(out_loc, "w") as out_f:
                #TODO: will this properly report errors?
                subprocess.run(query_str, stdin=df, stdout=out_f)
            
            with open(out_loc, "r") as out_f:
                log_probs = self.process_query_output(out_f)
            
        return log_probs
        
    def process_query_output(self, query_output_f):
        """
            A method for reading KenLM Query outputs and extracting the
            document-level log probability.
            
            Note: KenLM uses log base 10
            
            :param query_output_f: the file containing the KenLM Query output to be processed
            :type query_output_f: file
            
            :returns: A list containing the log probabilities extracted from the Query output
            :rtype: List[float]
        """
        
        total_p=re.compile(r"Total: (-?\d+\.?\d*)")
        
        log_probs = []
        for line in query_output_f:
            total_match = total_p.search(line)
            
            if total_match:
                log_probs.append(float(total_match.group(1)))
        
        return log_probs
    
    def get_perplexity(self,documents, tmpdir=None):
        """
            Calculate the perplexity of multiple documents at the same time.
            
            This method calls KenLM Query as a subprocess. Since this means the
            LM is read from disk for each call, getting perplexities for
            multiple documents in a single call is much more efficient.
            
            Note that the contents of documents as well as the output of KenLM
            Query will be written to temporary files on disk. If documents is
            very large, and/or your OS's default temporary directory is on an
            SSD, dir can be used to specify temp files to be written on a
            larger, possible hard-disk drive.
            
            :param documents: tokenized documents to query
            :type documents: Iterable[Iterable[str]]
            :param tmpdir: the directory temporary files will be created in. If None, your OS's default temp directory will be used
            :type tmpdir: str
            
            :returns: A list containing the perplexities of each document
            :rtype: List[float]
        """
        log_probs = self.get_probabilities(documents, tmpdir)
        
        perplexities = []
        for d, lp in zip(documents, log_probs):
            #perplexity is the exponent of the negative average per-token log probability of a sample
            #KenLM uses log base 10. KenLM also appends </s> on the end of documents
            perplexities.append(10 ** -(lp/(len(d) + 1)))
        
        return perplexities
    
if __name__ == "__main__":
    k = KenLMSubprocessWrapper("/mnt/d/Downloads/brown/brown.bin", "/mnt/d/git/kenlm-stable/build/bin/query")
    
    d=["this is a test .".split(),
       "so is this !".split()]
    
    print(k.get_perplexity(d, r"D:\temp"))
#     km = KenLMModel("/mnt/c/Users/Andrew/Desktop/kenlm models and raw text/wiki_pos_w_punc_4gram_prune.arpa")
#     km = KenLMModel("/mnt/c/Users/Andrew/Desktop/Senti140/tokens_2.arpa")
#     
#     client = MongoClient()
#     
#     atMentions = re.compile(ur"@\w+", flags=re.I|re.U)
#     atMidnight = re.compile(u"@midnight", flags=re.I|re.U)
#     hashtag = re.compile(ur"#\w+", flags=re.I|re.U)
#         
#     likes = "total likes"
#     feature = "ark token"
#     target = "ark perplexity 2 no user or hash"
#     cols = ["GentlerSongs", "OlympicSongs", "OceanMovies", "BoringBlockbusters"]
#     
#     for col in cols:
#         tweets = []
#         for tweet in client.tweets[col].find({"$and":[{likes : {"$gte" : 7}}, {feature : {"$exists" : True}}]}):
#             mentions = atMentions.findall(tweet["text"])
#             if len(mentions) > 1: #if more than 1 person is mentione
#                 continue
#             elif len(mentions) == 1:
#                 if not atMidnight.match(mentions[0]): #if the mention someone other than @midngiht
#                     continue
#             if len(hashtag.findall(tweet["text"])) > 1: #if there's more than 1 hashtag
#                 continue
#             
#             pos_tags = tweet["ark pos"].split()
#             tokens = tweet[feature].split()
#             
#             for i in xrange(len(pos_tags)-1, -1, -1):
#                 if pos_tags[i] in ["#", "@", "U"]:
#                     del tokens[i]
#             
#             tweet["text"] = "".join(tokens)
#                     
#             
#             tweets.append(tweet)
#             
#             
#         
#         for tweet in tweets:
#             perplex = km.perplexity(tweet[feature])
#             tweet[target] = perplex
#             client.tweets[col].update({"_id" : tweet["_id"]}, {"$set" : {target : tweet[target]}})
#             