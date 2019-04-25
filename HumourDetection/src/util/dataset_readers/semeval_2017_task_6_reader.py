'''
Created on Apr 25, 2019

@author: Andrew Cattle <acattle@connect.ust.hk>
'''
import codecs
import re
import os
import glob

_midnight = re.compile("@midnight", flags=re.I) #pattern for matching @midnight user mention

def read_semeval_file(file_loc, remove_midnight=True, remove_hashtag=True, tokenize_hashtag=False):
    """
    Reads SemEval 2017 Task 6 data file. Files expected to be in "<tweet id>\\t
    <tweet body>\\t<humour label>" format.
    
    :param file_loc: the location of the file to be read
    :type file_loc: str
    :param remove_midnight: Indicates if @midnight twitter acount should be removed from tweet body
    :type remove_midnight: bool
    :param remove_hashtag: Indicates if relevant Hastag War hashtag (specified by file name) should be removed. Overrides tokenize_hashtag.
    :type remove_hashtag: bool
    :param tokenize_hashtag: Indicates whether Hashtag War hashtag should be tokenized (according to file name). This flag is ignored if remove_hashtag is True
    :type tokenize_hashtag: bool
    
    :returns: A tuple containing the hashtag name and a list of tuples in (<tweet id>, <tweet body>, <label>) format
    :rtype: Tuple[str, List[Tuple[str, str, int]]]
    
    :raises ValueError: If a line is found to not be in '<tweet_id>\\t<tweetbody>\\t<label> format
    """
    
    docs = []
    
    hashtag_name = os.path.splitext(os.path.basename(file_loc))[0] #get the filename without path or extension
    hashtag = re.compile(f"#{hashtag_name.replace('_','')}",re.I) #pattern for matching Hashtag War hashtag
    
    with codecs.open(file_loc, "r", encoding="utf-8") as f:
        for line in f:
            try:
                tweet_id, tweet_body, tweet_label = line.split("\t")
                tweet_label = int(tweet_label)
            except ValueError as e:
                raise ValueError(f"Line '{line}' in file {file_loc} is not in expected '<tweet id>\\t<tweet body>\\t<label>' format. Original exception: {e}")
            
            if remove_midnight:
                tweet_body = _midnight.sub("", tweet_body) #remove @midnight, regardless of case
            
            if remove_hashtag:
                tweet_body = hashtag.sub("", tweet_body) #remove hashtag, regardless of case
            elif tokenize_hashtag:
                tweet_body = hashtag.sub(f"{hashtag_name.replace('_', ' ')}", tweet_body) #replace the Hashtag War hastag with the tokenized version from the filename
            
            docs.append((tweet_id, tweet_body, tweet_label))
    
    return hashtag_name, docs

def read_semeval_directory(data_dir, remove_midnight=True, remove_hashtag=True, tokenize_hashtag=False):
    """
    Reads all data in a directory in SemEval 2017 Task 6 format. Files must be
    tsv (tab separated values) with each file corresponding to one Hashtag War
    hashtag and each line corresponding to one tweet in '<tweet id>\\t<tweet
    body>\\t<label>" format. Hashtag tokenization relies on properly tokenized
    file names.
    
    :param data_dir: the directory containing the tsv files
    :type data_dir: str
    :param remove_midnight: Indicates if @midnight twitter acount should be removed from tweet body
    :type remove_midnight: bool
    :param remove_hashtag: Indicates if relevant Hastag War hashtags (specified by file name) should be removed. Overrides tokenize_hashtag.
    :type remove_hashtag: bool
    :param tokenize_hashtag: Indicates whether Hashtag War hashtags should be tokenized (according to file name). This flag is ignored if remove_hashtag is True
    :type tokenize_hashtag: bool
    
    :returns: A list of tuples in (<tweet id>, <tweet body>, <label>) format
    :rtype: List[Tuple[str, str, int]]
    
    :returns: A dictionary in {<hashtag name> : <list of tuples in (tweet_id, tweet_body, label) format>} format
    :rtype: Dict[str, List[Tuple[str, str, int]]]
    """
    hashtag_wars = {}
    for tsv_fileloc in glob.glob(os.path.join(data_dir, "*.tsv")):
        hashtag_name, tweets = read_semeval_file(tsv_fileloc, remove_midnight, remove_hashtag, tokenize_hashtag)
        hashtag_wars[hashtag_name] = tweets
    
    return hashtag_wars

def read_semeval_2017_task_6_data(semeval_data_root_dir, training_dirs=["trial_dir/trial_data","train_dir/train_data"], evaluation_dir="evaluation_dir/evaluation_data", validation_dir=None, remove_midnight=True, remove_hashtag=True, tokenize_hashtag=False):
    """
    Reads and returns SemEval 2017 Task 6 dataset from semeval_data_root_dir
    
    :param semeval_data_root_dir: the root directory of the SemEval 2017 Task 6 dataset
    :type semeval_data_root_dir: str
    :param training_dirs: the directories in the root directory containing training set. By default this include both Training and Trial
    :type training_dirs: List[str]
    :param evaluation_dir: the directory in the root directory containing evaluation set
    :type evaluation_dir: str
    :param validation_dir: the directory in the root directory containing the validation set. If None or blank, no validation set will be included
    :param remove_midnight: Indicates if @midnight twitter acount should be removed from tweet body
    :type remove_midnight: bool
    :param remove_hashtag: Indicates if relevant Hastag War hashtags (specified by file name) should be removed. Overrides tokenize_hashtag.
    :type remove_hashtag: bool
    :param tokenize_hashtag: Indicates whether Hashtag War hashtags should be tokenized (according to file name). This flag is ignored if remove_hashtag is True
    :type tokenize_hashtag: bool
    
    :returns: A tuple containing training, evaluation, and optionally validation data in (training, evaluation[, validation]) format
    :rtype: Tuple[Dict[str, List[Tuple[str, str, int]]], Dict[str, List[Tuple[str, str, int]]]] (or Tuple[Dict[str, List[Tuple[str, str, int]]], Dict[str, List[Tuple[str, str, int]]], Dict[str, List[Tuple[str, str, int]]]] if validation_dir is not None)
    """
    
    training_data = {}
    if type(training_dirs) == str: #special case for only 1 training dir
        training_dirs = [training_dirs]
    
    for train_dir in training_dirs:
        training_data.update(read_semeval_directory(train_dir, remove_midnight, remove_hashtag, tokenize_hashtag))
    
    evaluation_data = read_semeval_directory(evaluation_dir, remove_midnight, remove_hashtag, tokenize_hashtag)
    
    output = (training_data, evaluation_data) #default to only training and evaluation data
    
    if validation_dir: #if there is a specified validation dir
        validation_data = read_semeval_directory(validation_dir, remove_midnight, remove_hashtag, tokenize_hashtag)
        output = (training_data, evaluation_data, validation_data)
    
    return output