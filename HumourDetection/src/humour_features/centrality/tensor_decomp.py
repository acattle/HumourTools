import numpy as np
from tensorly.decomposition import parafac

def decompose_tensors(documents, win_size=5, cp_rank=10, binary_counts=True, tokenizer=str.split):
    """
    Method for computing tensor decompositions based on https://www.cs.ucr.edu/~epapalex/papers/asonam18-fakenews.pdf
    
    :param documents: the documents to be decomposed, tokenized
    :type documents: Iterable[Iterable[str]]
    :param win_size: the window size to use for word cooccurance. Will look win_size to the left and win_size to the right of each token
    :type win_size: int
    :param cp_rank:
    :type cp_rank: int
    :param binary_counts: Specifies whether cooccurance counts should be binary
    :type binary_counts: bool
    :param tokenizer: method for tokenizing documents. Default is str.split()
    :type tokenizer: Callable[str, [List[str]]] 
    """
    vocab = set()
    cooccurance_dicts = []
    
    
    #computer cooccurance matrices
    for doc in documents:
        cooccurance_dict = {}
        
        #tokenize do
        doc=tokenizer(doc)
        doc_length = len(doc)
        
        for i, word in enumerate(doc):
            vocab.add(word) #update the vocabulary
            
            if word not in cooccurance_dict:
                #if we haven't seen this word before...
                #... initialize its dict
                cooccurance_dict[word]={}
                
            #get the win_size words to the left of the target word...
            win_words = doc[max(0, i-win_size):i] #max() prevents negative indexes
            #because windows are symmetrical, we only need to look at one side of the window
            #i.e. being in a word's window on the left means they're in our window on the right
            
            for win_word in win_words:
                if binary_counts:
                    #set binary cooccurance value to 1
                    cooccurance_dict[word][win_word] = 1.0
                    cooccurance_dict[win_word][word] = 1.0
                else:
                    #increment the count by 1 (defaulting to 0 if the count is undefined)
                    cooccurance_dict[word][win_word] = cooccurance_dict[word].get(win_word, 0) + 1
                    cooccurance_dict[win_word][word] = cooccurance_dict[win_word].get(word, 0) + 1
        
        cooccurance_dicts.append(cooccurance_dict)
    
    #fix vocabulary word order
    vocab_map = {word:i for i,word in enumerate(vocab)}
    
    #construct cooccurance tensor
    tensor = np.zeros((len(documents), len(vocab), len(vocab))) # create a 3D array of 0s of size # documents x vocab size x vocab size
    
    for i, cooccurance_dict in enumerate(cooccurance_dicts):
        for word1, word1_dict in cooccurance_dict.items():
            for word2, val in word1_dict.items():
                tensor[i, vocab_map[word1], vocab_map[word2]] = val
    
    #perform CP_ALS decomposition
    decomp = parafac(tensor, cp_rank)
    
    return decomp[0], vocab_map #TODO: is vocab_map needed? It corresponds to the tensor, not decomp


if __name__ == "__main__":
    from time import strftime, localtime
    import glob
    import codecs
    import os
    from nltk.tokenize import word_tokenize
    from string import punctuation
    import re
    from scipy.spatial.distance import cdist
    from gensim.models import KeyedVectors
    from util.dataset_readers.semeval_2017_task_6_reader import read_semeval_2017_task_6_data
    
    punc = re.compile(f"[{punctuation}]+")
    midnight = re.compile("@midnight", flags=re.I)
    
    start_time = localtime()
    semeval_dir = "D:/datasets/SemEval Data"
#     semeval_dir = "/home/acattle/SemEval"
    dirs = ["trial_dir/trial_data",
            "train_dir/train_data"]#,
            #"evaluation_dir/evaluation_data"]
#     train_dir = "train_dir/train_data"
#     trial_dir = "trial_dir/trial_data"
    trial_dir= "evaluation_dir/evaluation_data"
#     experiment_dir = "{} top 5 feats on eval".format(strftime("%y-%m-%d_%H%M", start_time))
#     experiment_dir_full_path = os.path.join(semeval_dir,experiment_dir)
#     if not os.path.exists(experiment_dir_full_path):
#         os.makedirs(experiment_dir_full_path)
     
    prediction_dir = "D:/datasets/SemEval Data/predictions_50_test"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)


#     train_filenames = []
#     for d in dirs:
#         os.chdir(os.path.join(semeval_dir, d, tagged_dir))
#         for f in glob.glob("*.tsv"):
#             train_filenames.append(os.path.join(semeval_dir, d, tagged_dir,f))
    
#     test_filenames= []
#     os.chdir(os.path.join(semeval_dir, trial_dir))
#     docs = {}
#     all_texts=[]
#     text_map={}
#     for f in glob.glob("*.tsv"):
# #         hashtag = re.compile(f"#{f[:-4].replace('_','')}", flags=re.I)
#         hashtag = f[:-4].replace('_','')
# #         test_filenames.append(os.path.join(semeval_dir, trial_dir,f))
# #         
# #     
# #     for f in test_filenames:
#         labels = []
#         texts = []
#         ids = []
#         with codecs.open(f, "r", encoding="utf-8") as file:
#             for line in file:
#                 line = line.split("\t")
#                 ids.append(line[0])
#                 texts.append(midnight.sub("", hashtag.sub("", line[1])))
#                 labels.append(int(line[2]))
#     
#         texts = [[word for word in word_tokenize(text) if not punc.fullmatch(word)] for text in texts]
#         
#         docs[hashtag]=(texts,labels,ids)
#         all_texts.extend(texts)/
#         text_map.update((text, i) for i, text in enumerate(texts, len(all_texts)))
#         
# #         decomp, vocab_map = decompose_tensors(texts, cp_rank=65)
    semeval_data = read_semeval_2017_task_6_data(semeval_dir, remove_hashtag=True, tokenize_hashtag=True)
    
    eval_data = semeval_data[1]
    
    hashtag_indices={}
    for hashtag, docs in eval_data.items():
        all_texts=[]
#     for data in semeval_data:
#         for hashtag, docs in data.items():
        ids, texts, labels = zip(*docs)
        
        start = len(all_texts)
        end = start + len(texts)
        hashtag_indices[hashtag]=(start,end)
        all_texts.extend(texts)
    
        decomps, vocab_map = decompose_tensors(all_texts, cp_rank=50, binary_counts=False)
    
#     for hashtag, docs in eval_data.items():
#         ids, texts, labels = zip(*docs)
#         start, end = hashtag_indices[hashtag]
#         decomp = decomps[start:end,:]
        decomp=decomps
        center = np.mean(decomp, axis=0)
        distances=cdist(decomp, [center])
        
#         #w2v comparison
#         from util.model_wrappers.common_models import get_google_word2vec, get_stanford_glove
#         w2v = get_stanford_glove()
#         zeros = np.zeros(w2v.get_dimensions())
#         doc_embeddings = []
#         for doc in texts:
#             word_embeddings = [w2v.get_vector(word) for word in doc]
# #             word_embeddings = [embedding for embedding in word_embeddings if not np.array_equal(embedding, zeros)]
# #             if len(word_embeddings) == 0:
# #                 word_embeddings = [zeros]
#             doc_embeddings.append(sum(word_embeddings)/len(word_embeddings))
#         doc_embeddings = np.array(doc_embeddings)
#         center = np.mean(doc_embeddings, axis=0)
#         distances=cdist(doc_embeddings, [center])
        
        ranked=sorted(list(zip(ids, labels, distances)), key= lambda x : x[2]) #sort from most central to least
        
        global_predictions = list(zip(*ranked))[0]
        
        with open(os.path.join(prediction_dir,f"{hashtag}_PREDICT.tsv"), "w") as of:
            of.write("\n".join(global_predictions))
    
        
#         break #only do the first one
    
    
    
            