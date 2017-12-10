'''
Created on Jan 25, 2017

@author: Andrew
'''
#first line is header (num synsets, dimensions)
#each line after that is "synset_id dim_0 dim_1 ... dim_299"
#mapings from synset_id to actually usable synset name is in mappings.txt
from nltk.corpus import wordnet
from os.path import join
from numpy import array, zeros, isnan
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from nltk.corpus.reader.wordnet import WordNetError
import pickle
import warnings

class AutoExtendEmbeddings(object):
    def __init__(self, synset_vector_loc, mapping_loc,  wn_21_to_30_sensemap_dir, min_score=0):
        wn_21_keys_to_30_synsets = {}
        
        #load mono mappings for nouns and verbs
        for pos in ["noun", "verb"]:
            with open(join(wn_21_to_30_sensemap_dir, "2.1to3.0.{}.mono".format(pos)), "r") as mono_file:
                for line in mono_file:
                    key_21, offset_21, key_30, offset_30 = line.split()
                    
                    try:
                        synset_name_30 = wordnet.lemma_from_key(key_30).synset().name()
                        if key_21 not in wn_21_keys_to_30_synsets: #initialize to an empty list
                            wn_21_keys_to_30_synsets[key_21] = []
                        wn_21_keys_to_30_synsets[key_21].append(synset_name_30) #add the synset name to the list
                    except WordNetError:
                        #I guess that synset doesn't exist
                        warnings.warn("3.0 Synset {} does not exist".format(key_30))
                        
                
            with open(join(wn_21_to_30_sensemap_dir, "2.1to3.0.{}.poly".format(pos)), "r") as poly_file:
                for line in poly_file:
                    line_split = line.split()
                    if len(line_split) < 3: #if we don't have at last one WN3.0 synset to map too
                        continue
                    
                    score = line_split[0]
                    if score < min_score: #if we don't meet a minimum threshold
                        continue
                    
                    key_21, offset_21, sense_num_21 = line_split[1].split(";")

                    for sense_info_30 in line_split[2:]:
                        key_30, offset_30, sense_num_30 = sense_info_30.split(";")
                        try:
                            synset_name_30 = wordnet.lemma_from_key(key_30).synset().name()
                            if key_21 not in wn_21_keys_to_30_synsets: #initialize to an empty list
                                wn_21_keys_to_30_synsets[key_21] = []
                            wn_21_keys_to_30_synsets[key_21].append(synset_name_30) #add the synset name to the list
                        except WordNetError:
                            #I guess that synset doesn't exist
                            warnings.warn("3.0 Synset {} does not exist".format(key_30))
        
        #load AutoEx synset to vector name mapping
        id_to_synsets = {}
        with open(mapping_loc, "r") as mapping_file:
            for line in mapping_file:
                line_split = line.split()
                if len(line_split) < 2:
                    warnings.warn("Poorly formatted line. Ignoring.\n{}".format(line))
                    continue
                    
                synset_id, keys_21 = line_split
                for key_21 in keys_21.split(","):
                    if key_21 == "":
                        continue
                    
                    try:
                        if synset_id not in id_to_synsets: #initialize to an empty list
                            id_to_synsets[synset_id] = []
                        id_to_synsets[synset_id].extend(wn_21_keys_to_30_synsets[key_21])
                    except KeyError:
                            #I guess that synset doesn't exist
                            warnings.warn("2.1 Synset {} does not exist".format(key_21))
        
        
        #Now I can load the vectors
        self.autoex_vectors = {}
        with open(synset_vector_loc, "r") as synset_file:
            synset_file.readline() #pop the header line
            for line in synset_file:
                line_split = line.split()
                synset_id = line_split[0]
                vector = array([float(val) for val in line_split[1:]])
                
                if synset_id not in id_to_synsets:
                    warnings.warn("No sysnets found for {}".format(synset_id))
                    continue
                
                for synset_name_30 in id_to_synsets[synset_id]:
                    self.autoex_vectors[synset_name_30] = vector
        
        print len(self.autoex_vectors)
    
    def get_vector(self,synset):
        vector = None
        try:
            vector = self.autoex_vectors[synset]
        except KeyError:
            vector = zeros(300)
        return vector
    
    def get_similarity(self, synset1, synset2):
        sim = 1-cosine(self.get_vector(synset1), self.get_vector(synset2)) #since it's cosine distance
        if isnan(sim):
            sim=0.0
        return sim
    
    def get_relative_entropy(self, synset1, synset2):
        return entropy(self.get_vector(synset1), self.get_vector(synset2))
    
    def get_offset(self, synset1, synset2):
        return self.get_vector(synset1) - self.get_vector(synset2)

if __name__ == "__main__":
    vector_dir = r"C:\Users\Andrew\Desktop\vectors"
    ae = AutoExtendEmbeddings(join(vector_dir, "synsets.txt"), join(vector_dir, "mapping.txt"), join(vector_dir, "sensemap"))
    
    with open("autoextend.pkl", "wb") as ae_file:
        pickle.dump(ae,ae_file)