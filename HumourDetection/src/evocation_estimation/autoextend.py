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
from collections import defaultdict
from itertools import combinations
import operator

class AutoExtendEmbeddings(object):
    """Class for getting AutoExtend pretrained vectors in a usable format from strings"""
    
    def __init__(self, synset_vector_loc, mapping_dir, min_score=0.0):
        """
            This initialization method reads pretrained AutoExtend vectors from disk,
            converts them to numpy arrays, and then maps each vector's WordNet 1.7.1
            synset to an equivalent WordNet 3.0 synset according to the mappings in
            the directory specfieid by mapping_dir.
            
            @param synset_vector_loc: The location of the synsets.txt file from the pretrained AutoExtend vectors available at http://www.cis.lmu.de/~sascha/AutoExtend/
            @type sysnet_vector_loc: str
            @param mapping_dir: The directory of the WordNet 1.7.1 to 3.0 mappings. Assumes unofficial TALP mappings available at http://www.talp.upc.edu/index.php/technology/resources/multilingual-lexicons-and-machine-translation-resources/multilingual-lexicons/98-wordnet-mappings?highlight=WyJ3b3JkbmV0Il0
            @type mapping_dir: str
            @param min_score: The minimum mapping score to be considered (default: 0.0, i.e. all mappings considered)
            @type min_score: float
            
            @return A dictionary of WordNet 3.0 synset names to AutoExtend vectors
            @rtype {str : numpy.array}
        """
        
        
        #read the TALP WordNet mappings
        mapping_filename_template = "wn171-30.{}"
        offset_171_to_30 = {}
        for pos_tag, pos_name in [("n", "noun"), ("v", "verb"), ("a", "adj"), ("r", "adv")]:
            with open(join(mapping_dir, mapping_filename_template.format(pos_name)), "r") as mapping_file:
                offset_171_to_30[pos_tag] = defaultdict(list)
                for line in mapping_file:
                    #each line in TALP mapping files is in the format "old_offset( new_offset score)+"
                    line_split = line.split()
                    offset_171 = line_split[0]
                     
                    for i in range(1, len(line_split), 2):
                        offset_30 = line_split[i]
                        score = line_split[i+1]
                         
                        if score > min_score:
                            offset_171_to_30[pos_tag][offset_171].append((offset_30, score))
        
#         sensemap_filename_template = "2.1to3.0.{}.{}"
#         sensekey_21_to_30 = defaultdict(list)
# #         offset_21_to_30 = defaultdict(list)
#         for pos in ["noun", "verb"]:
#                 with open(join(sensemap_loc, sensemap_filename_template.format(pos, "mono")), "r") as sensemap_file:
#                     for line in sensemap_file:
#                         line_split = line.split()
#                         if len(line_split) != 4:
#                             print "Weird line length: {}".format(len(line_split))
#                             print line
#                             continue
#                         
#                         sensekey_21, offset_21, sensekey_30, offset_30 = line_split
#                         if sensekey_21 != sensekey_30:
#                             sensekey_21_to_30[sensekey_21].append(sensekey_30)
# #                         offset_21_to_30[offset_21].append(offset_30)
#                 
#                 with open(join(sensemap_loc, sensemap_filename_template.format(pos, "poly")), "r") as sensemap_file:
#                     for line in sensemap_file:
#                         line_split = line.split()
#                         if len(line_split) < 3:
#                             print "no 3.0 synsets given"
#                             print line
#                             continue
#                         
#                         score = line_split[0]
#                         if score > min_score:
#                             sensekey_21, offset_21, _ = line_split[1].split(";")
#                             for wordnet_30_mapping in line_split[2:]:
#                                 sensekey_30, offset_30, _ = wordnet_30_mapping.split(";")
#                                 if sensekey_21 != sensekey_30:
#                                     sensekey_21_to_30[sensekey_21].append(sensekey_30)
#                                 offset_21_to_30[offset_21].append(offset_30)

#         sensekey_21_to_synset_30 = defaultdict(list)
#         for sensekey_21 in sensekey_21_to_30:
#             for sensekey_30 in sensekey_21_to_30[sensekey_21]:
#                 sensekey_21_to_synset_30[sensekey_21].append(wordnet.lemma_from_key(sensekey_30).synset().name())
        
        #load AutoEx synset to vector name mapping
#         id_to_synsets = defaultdict(list)
#         no_lemma_given = []
#         bad_lemma = []
#         ambiguous_lemma = []
#         lemma_to_id = {}
#         matched =0
#         m_by_offset = 0
#         with open(mapping_loc, "r") as mapping_file:
# #             offsetmap=[]
# #             newmap=[]
#             lines = mapping_file.readlines()
#             print "{} total".format(len(lines))
#             for line in lines:
# #                 c=None
# #                 a=line.rstrip().split(' ')
# #                 if len(a) == 2:
# #                     b=a[1][:-1].split(',')
# #                     for i in b:
# #                         try:
# #                             c=wordnet.lemma_from_key(i)
# #                             matched += 1
# #                         except WordNetError:
# #                             pass
# # #                             warnings.warn("Lemma {} does not exist in WN3.0".format(i))
# # #                         os=str(c['offset']).zfill(8)+'-'+c['pos']
# # #                         offsetmap.append([a[0],os])
# # #                         newmap.append([os,','.join(c['lemmas_keys'])])
# #                         break
# #              
# # #             offsetmap2=map(lambda x: [x[0],'wn-3.0-'+x[1]],offsetmap)
# #             print "{} matched".format(matched)
#             
# #             for line in mapping_file:
#                 line_split = line.split()
#                 synset_id = line_split[0]
#                 synset_id_split = synset_id.split("-")
#                 offset = int(synset_id_split[2].lstrip("0"))
#                 pos = synset_id_split[3]
#                 
#                   
#                 #try to add by offset
#                 synsets = set()
# #                 try:
# #                     synsets.add(wordnet._synset_from_pos_and_offset(pos, offset).name())
# # #                     m_by_offset +=1
# # #                     matched += 1
# # #                     continue
# #                 except:
# #                     pass
# #                     if synset_id not in id_to_synsets: #initialize to an empty set
# #                         id_to_synsets[synset_id] = set()
# #                     id_to_synsets[synset_id].add(offset_to_synset[offset])
#                   
#                 #try to add by lemma
#                 if len(line_split) < 2:
# #                     no_lemma_given.append(line)
#                     continue
#                       
#                 lemma_keys = line_split[1]
#                 for key in lemma_keys.split(","):
#                     if key == "":
#                         continue
#                       
#                     lemma_to_id[key] = synset_id
#                     
#                     keys = [key]
#                     if key in sensekey_21_to_30:
#                         keys = sensekey_21_to_30[key]
#                         
#                     for key in keys:
#                         try:
#                             lemma = wordnet.lemma_from_key(key)
#                             synsets.add(lemma.synset().name())
# #                         break
#                         except WordNetError:
#                             pass
#                     
#                 if len(synsets) > 0:
#                     id_to_synsets[synset_id].extend(synsets)
#              
#             print "{} matched".format(len(id_to_synsets))
#             print "{} by offset".format(m_by_offset)
#                             #I guess that synset doesn't exist
#                             warnings.warn("Lemma {} does not exist in WN3.0".format(key))
#                             word, pos = k.split(":")[0].split("%")
#                             possible_synsets = wordnet.synsets(word, num_to_pos[int(pos)])
#                             if len(possible_synsets) == 1: #there is only 1 sysnet it could be
#                                 if synset_id not in id_to_synsets: #initialize to an empty set
#                                     id_to_synsets[synset_id] = set()
#                                 id_to_synsets[synset_id].add(possible_synsets[0].name())
#                             elif len(possible_synsets) > 1: #ambiguous word
#                                 ambiguous_lemma.append(k)
#                             else: #bad lemma
#                                 bad_lemma.append(k)
#                             continue
#                     
#                     if synset_id not in id_to_synsets: #initialize to an empty set
#                         id_to_synsets[synset_id] = set()
#                     id_to_synsets[synset_id].add(lemma.synset().name())
#             
#         for key in id_to_synsets:
#             if len(id_to_synsets[key]) >1:
#                 print "{} has {} synsets".format(key, len(id_to_synsets[key]))
#                 for synset in id_to_synsets[key]:
#                     print "\t{}".format(synset)
#                 print "\n"
#         
#         
        #Now I can load the vectors
        self.autoex_vectors = {}
        matched = 0
        unknown_w_lemmas = 0
        synset_to_vect = defaultdict(set)
        vect_to_synset = defaultdict(set)
        id_to_vect = {}
        with open(synset_vector_loc, "r") as synset_file:
            lines = synset_file.readlines()
            print "{} vectors".format(len(lines)-1)
            for line in lines[1:]: #skip header line
#             synset_file.readline() #pop the header line
#             for line in synset_file:
                
                line_split = line.split()
                synset_id = line_split[0]
                synset_id_split = synset_id.split("-")
                offset_171 = synset_id_split[2]
                pos = synset_id_split[3]
                vector = array([float(val) for val in line_split[1:]])
                id_to_vect[synset_id]=vector
                 
#                 if synset_id not in id_to_synsets:
#                     warnings.warn("No sysnets found for {}".format(synset_id))
#                     continue
                if offset_171 not in offset_171_to_30[pos]:
                    #No known WordNet 3.0 synset for this vector, skip it
                    continue
                 
#                 for synset_name_30 in id_to_synsets[synset_id]:
                added = False
                for offset_30, score in offset_171_to_30[pos][offset_171]:
                    try:
                        synset_name_30 = wordnet._synset_from_pos_and_offset(pos, int(offset_30.lstrip("0"))).name()
                        added = True
                        self.autoex_vectors[synset_name_30] = vector
                        vect_to_synset[synset_id].add((synset_name_30, score))
                        synset_to_vect[synset_name_30].add((synset_id, score))
                    except WordNetError:
                        #No synset found, ignore this vector
                        pass
                if added:
                    matched +=1
                    
        print "{} matched".format(matched)
        
#         print "\nVectors mapped to multiple synsets"
#         for s_id, synsets in vect_to_synset.items():
#             if len(synsets) > 1:
#                 print "{}\t{}".format(s_id, " ".join([str(synset) for synset in synsets]))
        
        print "\nSynsets mapped to multiple vectors"
        for synset, s_ids in synset_to_vect.items():
            if len(s_ids) > 1:
                print "{}\t{}".format(synset, " ".join(["{}:{}".format(s_id) for s_id in s_ids]))
            
#                 for s_1, s_2 in combinations(s_ids, 2):
#                     cos = cosine(id_to_vect[s_1[0]], id_to_vect[s_2[0]])
#                     if isnan(cos):
#                         print "NAN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
#                         print id_to_vect[s_1[0]]
#                         print id_to_vect[s_2[0]]
#                     print "\t\t\t{} {} - {}".format(s_1, s_2, cos)
#         print len(self.autoex_vectors)
    
    def get_vector(self,synset):
        vector = None
        try:
            vector = self.autoex_vectors[synset]
        except KeyError:
            warnings.warn("No vector found for " + synset)
            vector = zeros(300)
        return vector
    
    def get_similarity(self, synset1, synset2):
        return 1-cosine(self.get_vector(synset1), self.get_vector(synset2)) #since it's cosine distance
    
    def get_relative_entropy(self, synset1, synset2):
        return entropy(self.get_vector(synset1), self.get_vector(synset2))
    
    def get_offset(self, synset1, synset2):
        return self.get_vector(synset1) - self.get_vector(synset2)

if __name__ == "__main__":
    vector_dir = "C:/Users/Andrew/Desktop/vectors"
    mapping_dir = "C:/Users/Andrew/Desktop/mappings-upc-2007/mapping-171-30"
#     mapping_dir_2 = "C:/Users/Andrew/Desktop/mappings-upc-2007/mapping-171-30"
    ae = AutoExtendEmbeddings(join(vector_dir, "synsets.txt"), mapping_dir)
    
    with open("autoextend.pkl", "wb") as ae_file:
        pickle.dump(ae,ae_file)