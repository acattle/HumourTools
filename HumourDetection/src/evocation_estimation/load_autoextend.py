'''
Created on Feb 9, 2017

@author: Andrew
'''
from __future__ import print_function #for Python 2.7 compatibility
from os.path import join
from collections import defaultdict
from numpy import array
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import WordNetError

def load_autoextend(synset_vector_loc, mapping_dir, min_score=0.0):
    """
        This method reads pretrained AutoExtend vectors from disk, converts them
        to numpy arrays, and then maps each vector's WordNet 1.7.1 synset to an
        equivalent WordNet 3.0 synset according to the mappings in the directory
        specfieid by mapping_dir.
        
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
            offset_171_to_30[pos_tag] = defaultdict(set)
            for line in mapping_file:
                #each line in TALP mapping files is in the format "old_offset( new_offset score)+"
                line_split = line.split()
                offset_171 = line_split[0]
                
                for i in range(1, len(line_split), 2):
                    offset_30 = line_split[i]
                    score = line_split[i+1]
                    
                    if score > min_score:
                        offset_171_to_30[pos_tag][offset_171].add(offset_30)
    
    #Now I can load the vectors
    autoex_vectors = {}
    matched =0 #for debugging
    with open(synset_vector_loc, "r") as synset_file:
        synset_file.readline() #skip header line
        for line in synset_file:
            line_split = line.split()
            synset_id = line_split[0]
            synset_id_split = synset_id.split("-")
            offset_171 = synset_id_split[2]
            pos = synset_id_split[3]
            vector = array([float(val) for val in line_split[1:]])
            
            if offset_171 not in offset_171_to_30[pos]:
                #No known WordNet 3.0 synset for this vector, skip it
                continue
            
            match = False #for debugging
            for offset_30 in offset_171_to_30[pos][offset_171]:
                try:
                    synset_name_30 = wordnet._synset_from_pos_and_offset(pos, int(offset_30.lstrip("0"))).name()
                    autoex_vectors[synset_name_30] = vector
                    match = True #for debugging
                except WordNetError:
                    #No synset found, ignore this vector
                    pass
            
            #for debugging
            if match:
                matched+=1
            
    print(matched)
    return autoex_vectors

if __name__ == "__main__":
    vector_dir = "C:/Users/Andrew/Desktop/vectors"
    mapping_dir = "C:/Users/Andrew/Desktop/mappings-upc-2007/mapping-171-30"
    ae = load_autoextend(join(vector_dir, "synsets.txt"), mapping_dir)
    print(len(ae))