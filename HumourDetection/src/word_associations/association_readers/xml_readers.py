'''
Created on Feb 7, 2018

@author: Andrew
'''
import xml.etree.ElementTree as ET
from collections import defaultdict
from util.misc import mean
import os
import csv

CONTROLLED = "controlled"
MT_ALL = "mt_all"
MT_MOST = "mt_most"
MT_SOME = "mt_some"

class EvocationDataset(object):
    """
    This class contains functions for reading and using WordNet Evocation in its
    original format, as found at http://wordnet.cs.princeton.edu/downloads.html 
    """
    
    def __init__(self, evocation_dir, mt_group=MT_ALL):
        """ Read evocation scores from "controlled" group plus an optional Mechanical Turk group.
            
            :param evocation_dir: the directory where the evocation files are stored
            :type evocation_dir: str
            :param mt_group: The Mechanical Turk group of judgements to read in addition to the controlled group. Must be one of "mt_all", "mt_most". "mt_some", or None.
            :type mt_group: str or None
        """
        
        #read in the control data
        evocation_scores = self._read_scores(evocation_dir, CONTROLLED)
        
        #if we're using MT data
        if mt_group:
            #add mt judgements
            evocation_scores = self._read_scores(evocation_dir, mt_group, evocation_scores)
        
        #take simple average of scores
        self.evocation = {}
        for synset_pair, judgements in evocation_scores.items():
            if len(synset_pair) != 2:
                print(synset_pair)
            self.evocation[synset_pair] = mean(judgements)/100.0 #average judgements and convert to a probability
                
    def _read_scores(self, evocation_dir, group=CONTROLLED, evocation_scores=defaultdict(list)):
        """ Convenience method for reading WordNet Evocation files
            
            :param evocation_dir: The directory where the evocation files are stored
            :type evocation_dir: str
            :param group: The group of judgements to read. Must be one of "controlled", "mt_all", "mt_most", or "mt_some"
            :type group: str
            :param evocation_scores: A dictionary containing already counted evocation scores. Used for combining scores from multiple groups. Keys are (first synset, second synset). Values are a list of floats
            :type evocation_scores: defaultdict(list)
            
            :returns: A dictionary containing evocation scores. Keys are (first synset, second synset). Values are a list of floats
            :rtype: defaultdict(list)
            
            :raises ValueError: if group is not one of "controlled", "mt_all", "mt_most", or "mt_some"
        """
        group = group.lower()
        if group not in [CONTROLLED, MT_ALL, MT_MOST, MT_SOME]:
            raise ValueError("Unknown evocation group '{}'. Must be one of '{}', '{}', '{}', or '{}'.".format(group, CONTROLLED, MT_ALL, MT_MOST, MT_SOME))
        
        word_pos_loc = os.path.join(evocation_dir, "{}.word-pos-sense".format(group))
        raw_loc = os.path.join(evocation_dir, "{}.raw".format(group))
        with open(word_pos_loc, "r") as word_pos_file, open(raw_loc, "r") as raw_file:
            for word_pos_line, raw_line in zip(word_pos_file, raw_file):
                synset1, synset2 = word_pos_line.strip().split(",")
                scores = [float(score) for score in raw_line.split()]
            
                evocation_scores[(synset1, synset2)].extend(scores)
        
        return evocation_scores
    
    def get_all_associations(self):
        """ Method for getting all assocation scores
            
            @return A list of tuples representing (first synset, second synset, evocation score as a probability)
            @rtype List[Tuple[Tuple[str, str], float]]
        """
        return self.evocation.items()
    
class SWoW_Dataset:
    """
    Class for reading Small World of Words responses dataset from CSV
    and calcualting association strengths
    
    See https://smallworldofwords.org/en/project/research
    """
    
    def __init__(self, association_loc, complete=True):
        """
        Read Small World of Words response dataset in CSV format
        
        :param: association_loc: location of the SWoW csv
        :type: association_loc: str
        :param: complete: Specifies whether file at association_loc is the "complete" or "R100" version
        :type: complete: bool
        """
        
        totals = {}
        counts = {}
        self.vocab = set()
        self.assoc_dict = {}
        
        #SWOW-EN.complete columns
        cue_col = 11
        R123_cols = (15,16,17)
        no_more_str = "No more responses"
        if  not complete:
            cue_col = 9
            R123_cols = (10,11,12)
            no_more_str = "NA"
        
        with open(association_loc, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            
            #verify header
            header = next(reader)
            if header[cue_col] != "cue":
                raise RuntimeError(f"Expected 'cue' in column {cue_col}. Received '{header[cue_col]}'. Is SWoW format set correctly?")
            for col, label in zip(R123_cols,["R1", "R2", "R3"]):
                if header[col] != label:
                    raise RuntimeError(f"Expected '{label}' in column {col}. Received '{header[col]}'. Is SWoW format set correctly?")
            
            for row in reader:
                s = row[cue_col]
                
                self.vocab.add(s)
                totals[s] = totals.get(s,0) + 1
                
                for i in R123_cols:
                    r = row[i]
                    
                    #TODO: complete SWoW dataset includes "Unknown word" (and "unkown word") entries that seem to be auto-generated. Should we ignore them too?
                    if r == no_more_str:
                        #no more responses. Skip to next line
                        break
                    
                    self.vocab.add(r)
                    counts[(s,r)] = counts.get((s,r), 0) + 1
        
        self.assoc_dict = {}  
        for (s,r), count in counts.items():
            self.assoc_dict[(s,r)] = count / totals[s]
    
    def get_all_associations(self):
        """
        Return all SWoW associations as a tuple containing the cue and response (as a nested tuple) and strength
        
        :returns: list of tuples containing cue, response, and strength in that order
        :rtype: List[Tuple[Tuple[str,str],float]] 
        """
        return list(self.assoc_dict.items())

class SWoW_Strengths_Dataset:
    """
    Class for reading precompiled Small World of Words strength dataset from CSV
    
    See https://smallworldofwords.org/en/project/research
    """
    
    def __init__(self,association_loc):
        """
        Read Small World of Words strength dataset in CSV format
        
        :param: association_loc: location of the SWoW csv
        :type: association_loc: str
        """
        
        self.assoc_dict = {}
        with open(association_loc, "r", encoding="utf-8") as f:
            f.readline() #pop header
            
            for l in f:
                l=l.strip()
                if l: #ignore empty final line
                    cue, resp, _, _, stren = l.split("\t")
                    
                    self.assoc_dict[(cue, resp)] = float(stren)
    
    def get_all_associations(self):
        """
        Return all SWoW associations as a tuple containing the cue and response (as a nested tuple) and strength
        
        :returns: list of tuples containing cue, response, and strength in that order
        :rtype: List[Tuple[Tuple[str,str],float]] 
        """
        return list(self.assoc_dict.items())

class EAT_XML_Reader:
    """
    This class contians functions for reading and using Edinburgh Associative
    Thesaurus in its XML format, as found at
    http://rali.iro.umontreal.ca/rali/?q=en/Textual%20Resources/EAT
    """
    def __init__(self, xml_loc):
        self.root = ET.parse(xml_loc).getroot()
        
    def get_all_associations(self):
        """
            Get all word associations in EAT and their strengths
            
            :returns: word pairs and their association strength
            :rtype: List[Tuple[Tuple[str, str], float]]
        """
        associations=[]
        
        for stimulus_element in self.root:
            stimuli = stimulus_element.attrib["word"]
            total = float(stimulus_element.attrib["all"])
            
            for response_element in stimulus_element:
                response = response_element.attrib["word"]
                count = float(response_element.attrib["n"])
                
                associations.append(((stimuli, response), count/total))
        
        return associations


class USF_XML_Reader:
    """
    This class contians functions for reading and using Universitry of South
    Florida Free Associaiton Norms in their XML format, as found at
    http://rali.iro.umontreal.ca/rali/?q=en/USF-FAN
    """
    def __init__(self, xml_loc):
        self.root = ET.parse(xml_loc).getroot()
        
    def get_all_associations(self):
        """
            Get all word associations in USF and their strengths
            
            :returns: word pairs and their association strength
            :rtype: List[Tuple[Tuple[str, str], float]]
        """
        associations=[]
        
        for stimulus_element in self.root:
            stimuli = stimulus_element.attrib["word"]
            
            for response_element in stimulus_element:
                response = response_element.attrib["word"]
                total = float(response_element.attrib["g"]) #group size
                count = float(response_element.attrib["p"]) #number of participants
                
                associations.append(((stimuli, response), count/total))
        
        return associations