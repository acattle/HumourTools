'''
Created on Feb 7, 2018

@author: Andrew
'''
import xml.etree.ElementTree as ET
from collections import defaultdict
from util.misc import mean
import os

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
    
    
    

class EAT_XML_Reader:
    """
    This class contians functions for reading and using Edinburgh Associative
    Thesaurus in its XML format, as found at
    http://rali.iro.umontreal.ca/rali/?q=en/Textual%20Resources/EAT
    """
    def __init__(self, xml_loc):
        self.eat_root = ET.parse(xml_loc).getroot()
        
    def get_all_associations(self):
        """
            Get all word associations in EAT and their strengths
            
            :returns: word pairs and their association strength
            :rtype: List[Tuple[Tuple[str, str], float]]
        """
        associations=[]
        
        for stimulus_element in self.eat_root:
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
        self.eat_root = ET.parse(xml_loc).getroot()
        
    def get_all_associations(self):
        """
            Get all word associations in USF and their strengths
            
            :returns: word pairs and their association strength
            :rtype: List[Tuple[Tuple[str, str], float]]
        """
        associations=[]
        
        for stimulus_element in self.eat_root:
            stimuli = stimulus_element.attrib["word"]
            
            for response_element in stimulus_element:
                response = response_element.attrib["word"]
                total = float(response_element.attrib["g"]) #group size
                count = float(response_element.attrib["p"]) #number of participants
                
                associations.append(((stimuli, response), count/total))
        
        return associations