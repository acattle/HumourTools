'''
    Created on Dec 20, 2017

    :author: Andrew Cattle <acattle@cse.ust.hk>
    
    This module contains code for reading and using WordNet Domains
'''

import re

class WordNetDomains():
    """
        Class for querying WordNet Domains
    """
    
    def __init__(self,wnd_loc):
        """
            Loads WordNet Domains file and constructs synset/domain mappings
            
            As the official WordNet Domains file uses WordNet 2.0 synsets, this
            class requires the unoffical WordNet 3.0-mapped
            lifted-wordnet-domains, available at:
            
                https://github.com/morungos/lifted-wordnet-domains
            
            :param wnd_loc: location of the lifted-wordnet-domains file
            :type wnd_loc: str
        """
        
        self.synset_domain_map={}
        with open(wnd_loc, "r") as wnd_f:
            for line in wnd_f:
                _, synset_name, domains = line.split(maxsplit=2)
                
                synset_name = re.sub("#", ".", synset_name)
                domains = domains.split()
                
                self.synset_domain_map[synset_name] = domains
    
    def get_domains(self,synset_name):
        """
            Get all domains related to synset_name. Returns empty list if
            synset_name is not in WordNet Domains.
            
            :param synset_name: the name of the synset to lookup
            :type synset_name: str
            
            :return: list of relevant domains
            :rtype: Iterable[str]
        """
        return self.synset_domain_map.get(synset_name, [])