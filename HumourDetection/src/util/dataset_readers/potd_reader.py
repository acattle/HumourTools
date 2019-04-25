'''
Created on Apr 25, 2019

:author: Andrew Cattle <acattle@connect.ust.hk>

Methods for reading Pun of the Day Data from Yang et al. (2015)

Dataset is not publicly available and was obtained by contacting paper authors
directly.

        Yang, D., Lavie, A., Dyer, C., & Hovy, E. H. (2015). Humor Recognition
        and Humor Anchor Extraction. In EMNLP (pp. 2367-2376).
'''
import re
from string import punctuation

def read_potd_data(file_loc):
    """
    Method for reading Pun of the Day data, as obtained from Yang et al. (2015)
    authors
    
    NOTE: puns_pos_neg_data.csv is missing punctuation. If this is important,
    try read_raw_potd_data().
    
    :param file_loc: the location of the potd data file
    :type file_loc: str
    
    :returns: a list containing the Pun of the Day data  in (document, label) format
    :rtype: List[Tuple[str, int]]
    """
    
    potd_data = []
    with open(file_loc, "r") as pos_f: #potd_pos
        pos_f.readline() #pop the header
                    
        for line in pos_f:
            label, doc = line.split(",", maxsplit=1) #some document conatin commas
            potd_data.append((doc.split(), int(label)))
    
    return potd_data

def read_raw_potd_data(pos_loc, neg_loc, proverbs_loc):
    """
    Method for reading raw format of Pun of the Day data. The original csv file
    obtained from the Yang et al. (2015) authors had punctuation stripped out,
    making parsing the documents difficult. By using the raw files, some of
    this missing punctuation is restored.
    
    This method is designed to take positive examples from puns_of_day.csv
    (NOTE: the labels in this file are incorrect and thus ignored) and
    negative examples from new_select.txt and proverbs.txt.
    
    :param pos_loc: the location of the positive examples
    :type pos_loc: str
    :param neg_loc: the location of the negative examples
    :type neg_loc: str
    :param proverbs_loc: the location of the proverbs (also taken to be negative exmples)
    :type proverbs_loc: str
    
    :returns: a list of (doc, label) tuples
    :rtype: List[Tuple[str, int]]
    """
    potd_docs_and_labels=[]
    
    with open(pos_loc, "r") as pos_f:
        pos_f.readline() #pop the header
                 
        for line in pos_f:
            _, doc = line.split(",", maxsplit=1) #some document conaiting commas. Ignore incorrect label
            doc=doc.strip()[1:-1] #cut off quotation marks
            potd_docs_and_labels.append((doc, 1)) #the labels in this file are incorrect. All are positive
            
    with open(neg_loc, "r") as neg_f:
        for line in neg_f:
            potd_docs_and_labels.append((line.strip(), -1))
            
    with open(proverbs_loc, "r") as neg_f:
        p = re.compile(f'[{re.escape(punctuation)}]')
        for line in neg_f:
            line=line.strip()
            if len(p.sub(" ", line).split()) >5:
                potd_docs_and_labels.append((line, -1))
                #TODO: a couple documents are surrounded by quotes. Selectively remove them?
    
    return potd_docs_and_labels

#     with open(potd_pos, "r") as pos_f:
#         pos_f.readline() #pop the header
#                     
#         for line in pos_f:
#             label, doc = line.split(",", maxsplit=1) #some document conatin commas
#             doc=doc.strip()[1:-1] #cut off quotation marks
#             potd_docs_and_labels.append((doc, 1)) #the labels in this file are incorrect. All are postive
#     with open(potd_neg, "r") as neg_f:
#         for line in neg_f:
#             potd_docs_and_labels.append((line.strip(), -1))
#     with open(proverbs, "r") as neg_f:
#         p = punc_re = re.compile(f'[{re.escape(punctuation)}]')
#         for line in neg_f:
#             line=line.strip().lower()
#             if len(p.sub(" ", line).split()) >5:
#                 potd_docs_and_labels.append((line.strip(), -1))
#                 #TODO: a couple documents are surrounded by quotes. Selectively remove them?