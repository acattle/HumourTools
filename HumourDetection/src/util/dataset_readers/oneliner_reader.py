'''
Created on Apr 25, 2019

:author: Andrew Cattle <acattle@connect.ust.hk>
'''

def read_16000_oneliner_data(pos_loc, neg_loc):
    """
    Method for reading 16000 Oneliner dataset used in Mihalcea and Strapparava
    (2005). Dataset was obtained by contacting the original paper's authors
    directly.
    
        Mihalcea, R., & Strapparava, C. (2005, October). Making computers laugh:
        Investigations in automatic humor recognition. In Proceedings of the
        Conference on Human Language Technology and Empirical Methods in Natural
        Language Processing (pp. 531-538). Association for Computational
        Linguistics.
    
    :param pos_loc: the location of the positive examples
    :type pos_loc: str
    :param neg_loc: the location of the negative examples
    :type neg_loc: str
    """
    
    oneliner_docs_and_labels = []
    with open(pos_loc, "r", encoding="ansi") as ol_pos_f:
        for line in ol_pos_f:
            oneliner_docs_and_labels.append((line.strip(),1))
    with open(neg_loc, "r", encoding="ansi") as ol_neg_f:
        for line in ol_neg_f:
            oneliner_docs_and_labels.append((line.strip(),-1))
    
    return oneliner_docs_and_labels