'''
Created on Jun 22, 2018

@author: Andrew
'''
import gzip
import csv
import numpy as np
from humour_features.utils.common_features import get_interword_score_features
from util.text_processing import get_word_pairs


_relations = ["/r/RelatedTo",
              "/r/ExternalURL",
              "/r/FormOf",
              "/r/IsA",
              "/r/PartOf",
              "/r/HasA",
              "/r/UsedFor",
              "/r/CapableOf",
              "/r/AtLocation",
              "/r/Causes",
              "/r/HasSubevent",
              "/r/HasFirstSubevent",
              "/r/HasLastSubevent",
              "/r/HasPrerequisite",
              "/r/HasProperty",
              "/r/MotivatedByGoal",
              "/r/ObstructedBy",
              "/r/Desires",
              "/r/CreatedBy",
              "/r/Synonym",
              "/r/Antonym",
              "/r/DistinctFrom",
              "/r/DerivedFrom",
              "/r/SymbolOf",
              "/r/DefinedAs",
              "/r/Entails",
              "/r/MannerOf",
              "/r/LocatedNear",
              "/r/HasContext",
              "/r/dbpedia/genre",
              "/r/dbpedia/influencedBy",
              "/r/dbpedia/knownFor",
              "/r/dbpedia/occupation",
              "/r/dbpedia/language",
              "/r/dbpedia/field",
              "/r/dbpedia/product",
              "/r/dbpedia/capital",
              "/r/dbpedia/leader", 
              "/r/SimilarTo",
              "/r/EtymologicallyRelatedTo",
              "/r/EtymologicallyDerivedFrom",
              "/r/CausesDesire",
              "/r/MadeOf",
              "/r/ReceivesAction",
              "/r/InstanceOf",
              "/r/NotHasProperty",
              "/r/NotCapableOf",
              "/r/NotDesires"
              ]
_relations_index = dict(zip(_relations, range(len(_relations))))

def igraph_from_assertions(assertion_file_loc, english_only=True):
    from igraph import Graph
    
    def is_english(concept):
        return concept.split("/")[2] == "en"
    def get_word(concept):
        return concept.split("/")[3]
     
    edges = []
    with gzip.open(assertion_file_loc, "rt", encoding="utf-8") as f:
        r=csv.reader(f, delimiter="\t")
         
        for l in r:
             
            #check for english
            if english_only and (not is_english(l[2]) or not is_english(l[3])):
                continue
            
            relation = l[1]
            edges.append((get_word(l[2]), get_word(l[3]),relation))

    return Graph.TupleList(edges, edge_attrs="relation", directed=False)

def get_relation_count_vectors(documents, relations_dict, token_filter=None, pair_generator=None, include_backward=False):
    """
    
    :param documents: documents to be processed. Each document should be a sequence of tokens
    :type documents: Iterable[Iterable[str]]
    :param scorer: the scoring function to use when comparing words. Must take two strings as input and return a score as a float.
    :type scorer: Callable[[str, str], Number]
    :param token_filter: function for filtering the tokens in a document. If None, no token filtering will be done
    :type token_filter: Callable[[Iterable[str]], Iterable[str]]
    :param pair_generator: function for generating word pairs. Must take list of documents (as a list of tokens) and a token_filter fucntion. If None, default word pair generator will be used.
    :type pair_generator: Callable[[Iterable[str], Callable[[Iterable[str]], Iterable[str]]], List[List[Tuple[str, str]]]]
    
    :return: A matrix of relation count vectors of size 47 x # of docuemnts
    :rtype: numpy.array
    """
    
    if not pair_generator:
        pair_generator = get_word_pairs
    documents = pair_generator(documents, token_filter)
    
    
    feature_vectors = []
    for document in documents:
        relation_vector = np.zeros(len(_relations))
        for wp in document:
            for rel, count in relations_dict.get(wp, {}).items():
                relation_vector[_relations_index[f"/r/{rel}"]] += count
        
        feature_vectors.append(relation_vector)
    
    ret = np.vstack(feature_vectors)
    
    if include_backward:
        back_docs = []
        for doc in documents:
            back_docs.append([(b,a) for a,b in doc])
        back_matrix = get_relation_count_vectors(back_docs, relations_dict, pair_generator=lambda x,y: x, include_backward=False)
        ret = (ret, back_matrix)
            
    return ret
    
    
#TODO: is this even needed?
def get_numberbatch_sims(documents, scorer, token_filter=None, pair_generator=None):
    """
        A convenience method for automatically prefixing "/c/en/" to all tokens before scoring,
        making them suitable for use with ConceptNet Numberbatch vectors.
        
        :param documents: documents to be processed. Each document should be a sequence of tokens
        :type documents: Iterable[Iterable[str]]
        :param scorer: the scoring function to use when comparing words. Must take two strings as input and return a score as a float.
        :type scorer: Callable[[str, str], Number]
        :param token_filter: function for filtering the tokens in a document. If None, no token filtering will be done
        :type token_filter: Callable[[Iterable[str]], Iterable[str]]
        :param pair_generator: function for generating word pairs. Must take list of documents (as a list of tokens) and a token_filter fucntion. If None, default word pair generator will be used.
        :type pair_generator: Callable[[Iterable[str], Callable[[Iterable[str]], Iterable[str]]], List[List[Tuple[str, str]]]]
        
        :return: A matrix in the form (min_score, avg_score, max_score) x # of documents
        :rtype: numpy.array
    """
    
    pair_generator = pair_generator if pair_generator else get_word_pairs
    documents = pair_generator(documents)
    
    #prefix /c/en/ to each token
    prefixed_documents = [[(f"/c/en/{token1}", f"/c/en/{token2}")  for token1, token2 in doc] for doc in documents] #need to prefix post-pair generation due to humour anchors
    
    return get_interword_score_features(prefixed_documents, scorer, token_filter=token_filter, pair_generator=lambda x,y:x)


