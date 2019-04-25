'''
Created on Jul 6, 2018

@author: Andrew Cattle <acattle@connect.ust.hk>

Utility functions related to Katz Spreading Activation.
For more information see:

Simon De Deyne, Daniel J. Navarro, Amy Perfors, Gert Storms, 2016,
'Structure at every scale: A semantic network account of the
similarities between unrelated concepts.', Journal of Experimental
Psychology: General, vol. 145, no. 9, pp. 1228-1254

These Python scripts are based on the R and matlab scripts available at:
https://github.com/SimonDeDeyne/SWOWEN-2018/tree/master/R/functions
'''

from scipy.sparse import csc_matrix, diags, identity, save_npz, load_npz
from scipy.sparse.linalg import inv
from word_associations.association_readers.igraph_readers import iGraphFromTuples
import numpy as np
# from numpy.linalg import inv
# from util.util_classes import IndexLookupWrapper




########## MATRIX OPERATIONS ##########

def _sum_as_array(mat, axis=0):
    return np.squeeze(np.asarray(mat.sum(axis=axis)))

def _normalize(mat, norm_vec):
    #remove inf (happens if the nromalization value of a row is 0)
    norm_vec[np.isinf(norm_vec)] = 0
    
    return diags(norm_vec, format=mat.format) * mat
#     return np.diag(norm_vec) * mat

def l1_normalize(mat):
    """
    L1 normalize a matrix
    
    :param mat: the matrix to L1 normalize
    :type mat: a scipy.sparse matrix
    
    :returns: L1 normalized mat
    :rtype: mat
    """
    
    norm_vec = 1/_sum_as_array(mat,axis=1)
#     norm_vec = 1/mat.sum(axis=1)
    
    return _normalize(mat, norm_vec)

def l1_numpy(mat):
    row_sums = mat.sum(axis=1)
#     return mat / row_sums[:, np.newaxis] #np.newaxis implicitly reshapes row_sums from (n,) to (n,1)

    #perform normalization row-by-row to avoid memory error
    mat = np.copy(mat) #copy mat to avoid in-place normalization
    for i, rs in enumerate(row_sums):
        mat[i] = mat[i] / rs
        
    mat[np.isinf(mat)] = 0 #get rid of infs if they happen
    #TODO: is this needed?
    
    return mat

def l2_normalize(mat):
    """
    L2 normalize a matrix
    
    :param mat: the matrix to L2 normalize
    :type mat: a scipy.sparse matrix
    
    :returns: L2 normalized mat
    :rtype: mat
    """
    
    norm_vec = 1/np.sqrt(_sum_as_array(mat**2,axis=1))
    return _normalize(mat, norm_vec)

def ppmi(mat):
    """
    Positive Pointwise Mutual Information
    
    :param mat: the matrix to perform PPMI on
    :type mat: scipy.sparse.csc_matrix
    
    :returns: the PPMI matrix
    :rtype: scipy.sparse.csc_matrix
    """
    
    n=mat.shape[0]
    d=diags(1/(_sum_as_array(mat,axis=0)/n), format=mat.format)
#     d=np.diag(1/(mat.sum(axis=0)/n))
    mat = mat*d #TODO: check that mat is a sparse matrix and not a numpy array
    #TODO: currently we assume mat is sparse. Add check for numpy
    mat.data = np.log2(mat.data) #only take the logs of the non-zero elements
    mat.data[mat.data < 0] = 0 #replace negative values with 0s. This is the POSTIVIE part of PPMI
    mat.eliminate_zeros() #get rid of any 0 values we may have added
#     mat = np.log2(mat) #TODO: is this what "P@x <- log2(P@x)" does?
#     mat[mat < 0] = 0 #replace negative values with 0s. This is the POSTIVIE part of PPMI

    return mat

def ppmi_numpy(mat):
    """
    Positive Pointwise Mutual Information
    
    :param mat: the matrix to perform PPMI on
    :type mat: scipy.sparse.csc_matrix
    
    :returns: the PPMI matrix
    :rtype: scipy.sparse.csc_matrix
    """
    
    #pmi is log(p(x|y)/p(x))
    #the values in mat are p(x|y), how is d related to p(x)?
    
    n=mat.shape[0]
    d=np.diag(n/(mat.sum(axis=0)))
    mat = np.dot(mat, d)
    mat[np.nonzero(mat)] = np.log2(mat[np.nonzero(mat)]) #only take the log of the non-zero elements
    mat[mat < 0] = 0 #replace negative values with 0s. This is the POSTIVIE part of PPMI

    return mat


def katz_walk(P, alpha=0.75):
    """
    Performs the Katz Spreading Activation transformation
    described in De Deyne et al. (2016)
    
    :param P: adjacency matrix to calculate spreading activation for
    :type P: scipy.sparse.csc_matrix
    :param alpha: the decay weight for each path step
    :type alpha: float
    
    :returns: An adjacency matrix representing the results of the Katz walk
    :rtype: scipy.sparse.csc_matrix
    """
    return inv(identity(P.shape[0], format=P.format) - alpha*P)

def katz_numpy(P, alpha=0.75):
    return np.linalg.inv(np.identity(P.shape[0]) - alpha*P)




########## GRAPH OPERATIONS ##########

def extract_component(G,mode="strong"):
    """
    Extracts the largest strongly connected component from graph G
    and converts it to a sparse adjacency matrix
    
    :param G: the graph to extract the component from
    :type G: igraph.Graph
    :param mode: the clustering mode. Must be either "strong" (i.e. each node has in-degree and out-degree >= 1) or "weak" (i.e. in-degree or out-degree >= 1)
    :type mode: str
    
    :returns: the largest strongly connected component as a sparse adjacency matrix and its corresponding word->index mapping
    :rtype: Tuple[scipy.sparse.csc_matrix, Dict[str, int]
    """
    #get largest connected component only
    #this reduces computational complexity
    G = G.components(mode).giant()
    
#     s=time()
#     adj_mat_from_adj = np.array(G.get_adjacency(attribute="weight").data)
#     print(time()-s)
    
#     #for use converting from words to matrix indexes
#     word_index = dict((n,i) for i, n in enumerate(G.vs["name"]))
#     vocab_size = len(word_index)
    
    #reorder the vocabulary to be in alphabetical order
    #optional step but makes indexes easier to interpret
    old_index_map = {name : i for i, name in enumerate(G.vs["name"])}
    sorted_names = sorted(G.vs["name"])
    new_index_map = {name : i for i, name in enumerate(sorted_names)}
    old_to_new = {old_index_map[name] : new_index_map[name] for name in sorted_names}
    vocab_size = len(sorted_names)

    #for each edge, make an (x,y,weight) tuple.
    #Then split it into separate x, y, and weight lists for constructing sparse matrix
#     s=time()
    xs,ys,ws = zip(*((*edge.tuple,edge["weight"]) for edge in G.es))
    
    #update indexes
    xs = [old_to_new[x] for x in xs]
    ys = [old_to_new[y] for y in ys]
    
    adj_mat = csc_matrix((ws, (xs,ys)), shape=(vocab_size, vocab_size)) #solve is more efficient for csc matrixes
#     print(time()-s)
#     adj_mat = adj_mat.todense()
#     print(time()-s)
#     s=time()
#     adj_mat_from_zeros = np.zeros((vocab_size,vocab_size))
#     for x,y,w in zip(xs,ys,ws):
#         adj_mat_from_zeros[x,y]=w
#     print(time()-s)
#     
#     print(adj_mat_from_adj.nbytes)
#     print(adj_mat_dense.nbytes)
    
#     adj_mat_wrapped = IndexLookupWrapper(adj_mat, new_index_map, ignore_case=True)
    
#     return adj_mat, word_index
    return adj_mat, new_index_map




def generate_katz_walk(cue_response_strengths):
    #convert to iGraph
    G = iGraphFromTuples(cue_response_strengths).graph
    
    to_del = [v.index for v in G.vs if G.degree(v, mode="OUT") == 0]
    G.delete_vertices(to_del)
    
    #for compatibility with katz
    G=remove_UK_words(G)
    
    #remove self loops, multiple edges
    #TDOD: should I sum multiple edges? Do they ever happen?
    G.simplify(combine_edges="sum") #need to specify combine_edges or it erases the weights
    
    #get largest connected compornent and convert to adjacency matrix
    P, word_index = extract_component(G)
    
    print("starting dense")
    s=time()
    P_dense = P.todense()
    P_dense = l1_numpy(P_dense)
    P_dense = ppmi_numpy(P_dense)
    P_dense = l1_numpy(P_dense)
    P_dense = katz_numpy(P_dense)
    P_dense = ppmi_numpy(P_dense)
    P_dense = l1_numpy(P_dense)
    print(f"dense took {time()-s} seconds")
    
#     print(f"pre-katz density: {P.nnz/(P.shape[0]*P.shape[1])}")
#     
#     print("starting sparse")
#     s=time()
#     #ensure matrix values are probabilities
#     P = l1_normalize(P)
#     P = ppmi(P)
#     P = l1_normalize(P)
#     P = katz_walk(P)
#     P = ppmi(P)
#     P = l1_normalize(P)
#     
#     print(f"sparse took {time()-s} seconds")
#     
#     print(f"post-katz density: {P.nnz/(P.shape[0]*P.shape[1])}")
    P=None
    
    return P, word_index, P_dense


def remove_UK_words(G):
    """
    For compatibility with DeDeyne's implimentation
    """
    
#     brexit_words = set( w.upper() for w in ['aeroplane', 'arse', 'ax', 'bandana', 'bannister', 'behaviour', 'bellybutton', 'centre',
#               'cheque', 'chequered', 'chilli', 'colour', 'colours', 'corn-beef', 'cosy', 'doughnut',
#               'extravert', 'favour', 'fibre', 'hanky', 'harbour', 'highschool', 'hippy', 'honour',
#               'hotdog', 'humour', 'judgment', 'labour', 'light bulb', 'lollypop', 'neighbour',
#               'neighbourhood', 'odour', 'oldfashioned', 'organisation', 'organise', 'paperclip',
#               'parfum', 'phoney', 'plough', 'practise', 'programme', 'pyjamas',
#               'racquet', 'realise', 'recieve', 'saviour', 'seperate', 'theatre', 'tresspass',
#               'tyre', 'verandah', 'whisky', 'WIFI', 'yoghurt','tinfoil','smokey','seat belt','lawn mower',
#               'coca-cola','cell phone','breast feeding','break up','bubble gum','black out'])
    brexit_words = set(['aeroplane', 'arse', 'ax', 'bandana', 'bannister', 'behaviour', 'bellybutton', 'centre',
              'cheque', 'chequered', 'chilli', 'colour', 'colours', 'corn-beef', 'cosy', 'doughnut',
              'extravert', 'favour', 'fibre', 'hanky', 'harbour', 'highschool', 'hippy', 'honour',
              'hotdog', 'humour', 'judgment', 'labour', 'light bulb', 'lollypop', 'neighbour',
              'neighbourhood', 'odour', 'oldfashioned', 'organisation', 'organise', 'paperclip',
              'parfum', 'phoney', 'plough', 'practise', 'programme', 'pyjamas',
              'racquet', 'realise', 'recieve', 'saviour', 'seperate', 'theatre', 'tresspass',
              'tyre', 'verandah', 'whisky', 'WIFI', 'yoghurt','tinfoil','smokey','seat belt','lawn mower',
              'coca-cola','cell phone','breast feeding','break up','bubble gum','black out'])
    
    to_delete = [v.index for v in G.vs if v["name"] in brexit_words]
    G.delete_vertices(to_delete)
    
    return G

if __name__ == "__main__":
    from word_associations.association_readers.xml_readers import SWoW_Dataset
#     sm = csc_matrix(([0,1,2,3,4], ([0,1,2,3,3], [0,1,2,2,3])))
#     
#     print(sm.todense())
#     
#     nm = l2_normalize(sm)
#     print(type(nm))
#     print(nm.todense())
    
    #make SWoW graph (as igraph.Graph)
    
    swow_100 = SWoW_Dataset("D:/datasets/SWoW/SWOW-EN.R100.csv",complete=False, probs=False,response_types="R1").get_all_associations()
    from time import time
#     s=time()
    katz_sparse, word_index, katz_dense = generate_katz_walk(swow_100)
#     print(f"took {time()-s}s")
    
#     save_npz("katz_r1_sparse.npz",katz_sparse)
    np.save("katz_r1_dedeyne.npy", katz_dense)
    import pickle
    with open('word_index_r1_dedeyne.pkl', "wb") as f:
        pickle.dump(word_index, f)
        
    
    
    
    
#     xs, ys = map(array, zip(*graph.get_edgelist()))
#     if not graph.is_directed():
#         xs, ys = hstack((xs, ys)).T, hstack((ys, xs)).T
#     else:
#         xs, ys = xs.T, ys.T
#     return coo_matrix((ones(xs.shape), (xs, ys)))