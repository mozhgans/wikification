
from __future__ import division


from wikipedia import * # uncomment
from pagerank import * # uncomment
import gensim

#from utils import * # uncomment

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 215, The Wikisim Project"
__credits__ = ["Armin Sajadi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Armin Sajadi"
__email__ = "sajadi@cs.dal.ca"
__status__ = "Development"

_word2vec_model = None
def getword2vec_model():
    """ returns the word2vec model
    """
    
    return _word2vec_model

def conceptrep(wid, method ='rvspagerank', direction=DIR_BOTH, get_titles=True, cutoff=None):
    """ Calculates well-known similarity metrics between two concepts 
    Arg:
        id1, id2: the two concepts 
        method:
            rvspagerank: rvs-pagerank embedding
            word2vec : wor2vec representation
    Returns:
        The similarity score        
    """
        
    if method =='rvspagerank':
        return conceptrep_rvs(wid, direction, get_titles, cutoff)
    if 'word2vec' in method:
        return getentity2vector(wid)


def concept_embedding(wid, direction):
    """ Calculates concept embedding to be used in relatedness
    
    Args:
        wid: wikipedia id
        direction: 0 for in, 1 for out, 2 for all
        
    Returns:
        The neighbor ids, their scores and the whole neighorhood graph (for visualization purposes)
        
    """
    log('[concept_embedding started]\twid = %s, direction = %s', wid, direction)

    if direction == DIR_IN or direction==DIR_OUT:
        em = _concept_embedding_io(wid, direction)
    if direction == DIR_BOTH:
        em = _concept_embedding_both(wid, direction)
    log('[concept_embedding]\tfinished')
    return em
    
def _concept_embedding_io(wid, direction):
    wid = resolveredir(wid)
    cached_em = checkcache(wid, direction);
    if cached_em is not None:
        return cached_em;

    (ids, links) = getneighbors(wid, direction);
    if ids:
        scores = moler_pagerank_sparse_power(create_csr(links), reverse=True)
        em = pd.Series(scores, index=ids) 
    else:
        em = pd.Series([], index=[])  
    cachescores(wid, em, direction);
    return em
            

def _concept_embedding_both(wid, direction):            
        in_em = _concept_embedding_io(wid, DIR_IN);
        out_em = _concept_embedding_io(wid, DIR_OUT )
        if (in_em is None) or (out_em is None):
            return None;
        return in_em.add(out_em, fill_value=0)/2

def conceptrep_rvs(wid, direction, get_titles=True, cutoff=None):
    """ Finds a representation for a concept
    
        Concept Representation is a vector of concepts with their score
    Arg:
        wid: Wikipedia id
        direction: 0 for in, 1 for out, 2 for all
        titles: include titles in the embedding (not needed for mere calculations)
        cutoff: the first top cutoff dimensions (None for all)
        
    Returns:
        the vecotr of ids, their titles and theirs scores. It also returns the
        graph for visualization purposes. 
    """
    
    log('[conceptrep started]\twid = %s, direction = %s', wid, direction)
    
    em=concept_embedding(wid, direction);    
    if em.empty:
        return em;
    
    
    #ids = em.keys();
    
    if cutoff is not None:
        em = em.sort_values(ascending=False)
        em = em[:cutoff]
    if get_titles:
        em = pd.Series(zip(ids2title(em.index.tolist()), em.values.tolist()), index=em.index)
    log ('[conceptrep]\tfinished')
    return em

def gensim_loadmodel(model_path):
    """ Loads the word2vec model 
    Arg:
        model_path: path to the model
    """
    global _word2vec_model
    log('[getsim_word2vec]\tloading: %s', model_path)
    _word2vec_model = gensim.models.Word2Vec.load(model_path)                
    log('[getsim_word2vec]\loaded')
    return _word2vec_model
    
def getentity2vector(wid):
    if (wid is None) or (wid not in _word2vec_model.vocab):
        return  pd.Series(sp.zeros(_word2vec_model.vector_size))
    return pd.Series(_word2vec_model[wid])

def getword2vector(word):
    if word not in _word2vec_model.vocab:
        return  pd.Series(sp.zeros(_word2vec_model.vector_size))
    return pd.Series(_word2vec_model[word])
    