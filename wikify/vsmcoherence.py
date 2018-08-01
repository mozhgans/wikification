"""Diiferent coherence (context, key-entity) calculation, and 
    disambiguation.
"""
from __future__ import division

from wsd_util import *
import numpy as np

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 215, The Wikisim Project"
__credits__ = ["Armin Sajadi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Armin Sajadi"
__email__ = "sajadi@cs.dal.ca"
__status__ = "Development"


def get_candidate_representations(candslist, direction, method):
    '''returns an array of vector representations. 
       Inputs: 
           candslist: candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
           direction: embedding direction
           method: similarity method
      Outputs
           cvec_arr: Candidate embeddings, a two dimensional array, each column 
                   is the representation of a candidate
           cveclist_bdrs: a list of pairs (beginning, end), to indicate where 
                   the embeddings for a concepts indicates start and end. In other words
                   The embedding of candidates [ci1...cik] in candslist is
                   cvec_arr[cveclist_bdrs[i][0]:cveclist_bdrs[i][1]] 
    '''
    
    cframelist=[]
    cveclist_bdrs = []
    ambig_count=0
    for cands in candslist:
        if len(candslist)>1:
            ambig_count += 1
        cands_rep = [conceptrep(encode_entity(c[0], method, get_id=False), method=method, direction=direction, get_titles=False) for c in cands]
        cveclist_bdrs += [(len(cframelist), len(cframelist)+len(cands_rep))]
        cframelist += cands_rep
        
    cvec_fr = pd.concat(cframelist, join='outer', axis=1)
    cvec_fr.fillna(0, inplace=True)
    cvec_arr = cvec_fr.as_matrix().T
    return cvec_arr, cveclist_bdrs

def entity_to_context_scores(candslist, direction, method):
    ''' finds the similarity between each entity and its context representation
        Inputs:
            candslist: the list of candidates [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
            direction: embedding direction
            method: similarity method
        Returns:
           cvec_arr: Candidate embeddings, a two dimensional array, each column 
           cveclist_bdrs: a list of pairs (beginning, end), to indicate where the 
                   reperesentation of the candidates for cij reside        
           cands_score_list: scroes in the form of [[s11,...s1k],...[sn1,...s1m]]
                    where sij  is the similarity of c[i,j] to to ci-th context
                    
            '''
    cvec_arr, cveclist_bdrs =  get_candidate_representations(candslist, direction, method)    
    
    aggr_cveclist = np.zeros(shape=(len(candslist),cvec_arr.shape[1]))
    for i in range(len(cveclist_bdrs)):
        b,e = cveclist_bdrs[i]
        aggr_cveclist[i]=cvec_arr[b:e].sum(axis=0)
    
    from itertools import izip
    resolved = 0
    cands_score_list=[]        
    for i in range(len(candslist)):
        cands = candslist[i]
        b,e = cveclist_bdrs[i]
        cvec = cvec_arr[b:e]
        convec=aggr_cveclist[:i].sum(axis=0) + aggr_cveclist[i+1:].sum(axis=0)
        S=[]    
        for v in cvec:
            try:
                # We have zero vectors, so this can rais an exception
                # or return none                
                s = 1-sp.spatial.distance.cosine(convec, v);
            except:
                s=0                
            if np.isnan(s):
                s=0
            S.append(s)
        cands_score_list.append(S)

    return cvec_arr, cveclist_bdrs, cands_score_list

def key_criteria(cands_score):
    ''' helper function for find_key_concept: returns a score indicating how good a key is x
        Input:
            scroes for candidates [ci1, ..., cik] in the form of (i, [(ri1, si1), ..., (rik, sik)] ) 
            where (rij,sij) indicates that sij is the similarity of c[i][rij] to to cith context
            
    '''
    if len(cands_score[1])==0:
        return -float("inf")    
    if len(cands_score[1])==1 or cands_score[1][1][1]==0:
        return float("inf")
    
    return (cands_score[1][0][1]-cands_score[1][1][1]) / cands_score[1][1][1]

def find_key_concept(candslist, direction, method):
    ''' finds the key entity in the candidate list
        Inputs:
            candslist: the list of candidates [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
            cvec_arr: the array of all embeddings for the candidates
            cveclist_bdrs: The embedding vector for each candidate: [[c11,...c1k],...[cn1,...c1m]]
        Returns:
            cvec_arr: Candidate embeddings, a two dimensional array, each column 
            cveclist_bdrs: a list of pairs (beginning, end), to indicate where the 
            key_concept: the concept forwhich one of the candidates is the key entity
            key_entity: candidate index for key_cancept that is detected to be key_entity
            key_entity_vector: The embedding of key entity
            '''
    cvec_arr, cveclist_bdrs, cands_score_list = entity_to_context_scores(candslist, direction, method);
    S=[sorted(enumerate(S), key=lambda x: -x[1]) for S in cands_score_list]
        
    key_concept, _ = max(enumerate(S), key=key_criteria)
    key_entity = S[key_concept][0][0]
    
    b,e = cveclist_bdrs[key_concept]
    
    key_entity_vector =  cvec_arr[b:e][key_entity]    
    return cvec_arr, cveclist_bdrs, key_concept, key_entity, key_entity_vector

def keyentity_candidate_scores(candslist, direction, method):
    '''returns entity scores using key-entity scoring 
       Inputs: 
           candslist: candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
           direction: embedding direction
           method: similarity method
           
       Returns:
           Scores [[s11,...s1k],...[sn1,...s1m]] where sij is cij similarity to the key-entity
    '''
    
    cvec_arr, cveclist_bdrs, key_concept, key_entity, key_entity_vector = find_key_concept(candslist, direction, method)
    
    # Iterate 
    candslist_scores=[]
    for i in range(len(candslist)):
        cands = candslist[i]
        b,e = cveclist_bdrs[i]
        cvec = cvec_arr[b:e]
        cand_scores=[]

        for v in cvec:
            try:
                # We have zero vectors, so this can rais an exception
                # or return none                
                d = 1-sp.spatial.distance.cosine(key_entity_vector, v);
            except:
                d=0                
            if np.isnan(d):
                d=0
            
            cand_scores.append(d)    
        candslist_scores.append(cand_scores) 
    return candslist_scores



def coherence_scores_driver(C, ws=5, method='rvspagerank', direction=DIR_BOTH, op_method="keydisamb"):
    """ Assigns a score to every candidate 
        Inputs:
            C: Candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
            ws: Windows size for chunking
            method: similarity method
            direction: embedding type
            op_method: disambiguation method, either keyentity or entitycontext
        Output:
            Candidate Scores
        
    """
    windows = [[start, min(start+ws, len(C))] for start in range(0,len(C),ws) ]
    last = len(windows)
    if last > 1 and windows[last-1][1]-windows[last-1][0]<2:
        windows[last-2][1] = len(C)
        windows.pop()
    scores=[]    
    for w in windows:
        chunk_c = C[w[0]:w[1]]
        if op_method == 'keydisamb':
            scores += keyentity_candidate_scores(chunk_c, direction, method)
            
        if op_method == 'entitycontext':
            _, _, candslist_scores = entity_to_context_scores(chunk_c, direction, method);
            scores += candslist_scores
            
    return scores