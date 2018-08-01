

from wsd_util import *

import numpy as np
from itertools import chain
from itertools import combinations
from itertools import product


def coherence_scores_driver(C, ws, method='rvspagerank', direction=DIR_BOTH, op_method="keydisamb"):
    """ Assigns a score to every candidate 
        Inputs:
            C: Candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
            ws: Windows size for chunking
            method: similarity method
            direction: embedding type
            op_method: disambiguation method, either keyentity or entitycontext
            
        
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
            scores += keyentity_candidate_scores(chunk_c, direction, method,4)
            
        if op_method == 'simplecontext':
            _, _, candslist_scores = simple_entity_to_context_scores(chunk_c, direction, method);
            scores += candslist_scores
            
        if op_method == 'entitycontext':
            _, _, candslist_scores = entity_to_context_scores(chunk_c, direction, method);
            scores += candslist_scores
            
    return scores

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

    #print "ambig_count:", ambig_count
        
    cvec_fr = pd.concat(cframelist, join='outer', axis=1)
    cvec_fr.fillna(0, inplace=True)
    cvec_arr = cvec_fr.as_matrix().T
    return cvec_arr, cveclist_bdrs


def simple_entity_to_context_scores(candslist, direction, method):
    ''' finds the similarity between each entity and its context representation
        Inputs:
            candslist: the list of candidates [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
            cvec_arr: the array of all embeddings for the candidates
            cveclist_bdrs: The embedding vector for each candidate: [[c11,...c1k],...[cn1,...c1m]]
        Returns:
           cvec_arr: Candidate embeddings, a two dimensional array, each column 
           cveclist_bdrs: a list of pairs (beginning, end), to indicate where the 
                   reperesentation of the candidates for cij reside        
           cands_score_list: scroes in the form of [[s11,...s1k],...[sn1,...s1m]]
                    where sij  is the similarity of c[i,j] to to ci-th context
                    
            '''
    cvec_arr, cveclist_bdrs =  get_candidate_representations(candslist, direction, method)    
    convec = cvec_arr.sum(axis=0)
    
    #aggr_cveclist = np.zeros(shape=(len(candslist),cvec_arr.shape[1]))
    #for i in range(len(cveclist_bdrs)):
        #b,e = cveclist_bdrs[i]
        #aggr_cveclist[i]=cvec_arr[b:e].sum(axis=0)
    
    from itertools import izip
    #resolved = 0
    cands_score_list=[]        
    for i in range(len(candslist)):
        cands = candslist[i]
        b,e = cveclist_bdrs[i]
        cvec = cvec_arr[b:e]
        #convec=aggr_cveclist[:i].sum(axis=0) + aggr_cveclist[i+1:].sum(axis=0)
        S=[]    
        for v in cvec:
            try:
                s = 1-sp.spatial.distance.cosine(convec, v);
            except:
                s=0                
            if np.isnan(s):
                s=0
            S.append(s)
        cands_score_list.append(S)

    return cvec_arr, cveclist_bdrs, cands_score_list

def entity_to_context_scores(candslist, direction, method):
    ''' finds the similarity between each entity and its context representation
        Inputs:
            candslist: the list of candidates [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
            cvec_arr: the array of all embeddings for the candidates
            cveclist_bdrs: The embedding vector for each candidate: [[c11,...c1k],...[cn1,...c1m]]
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
                s = 1-sp.spatial.distance.cosine(convec, v);
            except:
                s=0                
            if np.isnan(s):
                s=0
            S.append(s)
        cands_score_list.append(S)

    return cvec_arr, cveclist_bdrs, cands_score_list

def simple_entity_context_disambiguate(candslist, direction=DIR_OUT, method='rvspagerank'):
    '''Disambiguate a sentence using entity-context method
       Inputs: 
           candslist: candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
           direction: embedding direction
           method: similarity method
       Returns: 
           a list of entity ids and a list of titles
    '''
    
        
    _, _, candslist_scores = simple_entity_to_context_scores(candslist, direction, method);
    # Iterate 
    true_entities = []
    for cands, cands_scores in zip(candslist, candslist_scores):
        max_index, max_value = max(enumerate(cands_scores), key= lambda x:x[1])
        true_entities.append(cands[max_index][0])

    titles = ids2title(true_entities)
    return true_entities, titles        

def entity_context_disambiguate(candslist, direction=DIR_OUT, method='rvspagerank'):
    '''Disambiguate a sentence using entity-context method
       Inputs: 
           candslist: candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
           direction: embedding direction
           method: similarity method
       Returns: 
           a list of entity ids and a list of titles
    '''
    
        
    _, _, candslist_scores = entity_to_context_scores(candslist, direction, method);
    # Iterate 
    true_entities = []
    for cands, cands_scores in zip(candslist, candslist_scores):
        max_index, max_value = max(enumerate(cands_scores), key= lambda x:x[1])
        true_entities.append(cands[max_index][0])

    titles = ids2title(true_entities)
    return true_entities, titles        

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

def find_key_concept(candslist, direction, method, ver=4):
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
        
    if ver ==1: 
        key_concept, _ = max(enumerate(S), key=lambda x: x[1][0][1] if len(x[1])>1 else -1)
    elif ver ==2: 
        key_concept, _ = max(enumerate(S), key=lambda x: (x[1][0][1]-x[1][1][1]) if len(x[1])>1 else -1)
    elif ver ==3: 
        key_concept, _ = max(enumerate(S), key=lambda x: (x[1][0][1]-x[1][1][1])/(x[1][0][1]+x[1][1][1]) if len(x[1])>1 else -1)
    elif ver ==4: 
        key_concept, _ = max(enumerate(S), key=key_criteria)
    key_entity = S[key_concept][0][0]
    
    b,e = cveclist_bdrs[key_concept]
    
    key_entity_vector =  cvec_arr[b:e][key_entity]    
    return cvec_arr, cveclist_bdrs, key_concept, key_entity, key_entity_vector


def keyentity_candidate_scores(candslist, direction, method, ver):
    '''returns entity scores using key-entity scoring 
       Inputs: 
           candslist: candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
           direction: embedding direction
           method: similarity method
           ver: 1 for the method explained in the paper
           
       Returns:
           Scores [[s11,...s1k],...[sn1,...s1m]] where sij is cij similarity to the key-entity
    '''
    
        
    cvec_arr, cveclist_bdrs, key_concept, key_entity, key_entity_vector = find_key_concept(candslist, direction, method, ver)
    
    # Iterate 
    candslist_scores=[]
    for i in range(len(candslist)):
        cands = candslist[i]
        b,e = cveclist_bdrs[i]
        cvec = cvec_arr[b:e]
        cand_scores=[]

        for v in cvec:
            try:
                d = 1-sp.spatial.distance.cosine(key_entity_vector, v);
            except:
                d=0                
            if np.isnan(d):
                d=0
            
            cand_scores.append(d)    
        candslist_scores.append(cand_scores) 
    return candslist_scores

def keyentity_disambiguate(candslist, direction=DIR_OUT, method='rvspagerank', ver=4):
    '''Disambiguate a sentence using key-entity method
       Inputs: 
           candslist: candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
           direction: embedding direction
           method: similarity method
           ver: 1 for the method explained in the paper
       Returns: 
           a list of entity ids and a list of titles
    '''
    
        
    candslist_scores = keyentity_candidate_scores (candslist, direction, method, ver)
    # Iterate 
    true_entities = []
    for cands, cands_scores in zip(candslist, candslist_scores):
        max_index, max_value = max(enumerate(cands_scores), key= lambda x:x[1])
        true_entities.append(cands[max_index][0])

    titles = ids2title(true_entities)
    return true_entities, titles        

## Plain context with word2vec

def word_context_candidate_scores (S, M, candslist, ws):
    '''returns entity scores using the similarity with their context
       Inputs: 
           S: Sentence
           M: Mentions
           candslist: candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
            ws: word size
       Returns:
           Scores [[s11,...s1k],...[sn1,...s1m]] where sij is cij similarity to the key-entity
    '''
    
    candslist_scores=[]
    for i in range(len(candslist)):
        cands = candslist[i]
        pos = M[i][0]
        #print "At: ", M[i]
        context = S[max(pos-ws,0):pos]+S[pos+1:pos+ws+1]
        #print context
        #print candslist[i], pos,context
        context_vec = sp.zeros(getword2vec_model().vector_size)
        for c in context:
            #print "getting vector for: " , c
            context_vec += getword2vector(c).as_matrix()
        #print context_vec
        cand_scores=[]

        for c in cands:
            try:
                cand_vector = getentity2vector(encode_entity(c[0],'word2vec', get_id=False))
                d = 1-sp.spatial.distance.cosine(context_vec, cand_vector);
            except:
                d=0                
            if np.isnan(d):
                d=0
            
            cand_scores.append(d)    
        candslist_scores.append(cand_scores) 

    return candslist_scores


        
        
def word_context_disambiguate(S, M, candslist, ws ):
    '''Disambiguate a sentence using word-context similarity
       Inputs: 
           S: Sentence
           M: Mentions
           candslist: candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
           
       Returns: 
           a list of entity ids and a list of titles
    '''
    
        
    candslist_scores = word_context_candidate_scores (S, M, candslist, ws)
                      
    # Iterate 
    true_entities = []
    for cands, cands_scores in zip(candslist, candslist_scores):
        max_index, max_value = max(enumerate(cands_scores), key= lambda x:x[1])
        true_entities.append(cands[max_index][0])

    titles = ids2title(true_entities)
    return true_entities, titles 

######

 

def tagme_vote(c, a, candslist, simmatrix, pop):
    ''' Calculate the vote for an antity
        Input:
            c: The given entity
            a: Index of the mention whose candidate is considered to be key entity
            simmatrix: Similarity Matrix
            pop: Whether or not consider popularity
        Output:
            Returns the vote value

    '''
    v = 0
    for b in  range(len(candslist)):
        if a==b:
            continue
        cands = candslist[b]
        if pop:
            vb = [ci[1]*simmatrix[c[0]][ci[0]] for ci in cands]
        else:
            vb = [simmatrix[c[0]][ci[0]]  for ci in cands]
        vb = sum(vb) / len(vb)
        v += vb    
    return v

def tagme(candslist, method, direction, pop=False):
    ''' The TagMe method for disambiguation
        Input:
            candslist: The list of candidates
            method: The similarity method
            direction: embedding type
            pop: Whether or not consider popularity
        Output:
        Resolved ids and entities
    '''
    res=[]
    simmatrix = get_sim_matrix(candslist, method, direction)
    for i in range(len(candslist)):
        cands = candslist[i]
        
        maxd=-1
        mi=0
        #print len(cands)
        for c in cands:
            d = tagme_vote(c, i, candslist , simmatrix, pop);
            if d>maxd:
                maxd=d
                index=mi
            mi +=1
        res.append(cands[index][0]) 
        #print index,"\n"
    titles = ids2title(res)
    return res, titles


    