import sys
from itertools import chain
from itertools import product
from itertools import combinations


sys.path.insert(0,'..')
from wikisim.config import *

from wikisim.calcsim import *

def generate_candidates(S, M, max_t=10, enforce=False):
    """ Given a sentence list (S) and  a mentions list (M), returns a list of candiates
        Inputs:
            S: segmented sentence [w1, ..., wn]
            M: mensions [m1, ... , mj]
            max_t: maximum candiate per mention
            enforce: Makes sure the "correct" entity is among the candidates
        Outputs:
         Candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
             where cij is the jth candidate for ith mention and pij is the relative frequency of cij
    
    """
    candslist=[]
    for m in M:
        
        clist = anchor2concept(S[m[0]])
        if not clist:
            clist=((0L,1L),)
        
        clist = sorted(clist, key=lambda x: -x[1])
        clist = clist[:max_t]
        
        smooth=0    
        if enforce:          
            wid = title2id(m[1])            
    #         if wid is None:
    #             raise Exception(m[1].encode('utf-8') + ' not found')
            
                        
            trg = [(i,(c,f)) for i,(c,f) in enumerate(clist) if c==wid]
            if not trg:
                trg=[(len(clist), (wid,0))]
                smooth=1

                
            if smooth==1 or trg[0][0]>=max_t: 
                if clist:
                    clist.pop()
                clist.append(trg[0][1])
            
        s = sum(c[1]+smooth for c in clist )        
        clist = [(c,float(f+smooth)/s) for c,f in clist ]
            
        candslist.append(clist)
    return  candslist 

def disambiguate_popular(C):
    '''Disambiguate using only popularity
        Input:
            C: Candlist
        Output:
            Disambiguated entities
    '''
    
    ids = [c[0][0] for c in C ]
    titles= ids2title(ids)
    return ids, titles

def get_sim_matrix(candslist,method, direction):
    ''' Computes a matrix of pairwise similarities
        Input: 
            candslist: The list of candidates
            method: Similarity Method
        Output:
            The similarity matrix matrix
    '''
    concepts=  list(chain(*candslist))
    concepts=  list(set(c[0] for c in concepts))
    sims = pd.DataFrame(index=concepts, columns=concepts)
    for cands1,cands2 in combinations(candslist,2):
        for c1,c2 in product(cands1,cands2):
            sims[c1[0]][c2[0]]= sims[c2[0]][c1[0]] = getsim(c1[0],c2[0] , method, direction)
    return sims     

def get_tp(gold_titles, ids):
    """Returns true positive number
       Inputs: goled_titles: The correct titles
               ids: The given ids
       Outputs: returns a tuple of (true_positives, total_number_of_ids)
    
    """
    tp=0
    for m,id2 in zip(gold_titles, ids):
        if title2id(m[1]) == id2:
            tp += 1
    return [tp, len(ids)]

def get_prec(tp_list):
    """Returns precision
       Inputs: a list of (true_positive and total number) lists
       Output: Precision
    """
    
    overall_tp = 0
    simple_count=0
    overall_count=0
    macro_prec = 0;
    for tp, count in tp_list:
        if tp is None:
            continue
        simple_count +=1    
        overall_tp += tp
        overall_count += count
        macro_prec += float(tp)/count
        
    macro_prec = macro_prec/simple_count
    micro_prec = float(overall_tp)/overall_count
    
    return micro_prec, macro_prec

