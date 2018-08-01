"""Diiferent coherence (context, key-entity) calculation, and 
    disambiguation.
"""
from itertools import izip
from pulp import *
import random
from itertools import chain
from itertools import product
from itertools import combinations

from wsd_util import *

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 215, The Wikisim Project"
__credits__ = ["Armin Sajadi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Armin Sajadi"
__email__ = "sajadi@cs.dal.ca"
__status__ = "Development"



def get_sim_matrix(candslist, method, direction):
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
            sims[c1[0]][c2[0]]= sims[c2[0]][c1[0]] = getsim(encode_entity(c1[0], method, get_id=False),
                                                            encode_entity(c2[0], method, get_id=False),
                                                            method, direction)
    return sims     

# ILP Coherence
def disambiguate_ilp(C, method, direction):
    """ Disambiguate using ILP 
        Inputs: 
            C: Candidate List [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
            method: Similarity method
            direction: embedding direction"""
    #C = [('a','b','c'), ('e', 'f', 'g'), ('h', 'i')]

    R1 = [zip([i]*len(c), range(len(c))) for i,c in enumerate(C)]

    #R1 = {r:str(r) for r in itertools.chain(*RI1)}
    #R1 = [[str(rij) for rij in ri] for ri in RI1]

    #RI1_flat = list(itertools.chain(*RI1))


    R2=[]
    for e in combination(R1,2):
        R2 += [r for r in itertools.product(e[0], e[1]) ]        


    #R2 = {r:str(r) for r in RI2}


    simmatrix = get_sim_matrix(C, method, direction)
    
    S = {((u0,u1),(v0,v1)):simmatrix[C[u0][u1][0]][C[v0][v1][0]] for ((u0,u1),(v0,v1)) in R2}


    prob = LpProblem("wsd", LpMaximize)

    R=list(itertools.chain(*R1)) + R2
    R_vars = LpVariable.dicts("R",R,
                                lowBound = 0,
                                upBound = 1,
                                cat = pulp.LpInteger)
    prob += lpSum([S[r]*R_vars[r] for r in R2])


    i=0
    for ri in R1:
        prob += lpSum([R_vars[rij] for rij in ri])==1, ("R1 %s constraint")%i
        i += 1


    for r in R2:
        prob += lpSum([R_vars[r[0]],R_vars[r[1]],-2*R_vars[r]]) >=0, ("R_%s_%s constraint"%(r[0], r[1]))

    prob.solve() 
    ids    = [C[r[0]][r[1]][0] for r in list(itertools.chain(*R1)) if R_vars[r].value() == 1.0]
    titles = ids2title(ids)
    return ids, titles

# key-entity based methods
def evalkey(c, a, candslist, simmatrix):
    ''' Calculated the coherence, given a key entity
        Input: 
                c: a possible key-entity
                a: index of the mention whose candidate is considered to be key entity
                simmatrix: Similarity Matrix
        Output:
            resolved: the resolved entitieis
            score: Coherence, using this entity
    '''
    resolved=[]
    score=0;
    for i in  range(len(candslist)):
        if a==i:
            resolved.append(c[0])
            continue
        cands = candslist[i]
        vb = [(cj[0], simmatrix[c[0]][cj[0]])  for cj in cands]
        max_concept, max_sc = max(vb, key=lambda x: x[1])
        score += max_sc
        resolved.append(max_concept)
    return resolved,score

def key_quad(candslist,  method, direction):
    '''Disambiguate using search-based (quadratic) key-entity method
        Input:
            candslist: The list of candidates
            method: Similarity method
            direction: The graph direction
        Output: 
            Resolved entities, and resolved titles
    '''
    res_all=[]
    simmatrix = get_sim_matrix(candslist, method, direction)

    for i in range(len(candslist)):
        for j in range(len(candslist[i])):
            res_ij =  evalkey(candslist[i][j], i, candslist, simmatrix)
            res_all.append(res_ij)
    res, score = max(res_all, key=lambda x: x[1])
    #print("Score:", score)
    titles = ids2title(res)
    return res, titles


def disambiguate(C, method, direction, op_method):
    """ Disambiguate C list using a disambiguation method 
        Inputs:
            C: Candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
            method: disambiguation method 
                        most important ones: ilp (integer linear programming), 
                                             key: Key Entity based method
        
    """
    if op_method == 'ilp':
        return disambiguate_ilp(C, method, direction)
    if op_method == 'keyq':
        return key_quad(C, method, direction)    
    return None

def disambiguate_driver(C, ws, method='rvspagerank', direction=DIR_BOTH, op_method="keyq"):
    """ Initiate the disambiguation by chunking the sentence 
        Disambiguate C list using a disambiguation method 
        Inputs:
            C: Candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
            ws: Windows size for chunking
            method: similarity method
            direction: embedding type
            op_method: disambiguation method 
                        most important ones: ilp (integer linear programming), 
                                             keyq: Key Entity based method
        
    """
    #TODO: modify this chunking to an overlapping version
    if ws == 0: 
        return  disambiguate(C, method)
    
    ids = []
    titles = []
    
    windows = [[start, min(start+ws, len(C))] for start in range(0,len(C),ws) ]
    last = len(windows)
    if last > 1 and windows[last-1][1]-windows[last-1][0]<2:
        windows[last-2][1] = len(C)
        windows.pop()
        
    for w in windows:
        chunk_c = C[w[0]:w[1]]
        
        chunk_ids, chunk_titles = disambiguate(chunk_c, method, direction, op_method)
        ids += chunk_ids
        titles += chunk_titles
    return ids, titles     