
from itertools import izip
from pulp import *
import random
from itertools import chain
from itertools import product
from itertools import combinations

from wsd_util import *

def getscore(x,y,method, direction):
    """Get similarity score for a method and a direction """
    x = encode_entity(x, method, get_id=False)
    y = encode_entity(y, method, get_id=False)
    return getsim(x,y ,method, direction)
    #return random.random()

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


    
    S = {((u0,u1),(v0,v1)):getscore(C[u0][u1][0],C[v0][v1][0], method, direction) for ((u0,u1),(v0,v1)) in R2}


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
    #print("Status:", LpStatus[prob.status])
    #print("Score:", value(prob.objective))
    ids    = [C[r[0]][r[1]][0] for r in list(itertools.chain(*R1)) if R_vars[r].value() == 1.0]
    titles = ids2title(ids)
    return ids, titles
        
def disambiguate_ilp_2(C, method, direction):
    
    #C = [('a','b','c'), ('e', 'f', 'g'), ('h', 'i')]

    #R1 = [zip(["R"z]*len(c),zip([i]*len(c), range(len(c)))) for i,c in enumerate(C)]
    R1 = [zip(['R']*len(c),[i]*len(c), range(len(c))) for i,c in enumerate(C)]
    Q = [zip(['Q']*len(c),[i]*len(c), range(len(c))) for i,c in enumerate(C)]

    #R1 = {r:str(r) for r in itertools.chain(*RI1)}
    #R1 = [[str(rij) for rij in ri] for ri in RI1]

    #RI1_flat = list(itertools.chain(*RI1))


    R2=[]
    for e in combination(R1,2):
            R2 += [('R',(i,k),(j,l)) for (_,i,k),(_,j,l) in itertools.product(e[0], e[1]) ]        


    #R2 = {r:str(r) for r in RI2}



    S = {('R',(i,k),(j,l)):getscore(C[i][k][0],C[j][l][0], method, direction) for _,(i,k),(j,l) in R2}


    prob = LpProblem("wsd", LpMaximize)

    R=list(itertools.chain(*R1)) + R2
    Q=list(itertools.chain(*Q))
    R_vars = LpVariable.dicts("R",R,
                                lowBound = 0,
                                upBound = 1,
                                cat = pulp.LpInteger)

    Q_vars = LpVariable.dicts("Q",Q,
                                lowBound = 0,
                                upBound = 1,
                                cat = pulp.LpInteger)

    prob += lpSum([S[r]*R_vars[r] for r in R2])


    i=0
    for ri in R1:
        prob += lpSum([R_vars[rij] for rij in ri])==1, ("R1 %s constraint")%i
        i += 1

    prob += lpSum(Q_vars.values())==1, ("Q constraint")


    for _,(i,k),(j,l) in R2:
        prob += lpSum([R_vars[('R',i,k)],R_vars[('R',j,l)],-2*R_vars[('R',(i,k),(j,l))]]) >=0, ("R_%s_%s constraint"%((i,k),(j,l)))
        prob += lpSum([Q_vars[('Q',i,k)],Q_vars[('Q',j,l)], -1*R_vars[('R',(i,k),(j,l))]]) >=0, ("Q_%s_%s constraint"%((i,k),(j,l)))

    prob.solve() 
#     print("Status:", LpStatus[prob.status])
#     print("Score:", value(prob.objective))

#     for (_,i,k), q in Q_vars.items():
#         if q.value()==1:
#             print("central concept: ", id2title(C[i][k][0]))
    ids    = [C[i][k][0] for _,i,k in list(itertools.chain(*R1)) if R_vars[('R',i,k)].value() == 1.0]
    
    
    titles = ids2title(ids)
    return ids, titles

# key
def evalkey(c, a, candslist, simmatrix):
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

def key_quad(candslist, method, direction):
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

# Parallel Keyquad
from functools import partial
from multiprocessing import Pool as ThreadPool 
def Pevalkey((c, a), candslist, simmatrix):
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
def Pkey_quad(candslist, method, direction):
    res_all=[]
    simmatrix = get_sim_matrix(candslist, method, direction)
    pool = ThreadPool(25) 
    
    partial_evalkey = partial(Pevalkey, candslist=candslist, simmatrix=simmatrix)
    I=[[j]*len(candslist[j]) for j in range(len(candslist))]
    
    res_all= pool.map(partial_evalkey, zip(itertools.chain(*candslist), itertools.chain(*I)))
    pool.close() 
    pool.join() 
    
    res, score = max(res_all, key=lambda x: x[1])
    titles = ids2title(res)
    return res, titles