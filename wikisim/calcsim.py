"""Calculating Relatedness."""
# uncomment

from __future__ import division

from embedding import *
#from collections import defaultdict
import json
import math
from scipy import stats
import requests
from config import *

#from utils import * # uncomment

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 215, The Wikisim Project"
__credits__ = ["Armin Sajadi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Armin Sajadi"
__email__ = "sajadi@cs.dal.ca"
__status__ = "Development"

#constants


def _unify_ids_scores(*id_sc_tuple):
    uids, id2in = e2i(*(ids for ids, _ in id_sc_tuple));
    
    uscs=tuple();            
    for ids,scs in id_sc_tuple:
        scs_u=sp.zeros(len(id2in))
        scs_u[[id2in[wid] for wid in ids]] = scs;            
        uscs += (scs_u,)                
    return uids, uscs       

def get_solr_count(s):
    """ Gets the number of documents the string occurs 
        NOTE: Multi words should be quoted
    Arg:
        s: the string (can contain AND, OR, ..)
    Returns:
        The number of documents
    """
    q='+text:(%s)'%(s,)
    qstr = 'http://localhost:8983/solr/enwiki20160305/select'
    params={'indent':'on', 'wt':'json', 'q':q, 'rows':0}
    r = requests.get(qstr, params=params)
    D = r.json()['response']
    return D['numFound']

def getsim_ngd(term1,term2):
    """ Calculates Normalized Google Distance (ngd) similarity between two concepts 
    Arg:
        id1, id2: the two concepts 
    Returns:
        The similarity score        
    """
    
    f1=get_solr_count('"%s"'%(term1,))
    
    f2=get_solr_count('"%s"'%(term2,))
    f12=get_solr_count('"%s" AND "%s"'%(term1, term2))
    
    if (f1==0) or (f2==0) or (f12==0):
        return 0;
    dist = (sp.log(max(f1,f2))-sp.log(f12))/(sp.log(WIKI_SIZE)-sp.log(min(f1,f2)));
    sim = 1-dist if dist <=1 else 0
    return sim
        
def getsim_word2vec(id1, id2):
    """ Calculates wor2vec similarity between two concepts 
    Arg:
        id1, id2: the two concepts 
    Returns:
        The similarity score        
    """
    model  = getword2vec_model()
    if model is None:
        log('[getsim_word2vec]\tmodel not loaded')
        raise Exception('model not loaded, try gensim_loadmodel()')
        
    if id1 not in model.vocab:
        #print '%s,%s skipped, %s not in vocab ' % (id1, id2, id1)
        return 0
    if id2 not in model.vocab:
        #print '%s,%s skipped, %s not in vocab ' % (id1, id2, id2)
        return 0
    return model.similarity(id1, id2)


def getsim_wlm(id1, id2):
    """ Calculates wlm (ngd) similarity between two concepts 
    Arg:
        id1, id2: the two concepts 
    Returns:
        The similarity score        
    """
    in1 = set(getlinkedpages(id1, DIR_IN))
    in2 = set(getlinkedpages(id2, DIR_IN))
    f1 = len(in1)
    f2 = len(in2)
    f12=len(in1.intersection(in2))
    if (f1==0) or (f2==0) or (f12==0):
        return 0;
    dist = (sp.log(max(f1,f2))-sp.log(f12))/(sp.log(WIKI_SIZE)-sp.log(min(f1,f2)));
    sim = 1-dist if dist <=1 else 0
    return sim

def getsim_cocit(id1, id2):
    """ Calculates co-citation similarity between two concepts 
    Arg:
        id1, id2: the two concepts 
    Returns:
        The similarity score        
    """
    in1 = set(getlinkedpages(id1, DIR_IN))
    in2 = set(getlinkedpages(id2, DIR_IN))
    f1 = len(in1)
    f2 = len(in2)
    if (f1==0) or (f2==0):
        return 0;
    
    f12=len(in1.intersection(in2))
    sim = (f12)/(f1+f2-f12);
    return sim


def getsim_coup(id1, id2):
    """ Calculates coupler similarity between two concepts 
    Arg:
        id1, id2: the two concepts 
    Returns:
        The similarity score        
    """
    in1 = set(getlinkedpages(id1, DIR_OUT))
    in2 = set(getlinkedpages(id2, DIR_OUT))
    f1 = len(in1)
    f2 = len(in2)
    if (f1==0) or (f2==0):
        return 0;
    
    f12=len(in1.intersection(in2))
    sim = (f12)/(f1+f2-f12);
    return sim

def getsim_ams(id1, id2):
    """ Calculates amlser similarity between two concepts 
    Arg:
        id1, id2: the two concepts 
    Returns:
        The similarity score        
    """
    in1 = set(getlinkedpages(id1, DIR_IN))
    out1 = set(getlinkedpages(id1, DIR_OUT))
    link1 = in1.union(out1)
    
    in2 = set(getlinkedpages(id2, DIR_IN))
    out2 = set(getlinkedpages(id2, DIR_OUT))
    link2 = in2.union(out2)
    
    f1 = len(link1)
    f2 = len(link2)
    if (f1==0) or (f2==0):
        return 0;
    
    f12=len(link1.intersection(link2))
    sim = (f12)/(f1+f2-f12);
    return sim


def getsim_emb(id1,id2, direction):
    """ Calculates the similarity between two concepts
    Arg:
        id1, id2: the two concepts
        direction: 0 for in, 1 for out, 2 for all
        
    Returns:
        The similarity score
    """
    em1 = concept_embedding(id1, direction);
    em2 = concept_embedding(id2, direction);
    if em1.empty or em2.empty:
        return 0;
    
    em1, em2 = em1.align(em2, fill_value=0)
#     print em1
#     print em2
    return 1-sp.spatial.distance.cosine(em1.values,em2.values);

def encode_entity(term, method,get_id=True):    
    if method in {'ngd'}:
        return term
    if get_id:
        term = title2id(term)
    if term is None:
        return None
    if method in {'rvspagerank','wlm','cocit','coup','ams'} :
        return term
    if 'word2vec_id' in method:
        term = 'id_' + str(term)
        return term
    if 'word2vec':
        term = str(term)
        return term
    
    return term

def getsim(id1,id2, method='rvspagerank', direction=DIR_BOTH, sim_method=None):
    """ Calculates well-known similarity metrics between two concepts 
    Arg:
        id1, id2: the two concepts 
        method:
            wlm: Wikipedia-Miner method
            cocit: cocitation
            coup: coupling
            ams: amsler
            rvspagerank: ebedding based similarity (in our case, 
                 reversed-page rank method)
    Returns:
        The similarity score        
    """
    log('[getsim started]\method = %s, direction = %s, id1=%s, id2=%s', method, direction, id1, id2)
    
    if method=='rvspagerank':
        sim = getsim_emb(id1,id2, direction)
    elif method=='wlm':
        sim = getsim_wlm(id1,id2)
    elif method=='cocit':
        sim = getsim_cocit(id1,id2)
    elif method=='coup':
        sim = getsim_coup(id1,id2)
    elif method=='ams':
        sim = getsim_ams(id1,id2)
    elif 'word2vec' in  method:
        sim = getsim_word2vec(id1, id2)
    elif 'ngd' in  method:
        sim = getsim_ngd(id1, id2)
    elif sim_method is not None:    
        sim = sim_method(id1,id2)
    else:
        sim=None
    log('[getsim]\tfinished')
    return sim

    
def getsim_file(infilename, outfilename, method='rvspagerank', direction=DIR_BOTH, sim_method=None):
    """ Batched (file) similarity.
    
    Args: 
        infilename: tsv file in the format of pair1    pair2   [goldstandard]
        outfilename: tsv file in the format of pair1    pair2   similarity
        direction: 0 for in, 1 for out, 2 for all
                        
    Returns:
        vector of scores, and Spearmans's correlation if goldstandard is given
    """
    log('[getsim_file started]\t%s -> %s', infilename, outfilename)
    outfile = open(outfilename, 'w');
    dsdata=readds(infilename);
    gs=[];
    scores=[];
    spcorr=None;
    for row in dsdata.itertuples():   
        log('processing %s, %s', row[1], row[2])
        if (row[1]=='null') or (row[2]=='null'):
            continue;
        if len(row)>3: 
            gs.append(row[3]);
            
        term1 = encode_entity(row[1], method, get_id=True)
        term2 = encode_entity(row[2], method, get_id=True)
            
        if (term1 is None) or (term2 is None):
            sim=0;
        else:
            sim=getsim(term1, term2, method, direction, sim_method);
        outfile.write("\t".join([str(row[1]), str(row[2]), str(sim)])+'\n')
        scores.append(sim)
    outfile.close();
    if gs:
        spcorr = sp.stats.spearmanr(scores, gs);
    log('[getsim_file]\tfinished')
    return scores, spcorr

    

def getembed_file(infilename, outfilename, direction, get_titles=False, cutoff=None):
    """ Batched (file) concept representation.
    
    Args: 
        infilename: tsv file in the format of pair1    pair2   [goldstandard]
        outfilename: tsv file in the format of pair1    pair2   similarity
        direction: 0 for in, 1 for out, 2 for all
        titles: include titles in the embedding (not needed for mere calculations)
        cutoff: the first top cutoff dimensions (None for all)        

    """
    
    log('[getembed_file started]\t%s -> %s', infilename, outfilename)
    outfile = open(outfilename, 'w');
    dsdata=readds(infilename, usecols=[0]);
    scores=[];
    for row in dsdata.itertuples():        
        wid = encode_entity(row[1], method='rvspagerank', get_id=True)
        if wid is None:
            em=pd.Series();
        else:
            em=conceptrep(wid, method='rvspagerank', direction = direction, 
                          get_titles = get_titles, cutoff=cutoff)
        outfile.write(row[1]+"\t"+em.to_json()+"\n")
    outfile.close();
    log('[getembed_file]\tfinished')
