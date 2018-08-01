"""A few general modules for disambiguation
"""
from __future__ import division

import sys, os
from itertools import chain
from itertools import product
from itertools import combinations
import unicodedata

dirname = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(dirname, '..'))
from wikisim.config import *
from wikisim.calcsim import *
from requests.packages.urllib3 import Retry

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 215, The Wikisim Project"
__credits__ = ["Armin Sajadi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Armin Sajadi"
__email__ = "sajadi@cs.dal.ca"
__status__ = "Development"

dirname = os.path.dirname(__file__)
MODELDIR = os.path.join(dirname, "../models")

session = requests.Session()
http_retries = Retry(total=20,
                backoff_factor=.1)
http = requests.adapters.HTTPAdapter(max_retries=http_retries)
session.mount('http://', http)


def generate_candidates(S, M, max_t=20, enforce=False):
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


def solr_escape(s):
    """
        Escape a string for solr
    """
    #ToDo: probably && and || nead to be escaped as a whole, and also AND, OR, NOT are not included
    to_sub=re.escape(r'+-&&||!(){}[]^"~*?:\/')
    return re.sub('[%s]'%(to_sub,), r'\\\g<0>', s)

def solr_unescape(s):
    """
        Escape a string for solr
    """
    #ToDo: probably && and || nead to be escaped as a whole, and also AND, OR, NOT are not included
    to_sub=re.escape(r'+-&&||!(){}[]^"~*?:\/')
    return re.sub('\\\([%s])'%(to_sub,), r'\g<1>', s)

def throw_unicodes(inputstr):
    '''This function "ideally" should prepare the text in the correct encoding
        which is utf-16, but I couldn't (cf. my encoding notes)
        so for know, just make everything ascii!
        Input: 
            A unicode string with any encoding
        Output: 
            Ascii encoded string
    '''
    if isinstance(inputstr, str):
        return inputstr
    log('[throw_unicodes]\t Encoded to ascii')
    return unicodedata.normalize('NFKD', inputstr).encode('ascii', 'ignore')


#Evaluation Methods
def get_tp(ids, gold_titles):
    """Returns true positive number in id, compared to gold_titles 
        this function is used to evaluate WSD
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

from difflib import SequenceMatcher
def strsimilar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_sentence_measures(S, M, S_gold, M_gold, wsd_measure=False):
    ''' Calcuates precision/recall/F1 for mention detection/wsd for a given sentence
        Input:
            S_gold: The correct tokenized sentence 
            M_gold: The correct mention
            S: The given sentence to evaluate
            M: The given mentions to evaluate
            wsd_measure: if True, it returns the wikifying measures,
                        if false, returns the measures of the mention dection  prcess
        Output:
            precision, recall, f-measure
            
    '''
    
    S_gold = [throw_unicodes(s.replace(' ','')) for s in S_gold]    
    Sgi=[]
    Mgi=[]
    last_index=0
    for s in S_gold:
        Sgi.append ([last_index, last_index+len(s)])
        last_index += len(s)
    Mgi = [Sgi[m[0]] for m in M_gold]    
    
    S = [throw_unicodes(s.replace(' ','')) for s in S]    
    Sj=[]
    last_index=0
    for s in S:
        Sj.append ([last_index, last_index+len(s)])
        last_index += len(s)
    Mj = [Sj[m[0]] for m in M]    
    
    i=0
    j=0
    tp=fp=fn=0
    
    
    while True:
        if i >= len(Mgi):
            fp += (len(Mj)-j)
            break
            
        if j >= len(Mj):
            fn += (len(Mgi)-i)
            break
            
        if (Mgi[i][1] <= Mj[j][0]) or ((Mgi[i][1] <= Mj[j][1]) and strsimilar(S_gold[M_gold[i][0]], S[M[j][0]])<0.5):
            fn += 1
            i += 1
            continue
            
        if  (Mgi[i][0] >= Mj[j][1]) or ((Mgi[i][0] <= Mj[j][1]) and strsimilar(S_gold[M_gold[i][0]], S[M[j][0]])<0.5):
            fp += 1
            j += 1
            continue

        if wsd_measure:
            if title2id(M_gold[i][1]) != title2id(M[j][1]):
                fp += 1
                i += 1
                j += 1
                continue
        #print "match:%s, %s (%s, %s) " % (Mgi[i], Mj[j], S_gold[M_gold[i][0]], S[M[j][0]])
        tp +=1
        i += 1
        j += 1
        
    return tp, fp, fn   
            
def get_overall_measures(tp_list):
    """Returns micro/macro measures, given a list of (tp, fp, fn)
       Inputs: a list of (tp, fp, fn) tuples
       Output: macro_prec, macro_rec, macro_f1, micro_prec, micro_rec, micro_f1
    """
    overall_tp = overall_fp = overall_fn = 0
    macro_prec = macro_rec = macro_f1 = 0;
    for tp, fp, fn in tp_list:

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

        prec = float(tp)/(tp+fp) if tp+fp > 0 else 0
        rec  = float(tp)/(tp+fn) if tp+fn > 0 else 0
        macro_prec += prec
        macro_rec  += rec
        macro_f1   += 2*(prec * rec)/(prec + rec) if (prec + rec)>0 else 0
        
    macro_prec = macro_prec/len(tp_list)
    macro_rec = macro_rec/len(tp_list)
    macro_f1 = macro_f1/len(tp_list)

    micro_prec =  float(overall_tp) / (overall_tp + overall_fp)
    micro_rec  =  float(overall_tp) / (overall_tp + overall_fn)
    micro_f1   = 2*(micro_prec * micro_rec)/(micro_prec + micro_rec)
    
    return macro_prec, macro_rec, macro_f1, micro_prec, micro_rec, micro_f1
     

        