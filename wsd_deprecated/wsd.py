""" Evaluating the method on Semantic Relatedness Datasets."""


import sys
import os
import time;
import json 
import requests

import numpy as np
        


from wsdcoherence import *
from wsdvsm import *

#reopen()


  

def disambiguate(C, method, direction, op_method):
    """ Disambiguate C list using a disambiguation method 
        Inputs:
            C: Candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
            method: similarity method
            direction: embedding type
            op_method: disambiguation method 
                        most important ones: ilp (integer linear programming), 
                                             key: Key Entity based method
        
    """
    if op_method == 'popularity':
        return disambiguate_popular(C)
    if op_method == 'ilp':
        return disambiguate_ilp(C, method, direction)
    if op_method == 'ilp2':
        return disambiguate_ilp_2(C, method, direction)
    if op_method == 'keyq':
        return key_quad(C, method, direction)
    if op_method == 'pkeyq':
        return Pkey_quad(C, method, direction)
    if  op_method == 'simplecontext'  :
        return simple_entity_context_disambiguate(C, direction, method)
    if  op_method == 'context2'  :
        return contextdisamb_2(C, direction)
    if  op_method == 'context3'  :
        return contextdisamb_3(C, direction)
    if  op_method == 'entitycontext'  :
        return entity_context_disambiguate(C, direction, method)

        
    if  op_method == 'context4_1'  :
        return keyentity_disambiguate(C, direction, method, 1)
    if  op_method == 'context4_2'  :
        return keyentity_disambiguate(C, direction, method, 2)
    if  op_method == 'context4_3'  :
        return keyentity_disambiguate(C, direction, method, 3)    
    if  op_method == 'keydisamb'  :
        return keyentity_disambiguate(C, direction, method, 4)
    
    if  op_method == 'tagme'  :
        return tagme(C, method, direction)
    if  op_method == 'tagme2'  :
        return tagme(C, method, direction, True)
    

    
    return None



def disambiguate_driver(C, ws, method='rvspagerank', direction=DIR_BOTH, op_method="keydisamb"):
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
        return  disambiguate(C, method, direction, op_method)
    
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




# Integer Programming
