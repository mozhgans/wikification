""" Config file.
"""
#%load_ext autoreload
#%autoreload

import os.path
import sys
import datetime
#import logging
from wikipedia import *


home = os.path.expanduser("~");
dsdir = os.path.join("../datasets/similarity");
#dsdir = os.path.join(home ,"backup/projects/wikisim/datasets/similarity.orig");
workingdir = os.path.join(home , 'backup/tmp/');
baseresdir = path = os.path.join(workingdir, 'results')


#logging.basicConfig(filename=os.path.join(workingdir,'myapp.log'), level=logging.INFO);    

    
def resdir(hitsver, direction=None):   
    path = os.path.join(baseresdir , graphtype(direction), hitsver);
    if not os.path.exists(path):
        os.makedirs(path)
    return path

    
def tmpdir(direction, hitsver):
    return os.path.join(resdir(hitsver, direction),'tmp/');

def graphdir(direction):    
    return os.path.join(getworkingdir , 'graphs' , wikisim.graphtypestr(direction));

def initdirs(hitsver, direction):
    path = resdir(hitsver, direction)
    if not os.path.exists(path):
        os.makedirs(path)
    path = tmpdir(direction, hitsver)
    if not os.path.exists(path):
        os.makedirs(path)
def printflush(*str):
    print str
    sys.stdout.flush()
    
    
def graphtype(direction):
    if direction is None:
        return ''
    if direction == DIR_IN:
        return 'in'
    if direction == DIR_OUT:
        return 'out'
    if direction == DIR_BOTH:
        return 'both' 