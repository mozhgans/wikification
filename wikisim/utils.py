"""Utility functions"""
# uncomment

import os
import re
import itertools
import scipy as sp
import pandas as pd
import datetime

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 215, The Wikisim Project"
__credits__ = ["Armin Sajadi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Armin Sajadi"
__email__ = "sajadi@cs.dal.ca"
__status__ = "Development"

dirname = os.path.dirname(__file__)

def readds(url, usecols=None):    
    data = pd.read_table(url, header=None, usecols=usecols)
    return data

DISABLE_LOG=False;

def clearlog(logfile):
    with open(logfile, 'w'):
        pass;

def logres(outfile, instr, *params):
    outstr = instr % params;
    with open(outfile, 'a') as f:
        f.write("[%s]\t%s\n" % (str(datetime.datetime.now()) , outstr));          
        
def log(instr, *params):
    if DISABLE_LOG:
        return
    logres(logfile, instr, *params)
    
outdir = os.path.join(dirname, '../out')    
if not DISABLE_LOG:    
    logfile=os.path.join(outdir, 'log.txt');
    if not os.path.exists(logfile):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        log('log created') 
        os.chmod(logfile, 0777)    
    
    
def timeformat(sec):
    return datetime.timedelta(seconds=sec)

def str2delta(dstr):
    r=re.match(('((?P<d>\d+) day(s?), )?(?P<h>\d+):(?P<m>\d+):(?P<s>\d*\.\d+|\d+)'),dstr)
    d,h,m,s=r.group('d'),r.group('h'),r.group('m'),r.group('s')
    d=int(d) if d is not None else 0
    h,m,s = int(h), int(m), float(s)    
    return datetime.timedelta(days=d, hours=h, minutes=m, seconds=s)