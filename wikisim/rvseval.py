""" Evaluating the method on Semantic Relatedness Datasets."""

#%load_ext autoreload
#%autoreload

import os
import time;
import pandas as pd


#%aimport wikipedia
#%aimport calcsim
from config import *
from calcsim import *
import gensim

import functools

#print DISABLE_CACHE
#clearcache()
direction=DIR_BOTH;
#method = 'word2vec.300.orig'
#method = 'word2vec.300.ehsan'
#method = 'word2vec.graph.500'
#method = 'word2vec.500'
#method = 'word2vec.500.noneg'
method = 'rvspagerank'
#method = 'ngd'
    
#word2vec_path = os.path.join(home, 'backup/wikipedia/WikipediaClean5Negative300Skip10.Ehsan/WikipediaClean5Negative300Skip10')
#word2vec_path = os.path.join(home, '/users/grad/sajadi/backup/wikipedia/20160305/embed/word2vec.enwiki-20160305-replace_surface.1.0.500.10.5.15.5.5/word2vec.enwiki-20160305-replace_surface.1.0.500.10.5.15.5.5')
#word2vec_path = os.path.join(home, '/users/grad/sajadi/backup/wikipedia/20160305/embed/word2vec.enwiki-20160305-cleantext.1.1.300.10.5.20.5.5/word2vec.enwiki-20160305-cleantext.1.1.300.10.5.20.5.5')


#word2vec_path  = '/users/grad/sajadi/backup/wikipedia/20160305/embed/word2vec.enwiki-20160305-pagelinks.dumped.shuf.1.0.500.2.5.20.0.15/word2vec.enwiki-20160305-pagelinks.dumped.shuf.1.0.500.2.5.20.0.15'
#word2vec_path  = '/users/grad/sajadi/backup/wikipedia/20160305/embed/word2vec.enwiki-20160305-pagelinks.dumped.shuf.1.0.500.2.5.15.5.5/word2vec.enwiki-20160305-pagelinks.dumped.shuf.1.0.500.2.5.15.5.5'
#word2vec_path = '/users/grad/sajadi/backup/wikipedia/20160305/embed/word2vec.enwiki-20160305-pagelinks.dumped.shuf.1.0.500.2.5.20.10.15/word2vec.enwiki-20160305-pagelinks.dumped.shuf.1.0.500.2.5.20.10.15'
#word2vec_path = '/users/grad/sajadi/backup/wikipedia/20160305/embed/word2vec.enwiki-20160305-pagelinks.dumped.shuf.1.0.500.2.5.25.20.20/word2vec.enwiki-20160305-pagelinks.dumped.shuf.1.0.500.2.5.25.20.20'



initdirs(method, direction)
resfilename =  os.path.join(baseresdir, 'reslog.txt')

dsfiles=('MC_28-edited.csv', 'RG-edited.csv', 'wsim353-edited.csv', 'Kore-edited.csv', 'MiniMayoSRS-edited.csv', 
         'MayoSRS-edited.csv', 'UMNSRS_relatedness-edited.csv', 'UMNSRS_similarity-edited.csv')

# dsfiles=('MC_28.orig.lower.csv', 'RG.orig.lower.csv', 'wsim353.orig.lower.csv', 'Kore-edited.csv','MiniMayoSRS.orig.lower.csv', 
#          'MayoSRS.orig.lower.csv', 'UMNSRS_relatedness.orig.lower.csv', 'UMNSRS_similarity.orig.lower.csv')

dsfiles=('MC_28-edited.csv',)
#dsfiles=('MC_28.orig.lower.csv',)

if 'word2vec' in method:
    wmodel = gensim_loadmodel(word2vec_path)
    method += '.' + str(wmodel.negative)
for dsname in dsfiles:
    start = time.time()
    
    printflush ("Processing",dsname)
    dsbase, dsext = os.path.splitext(dsname);
    infilename = os.path.join(dsdir, dsname)
    outfilename = os.path.join(resdir(method, direction), dsbase+ '.out'+ dsext)
    _ , corr = getsim_file(infilename, outfilename, method, direction);
    logres(resfilename, '%s\t%s\t%s\t%s\t%s', method, dsname, graphtype(direction), corr.correlation
                        , corr.pvalue)
    print corr
    print str(timeformat(int(time.time()-start)));
    
#close()   