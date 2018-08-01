
import os
from collections import defaultdict
import sys
import time
import requests
import random
import json
import math
from multiprocessing import Pool, Process, Manager 
import functools
import thread
from requests.packages.urllib3 import Retry
sys.path.insert(0,'../..')
home = os.path.expanduser("~");
from wikisim.wikipedia import *

qstr = 'http://localhost:8983/solr/enwiki20160305_context/select'
process_no=25
tr_percent=0.8

down_sample = False
max_anchor = -1
skip_line=0

random.seed(3)
written_sofar=0
example_per_anchor=10

session = requests.Session()
http_retries = Retry(total=20,
                backoff_factor=.1)
http = requests.adapters.HTTPAdapter(max_retries=http_retries)
session.mount('http://localhost:8983/solr', http)

def solr_escape(s):
    return re.sub(r'''['"\\]''', r'\\\g<0>', s)

def get_context(anchor, eid):
    
    params={'wt':'json', 'rows':'50000'}
    anchor = solr_escape(anchor)
    
    q='anchor:"%s" AND entityid:%s' % (anchor, eid)
    params['q']=q
    
#     session = requests.Session()
#     http_retries = Retry(total=20,
#                     backoff_factor=.1)
#     http = requests.adapters.HTTPAdapter(max_retries=http_retries)
#     session.mount('http://localhost:8983/solr', http)
    
    r = session.get(qstr, params=params).json()
    if 'response' not in r: 
        print "[terminating]\t%s",(str(r),)
        sys.stdout.flush()
        os._exit(0)
        
    if not r:
        return []
    return r['response']['docs']

def loadanchors(min_count=5):
    rows = load_table('anchors')
    anchors = defaultdict(list)
    for r in rows:
        if r[2] >= min_count:
            anchors[r[0]].append((r[1], r[2]))        
    return anchors.items()


def mater_anchor((a,l), trq, tsq, lgq):
    global written_sofar
    if down_sample and written_sofar >=max_anchor:
        return
    if (not a) or len(l)<2:
        lgq.put( "[Error]\tanchor_empty_or_not_ambig\t%s]" % json.dumps({"anchor": a, "length": l}))
        return
    #print '(wid,n) = ', (a,l)
    for i in range(len(l)):
        (wid,f) = l[i]
        neg = l[:i]+l[i+1:]
        #neg = [nid for (nid, _) in neg]
        contexts = get_context(a,wid)        
        n=len(contexts)
        
        random.shuffle(contexts)
        
        if down_sample:
            contexts = contexts[:example_per_anchor]
            n=len(contexts)
                        
        if not contexts:
            lgq.put("[Error]\tcontext_empty\t%s" % json.dumps({"wid": wid, "frq": f}))
            continue
        # now we have a     
        cutpoint=int(math.ceil(tr_percent*n))
        if skip_line==-1:
            train = contexts[:cutpoint]
            test = contexts[cutpoint:]
        else:
            train = [c for c in contexts if skip_line not in c['paragraph_no']]
            test = [c for c in contexts if skip_line in c['paragraph_no']]
            
        lgq.put ("[success]\t%s" % json.dumps({"anchor": a,"wid": wid, "freq": f, "context_length": n,
                                            "train_size":len(train), "test_size":len(test)}))
        
        mater_sample(train, neg, trq)    
        mater_sample(test, neg, tsq)    
        if down_sample:
            written_sofar += 1
def mater_sample(context, neg, q):
    for c in context:
        c.pop('id', None)
        c.pop('_version_', None)
        q.put(json.dumps({"context":c, "neg": neg, "freq": len(c)},ensure_ascii=False).encode('utf-8'))
        
def worker(fname, q):
    w = open(fname,'w')
    print "[Writer started]"
    sys.stdout.flush()
    while True:
        s = q.get()
        if s=='kill':
            print "[Writer worker closing]"
            sys.stdout.flush()
            break
        w.write(s+"\n")
    w.close()    

    
startTime = time.time()
anchors = loadanchors()    
print '[anchors loaded to memory]'    
print time.time()-startTime
sys.stdout.flush()
        
startTime = time.time()

manager= Manager()

extension='%s.%s.json'%(down_sample, skip_line)
if down_sample:
    extension="%s.%s"%(max_anchor, extension)
    
train_name = os.path.join(home,'backup/datasets/cmod/train.%s'%(extension))
test_name = os.path.join(home,'backup/datasets/cmod/test.%s'%(extension))
log_name = os.path.join(home,'backup/datasets/cmod/log.%s'%(extension))
    
train_q = manager.Queue()
test_q = manager.Queue()
log_q = manager.Queue()


train_proc = Process(target=worker, args=(train_name, train_q))
train_proc.start()   
        
test_proc = Process(target=worker, args=(test_name, test_q))
test_proc.start()   

log_proc = Process(target=worker, args=(log_name, log_q))
log_proc.start()   


#pool = Pool(process_no) 
#pool.map(functools.partial(mater_anchor, trq=train_q, tsq=test_q ), anchors)
map(functools.partial(mater_anchor, trq=train_q, tsq=test_q, lgq=log_q   ), anchors)

train_q.put('kill')    
test_q.put('kill')
log_q.put('kill')

train_proc.join()
test_proc.join()
log_proc.join()

print 'Done'    
print time.time()-startTime
sys.stdout.flush()
