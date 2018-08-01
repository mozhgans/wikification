
''' Generate a json file (contexts.json) where eahc line is an anchor text 
    with its left and right context
'''

import os
import requests
import json
import string
import time
import sys
from multiprocessing import Pool, Process, Manager 
import functools
sys.path.insert(0,'../..')

from memapi.memwiki import *

startTime = time.time()
load_tables()
print 'wiki loaded to memory'    
print time.time()-startTime
sys.stdout.flush()

batch_rows=30000
init_start=0
process_no=30
max_rows=-1
window_size=10

qstr = 'http://localhost:8983/solr/enwiki20160305/select'
q='*:*'
#q='id:"60508"'

params={'indent':'on', 'wt':'json', 'q':q, "start":init_start,
        "rows":batch_rows}
home = os.path.expanduser("~");
#titles_out.write('kalam\n')

start=0
max_count=0
count_nottop=0

def find_nth(text, c, i, n, direction=0):
    if direction ==0:
        offset = text.find(c, i+1)
    else:    
        offset = text.rfind(c, 0,i)
    while offset >= 0 and n > 1:
        if direction ==0:
            offset = text.find(c, offset+len(c))
        else:    
            offset = text.rfind(c, 0, offset)
        n -= 1
    return offset

        
def getsentence(r, ann):
    text = r["text"]
    source_entity=r['id']
    source_title=r['title']
    
    entity = ann['url'] if ann['url']!="" else r['title']
    entityid = title2id(entity)
    if entityid is None:
        entity=entity[0].upper()+entity[1:]    
        entityid = title2id(entity)
    if entityid is None:
        return None
    trans_tab = dict.fromkeys(map(ord, string.punctuation), None)
    
    s = find_nth(text,' ', ann['from'],2*window_size,1)

    if s==-1:
        s=0
    l = text[s:ann['from']].translate(trans_tab).strip().split()
    l=" ".join(l[-window_size:])

    s = find_nth(text,' ', ann['from'],2*window_size,0)
    if s==-1:
        s=len(text)
    r = text[ann['to']:].translate(trans_tab).strip().split()
    r=" ".join(r[:window_size])
    pno = text[:ann['from']].count('\n')
    return {"left":l, "anchor":ann['surface_form'],"source_entity":source_entity,"source_title":source_title,
            "entity": entity, "entityid":entityid, "right": r, "paragraph_no":pno}


def process(r, tq, cq):
    tq.put(r['title'].encode('utf-8'))
    annotations = json.loads(r['annotation'])    
    for ann in annotations:
        sentence = getsentence(r, ann)
        if sentence is None:
            continue
        entity_context = json.dumps(sentence, ensure_ascii=False).encode('utf-8')
        cq.put(entity_context)
        
pool =Pool(process_no) 

def worker(fname, q):
    w = open(fname,'w')
    print "Writer started"
    sys.stdout.flush()
    while True:
        s = q.get()
        if s=='kill':
            print "Writer worker closing"
            sys.stdout.flush()
            break
        w.write(s+"\n")
    w.close()    
    
manager= Manager()

titles_name = os.path.join(home,'backup/wikipedia/20160305/cmod/titles.txt')
title_q = manager.Queue()

cont_name = os.path.join(home,'backup/wikipedia/20160305/cmod/contexts.json')
cont_q = manager.Queue()


title_proc = Process(target=worker, args=(titles_name, title_q))
title_proc.start()   
        
cont_proc = Process(target=worker, args=(cont_name, cont_q))
cont_proc.start()   


start=init_start
rows=0
while True:
    params["start"] = start

    r = requests.get(qstr, params=params)
    print ("A batch retrieved: "+ str(start)+'\n')
    sys.stdout.flush()
    if len(r.json()['response']['docs'])==0:
        break
    #print r.json()['response']['docs']
    D = r.json()['response']['docs']
        
    pool.map(functools.partial(process, tq=title_q, cq=cont_q ), D)
    #pool.close() 
    #pool.join() 
    
    start += batch_rows
    rows += batch_rows
    if max_rows !=-1 and rows >= max_rows:
        break
        
cont_q.put('kill')    
title_q.put('kill')

title_proc.join()
cont_proc.join()

print 'done'    
print time.time()-startTime