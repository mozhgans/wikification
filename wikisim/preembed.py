"""Pre calculation of the embeddings"""

from config import *

# %load_ext autoreload
# %autoreload

# %aimport calcsim
from calcsim import *
direction = DIR_IN;
dirstr = graphtype(direction)
#wid_fname  = os.path.join(home, 'backup/wikipedia/20160305/embed/enwiki-20160305-page.dumped.ssv')
wid_fname = os.path.join(home, 'backup/wikipedia/20160305/embed/enwiki-20160305-embeddings.'+dirstr+'.dead_2.ssv')

done_fname = os.path.join(home, 'backup/wikipedia/20160305/embed/enwiki-20160305-embeddings.'+dirstr+'.done.ssv')
dead_fname = os.path.join(home, 'backup/wikipedia/20160305/embed/enwiki-20160305-embeddings.'+dirstr+'.dead_3.ssv')
rewrite = True
lastwid = ""
if os.path.exists(done_fname):
    with open(done_fname) as done_f:
        for lastwid in done_f:
            pass
        if lastwid is not None:
            lastwid = lastwid.strip() 
            

wid_f = open(wid_fname)
done_f = open(done_fname, 'a')
dead_f = open(dead_fname, 'a')
    
if lastwid:
    for line in wid_f:
        if line.strip() == lastwid:
            break
    print "Continuing from ", lastwid
else: 
    print "Fresh start"
    
for line in wid_f:
    wid = line.strip().split('\t')[0]
    if rewrite:
        deletefromcache(wid, direction)
    em = concept_embedding(wid, direction)
    if em.empty:
        count = str(len(getlinkedpages(wid, direction)))
        dead_f.write(wid+'\t'+id2title(wid)+'\t'+count+'\n')
    done_f.write(wid+'\n')
wid_f.close()
done_f.close()
dead_f.close()

print "done"