""" Create a train-set 
    entity_id, query_id, scores1, score2, ..., scoren, true/false (is it a correct entity)
"""
from __future__ import division
from wsd import *

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 215, The Wikisim Project"
__credits__ = ["Armin Sajadi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Armin Sajadi"
__email__ = "sajadi@cs.dal.ca"
__status__ = "Development"


sys.stdout.flush()

max_t=20
max_count=5000
#np.seterr(all='raise')

outdir = os.path.join(baseresdir, 'wsd')
outfile = os.path.join('../datasets/ner/trainrepository.%s.30000.tsv'%(max_count,))
if os.path.isfile(outfile): 
    sys.stderr.write(outfile + " already exist!\n")
    #sys.exit()

dsname = os.path.join('../datasets/ner/wiki-mentions.30000.json')

count = 0          
with open(dsname,'r') as ds, open(outfile,'w') as outf:
    qid=0
    for line in ds:                           
        js = json.loads(line.decode('utf-8').strip());
        S = js["text"]
        M = js["mentions"]
        count +=1        
        print "%s:\tS=%s\n\tM=%s" % (count, json.dumps(S, ensure_ascii=False).encode('utf-8'),json.dumps(M, ensure_ascii=False).encode('utf-8'))        
        C = generate_candidates(S, M, max_t=max_t, enforce=False)
        all_scores=get_all_scores(S,M,C)
        for i in  range(len(C)):
            m=M[i]
            cands = C[i]
            cand_scores = all_scores[i]
            wid = title2id(m[1]) 
            for (eid,_),scores in zip (cands, cand_scores):
                is_true_eid = (wid == eid)
                string_scores=[str(s) for s in scores]
                outf.write("\t".join([str(eid), str(qid)]+string_scores+[str(int(is_true_eid))])+"\n")
            qid += 1
        if count >= max_count:
            break
print "Done"             
        

        