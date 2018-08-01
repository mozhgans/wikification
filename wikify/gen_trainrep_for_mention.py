""" Create a train-set 
    entity_id, query_id, scores1, score2, ..., scoren, true/false (is it a correct entity)
"""
from __future__ import division
from mention_detection import *

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 215, The Wikisim Project"
__credits__ = ["Armin Sajadi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Armin Sajadi"
__email__ = "sajadi@cs.dal.ca"
__status__ = "Development"



max_count=5000
skip_lines=0

outdir = os.path.join(baseresdir, 'wsd')
outfile = os.path.join('../datasets/ner/mentiontrainrepository.%s.30000.tsv'%(max_count,))
if os.path.isfile(outfile): 
    sys.stderr.write(outfile + " already exist!\n")
    sys.exit()

dsname = os.path.join('../datasets/ner/wiki-mentions.30000.json')

count = 0  
mention_id = 0
with open(dsname,'r') as ds, open(outfile,'w') as outf:
    for line in ds:                           
        count +=1  
        if count <= skip_lines:
            continue
            
        js = json.loads(line.decode('utf-8').strip());
        S = js["text"]
        M = js["mentions"]
        text= " ".join(S)
        print "%s:\tS=%s\n\tM=%s\ttext=%s" % (count, json.dumps(S, ensure_ascii=False).encode('utf-8'),json.dumps(M, ensure_ascii=False).encode('utf-8'),text.encode('utf-8'))        
        
        solr_S, solr_M, scores = detect_and_score_mentions(text)
        correct_mention = mention_overlap(solr_S, solr_M, S, M)
        for i in  range(len(solr_M)):
            string_scores=[str(s) for s in scores[i]]
            outf.write("\t".join([str(mention_id)] + string_scores+[str(correct_mention[i])])+"\n")
            mention_id += 1
        if count >= max_count:
            break
print "Done"             
        

        