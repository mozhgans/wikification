import sys
import logging
from gensim import *
import json
import os

in_file = "/users/grad/sajadi/backup/wikipedia/20160305/edited/enwiki-20160305-pagelinks.dumped.shuf.ssv"
#in_file = "/users/grad/sajadi/backup/wikipedia/20160305/texts/enwiki-20160305-cleantext.txt"
#in_file = "/users/grad/sajadi/backup/wikipedia/20160305/texts/enwiki-20160305-replace_surface.txt"
output_root = "/users/grad/sajadi/backup/wikipedia/20160305/embed/"

params={"in_file_path": in_file,
         "phrasing":1,
         "sg":1, "size":500, "window":2, "min_count":1, "workers":28, "negative":20, "iter":50,
         }
baseinname=os.path.splitext(os.path.basename (params['in_file_path']))[0]
params['params_desc'] = 'word2vec.%s.%s.%s.%s.%s.%s.%s.%s.%s' % (baseinname, params["sg"], params['phrasing'],params["size"], params["window"],
                        params["min_count"], params["workers"], params["negative"],params["iter"] )
params["out_dir"] = os.path.join(output_root, params['params_desc'])
params["out_file"] = os.path.join(params["out_dir"], params['params_desc'])
if not os.path.exists(params["out_dir"]):
        os.makedirs(params["out_dir"])
with open(os.path.join(params["out_dir"], 'README.md'), 'w') as f:
    f.write(json.dumps(params))
####################################

logging.basicConfig(filename='logging.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print "loading"   
sys.stdout.flush()
sentences = models.word2vec.LineSentence(params['in_file_path'])
if params['phrasing']==1:
    bigram_transformer = models.phrases.Phraser(models.Phrases(sentences))
    sentence_stream = bigram_transformer[sentences]
else:
    sentence_stream = sentences
print "training"    
sys.stdout.flush()
model = models.Word2Vec(sentence_stream, sg=params["sg"], size=params["size"], window=params["window"],
                        min_count=params["min_count"], workers=params["workers"], 
                        negative=params["negative"], iter=params["iter"])
model.save(params["out_file"])
print "done"    
sys.stdout.flush()