'''The Legacy WSD evaluation
This is how we evaluated for the thesis. 
The main difference is "enforce=True" in the candidate generation, 
an option which always makes sure the correct entity is among the candidates
'''


import sys
from optparse import OptionParser

#sys.path.insert(0,'..')
#from wikisim.calcsim import *
from wsd import *
import time
from random import shuffle
from config import *
np.seterr(all='raise')


parser = OptionParser()
parser.add_option("-t", "--max_t", action="store", type="int", dest="max_t", default=5)
parser.add_option("-c", "--max_count", action="store", type="int", dest="max_count", default=-1)
parser.add_option("-v", action="store_true", dest="verbose", default=False)

(options, args) = parser.parse_args()



max_t = options.max_t
max_count = options.max_count
verbose = options.verbose
# ws = options.win_size


# max_t = 15
# max_count = 20
# verbose = True

fresh_restart=False



dsnames = [os.path.join(dsdir_ner, 'kore.json'),
          os.path.join(dsdir_ner, 'wiki-mentions.5000.json'),
          os.path.join(dsdir_ner, 'aida.json'),  
          os.path.join(dsdir_ner, 'msnbc.txt.json'),
          os.path.join(dsdir_ner, 'aquaint.txt.json') 
          ]

methods = ['popularity','keydisamb','entitycontext','mention2entity','context2context','context2profile', 'learned']
methods = ['popularity','mention2entity','context2context','context2profile', 'learned']
methods = ['context2context']

outdir = os.path.join(baseresdir, 'wsd_legacy')
# if not os.path.exists(outdir): #Causes synchronization problem
#     os.makedirs(outdir)

tmpdir = os.path.join(outdir, 'tmp')
# if not os.path.exists(tmpdir): #Causes synchronization problem
#     os.makedirs(tmpdir)
    
resname =  os.path.join(outdir, 'reslog.csv')
#clearlog(resname)

detailedresname=  os.path.join(outdir, 'detailedreslog.txt')
#clearlog(detailedresname)



for method in methods:
    if 'word2vec' in method:
        gensim_loadmodel(word2vec_path)
        print "loaded"
        sys.stdout.flush()
    for dsname in dsnames:
        start = time.time()
        
        print "dsname: %s, method: %s, op_method: %s, direction: %s, max_t: %s, ws: %s ..."  % (dsname,
                "rvspagerank", method, "both", max_t, 5)
        
        sys.stdout.flush()
        
        tmpfilename = os.path.join(tmpdir, 
                                   '-'.join(['rvspagerank', str(DIR_BOTH), method, str(max_t), '5', os.path.basename(dsname)]))
        
        overall=[]
        start_count=-1
        if os.path.isfile(tmpfilename) and not fresh_restart:
            with open(tmpfilename,'r') as tmpf:
                for line in tmpf:
                    js = json.loads(line.strip())
                    start_count = js['no']
                    if js['tp'] is not None:
                        overall.append(js['tp'])
        
        if start_count !=-1:
            print "Continuing from\t", start_count
            
        count=0
        with open(dsname,'r') as ds, open(tmpfilename,'a') as tmpf:
            for line in ds:
                if (max_count !=-1) and (count >= max_count):
                    break
                
                js = json.loads(line.decode('utf-8').strip());
                S = js["text"]
                M = js["mentions"]
                count +=1
                if count <= start_count:
                    continue
                    
                if verbose:
                    print "%s:\tS=%s\n\tM=%s" % (count, json.dumps(S, ensure_ascii=False).encode('utf-8'),json.dumps(M, ensure_ascii=False).encode('utf-8'))
                    sys.stdout.flush()
                    
                C = generate_candidates(S, M, max_t=max_t, enforce=True)
                
                try:
                    #ids, titles = disambiguate_driver(S,M, C, ws=0, method=method, direction=direction, op_method=op_method)
                    ids, titles = wsd(S,M,C, method=method)
                    tp = get_tp(ids, M) 
                except Exception as ex:
                    tp = (None, None)
                    print "[Error]:\t", type(ex), ex
                    raise
                    continue
                
                overall.append(tp)
                tmpf.write(json.dumps({"no":count, "tp":tp})+"\n")
                    

        elapsed = str(timeformat(int(time.time()-start)));
        print "done"
        
        detailedres ={"dsname":dsname, "method": "rvspagerank", "op_method": method, "driection": 'both',
                      "max_t": max_t, "tp":overall, "elapsed": elapsed, "ws": 5}
        
        
        logres(detailedresname, '%s',  json.dumps(detailedres))
        
        micro_prec, macro_prec = get_prec(overall)        
        logres(resname, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s', 'rvspagerank', method, 'both', max_t , 5, 
               dsname, micro_prec, macro_prec, elapsed)
        

print "done"