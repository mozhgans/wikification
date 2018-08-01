"""Evaluating the wsd module. It assumes the sentences are already segmented
"""
from wikify import *
import sys
from optparse import OptionParser
#sys.path.insert(0,'..')
#from wikisim.calcsim import *
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


# max_t = 20
# max_count = 30
# verbose = True

fresh_restart=False


dsnames = [os.path.join(dsdir_ner, 'kore.json'),
          os.path.join(dsdir_ner, 'wiki-mentions.5000.json'),
#          os.path.join(dsdir_ner, 'aida.json'),  
          os.path.join(dsdir_ner, 'msnbc.txt.json'),
          os.path.join(dsdir_ner, 'aquaint.txt.json') 
          ]




outdir = os.path.join(baseresdir, 'wikify')
# if not os.path.exists(outdir): #Causes synchronization problem
#     os.makedirs(outdir)

tmpdir = os.path.join(outdir, 'tmp')
# if not os.path.exists(tmpdir): #Causes synchronization problem
#     os.makedirs(tmpdir)
    
resname =  os.path.join(outdir, 'reslog.csv')
#clearlog(resname)

detailedresname=  os.path.join(outdir, 'detailedreslog.txt')
#clearlog(detailedresname)


allparams = (   (CORE_NLP,None, None, LTR_NROWS_L),
#               (LEARNED_MENTION, SVC_HP_NROWS_S, SVC_HP_CV_S, LTR_NROWS_L), 
#               (LEARNED_MENTION, SVC_HR_NROWS_S, SVC_HR_CV_S, LTR_NROWS_L),
              (LEARNED_MENTION, SVC_HP_NROWS_L, SVC_HP_CV_L, LTR_NROWS_L), 
              (LEARNED_MENTION, SVC_HR_NROWS_L, SVC_HR_CV_L, LTR_NROWS_L))





for mentionmethod, svc_nrows, svc_cv,  ltr_nrows in allparams:
    load_wsd_model(ltr_nrows)
    load_mention_model(svc_nrows, svc_cv)
    for dsname in dsnames:
        start = time.time()

        print "dsname: %s, mentionmethods: %s, max_t: %s ..."  % (dsname,
                mentionmethod, max_t)
        sys.stdout.flush()

        tmpfilename = os.path.join(tmpdir, 
                                   '-'.join([str(mentionmethod), str(max_t), str(svc_nrows), str(svc_cv), str(ltr_nrows) ,os.path.basename(dsname)]))
        mention_overall=[]
        wikify_overall=[]
        start_count=-1
        if os.path.isfile(tmpfilename) and not fresh_restart:
            with open(tmpfilename,'r') as tmpf:
                for line in tmpf:
                    js = json.loads(line.strip())
                    start_count = js['no']

                    if js['mention_measures'] is not None:
                        mention_overall.append(js['mention_measures'])

                    if js['wikify_measures'] is not None:
                        wikify_overall.append(js['wikify_measures'])
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
                text= " ".join(S)    
                #S2,M2 = detect_mentions(text, mentionmethod=mentionmethod)      
                S2,M2 = wikify_string(text, mentionmethod=mentionmethod)

                mention_measures = get_sentence_measures(S2, M2, S, M, wsd_measure=False)
                mention_overall.append(mention_measures)

                wikify_measures = get_sentence_measures(S2, M2, S, M, wsd_measure=True)
                wikify_overall.append(wikify_measures)

                tmpf.write(json.dumps({"no":count, "mention_measures":mention_measures, "wikify_measures":wikify_measures})+"\n")



        elapsed = str(timeformat(int(time.time()-start)));
        print "done"
        detailedres ={"mentionmethod": mentionmethod, "max_t":max_t, "svc_nrows": svc_nrows, "svc_cv": svc_cv, "ltr_nrows":ltr_nrows, "dsname": dsname,
                      "max_t": max_t, "mention_overall":mention_overall, "wikify_overall": wikify_overall, "elapsed": elapsed}


        logres(detailedresname, '%s',  json.dumps(detailedres))
        #print mention_overall

        mention_overall_measures = get_overall_measures(mention_overall)    
        output = ('mention_evaluation',mentionmethod, max_t, svc_nrows, svc_cv, ltr_nrows, dsname) + mention_overall_measures + (elapsed,)        
        logres(resname, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', *output)

        #print wikify_overall
        wikify_overall_measures = get_overall_measures(wikify_overall)  
        output = ('wikify_evaluation',mentionmethod, max_t,  svc_nrows, svc_cv, ltr_nrows, dsname) + wikify_overall_measures + (elapsed,)
        logres(resname, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n', *output)
            

print "done"