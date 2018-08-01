#!/users/grad/sajadi/backup/anaconda2/envs/wikisim/bin/python
#/home/asajadi/anaconda2/envs/wikisim/bin/python

import json
import cgi, cgitb 

import sys, os

dirname = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(dirname, '..'))

from wikify.wikify import *

log('cgi-wikify started');

# Import modules for CGI handling 

# Create instance of FieldStorage 
form = cgi.FieldStorage() 

# Get data from fields
params = int(form.getvalue('modelparams'))
wikitext  = form.getvalue('wikitext')

print "Content-type:application/json\r\n\r\n"

if params==0:
    mentionmethod = 0
    load_wsd_model(LTR_NROWS_S)    
    
if params==1 or params==2:
    mentionmethod = 1
    svc_nrows, svc_cv, ltr_nrows = get_wikifify_params(params)    
    load_mention_model(svc_nrows, svc_cv)
    load_wsd_model(ltr_nrows)

wikifiedtext = wikify_api(wikitext, mentionmethod=mentionmethod)


print json.dumps({"outtext":wikifiedtext})

#close()

log('finished');
