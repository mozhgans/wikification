#!/users/grad/sajadi/backup/anaconda2/envs/wikisim/bin/python
#/home/asajadi/anaconda2/envs/wikisim/bin/python

import sys
import cgi, os
import cgitb; cgitb.enable()
import tempfile
import time
import datetime
from bs4 import BeautifulSoup as soup
from shutil import copyfile
import json


dirname = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(dirname, '..'))

from wikify.wikify import *

log('cgi-batchwikify started');
print "Content-type:application/json\r\n\r\n"

tmpdir = os.path.join(outdir, 'tmp')
if not os.path.exists(tmpdir):
    os.makedirs(tmpdir)
    os.chmod(tmpdir, 0777)    
    log('tmp directory generated');

resultfile = os.path.join(outdir, 'results.html')
if not os.path.exists(resultfile):
    copyfile(os.path.join(outdir, 'results.orig.html'),resultfile)
    os.chmod(resultfile, 0777)    


form = cgi.FieldStorage()


# Get filename here.

fileitem = form['file']
import time
jobid = time.strftime("%Y%m%d-%H%M%S")


#Test if the file was uploaded
if fileitem.filename:

    # strip leading path from file name to avoid 
    # directory traversal attacks
    fn = os.path.basename(fileitem.filename)
    fnbase, fnext = os.path.splitext(fn);
    # sudo chown www-data tmp

    infn=os.path.join(tmpdir, jobid+'-'+fn)
    outfn_rel= os.path.join('tmp', jobid+'-'+fnbase+'.out'+fnext)
    outfn=      os.path.join(outdir, outfn_rel)

    with open(infn, 'wb') as f:
        f.write(fileitem.file.read())   
else:
    print json.dumps({"redirect":"../error.html"})
    exit()


# Update result file

with open(resultfile,'r') as f:
    html=f.read()
sp = soup(html,'lxml')
resdiv= sp.find("div", {"id": "resultsdiv"})
newhtml="""
            <div class="alert alert-warning " role="alert">
                <p>
                	<b>Job Id {0}</b>: Processing {1} started at: {2}, save this Job Id and refresh this page 
                	until the results are posted
                </p>
            </div>            
        """.format(jobid, infn, datetime.datetime.now());
new_tag = soup(newhtml, 'lxml')
new_tag=new_tag.body.next
resdiv.append(new_tag)
with open(resultfile,'w') as f:
    f.write(sp.prettify())

# Starting the process

params = int(form.getvalue('modelparams'))

if params==0:
    mentionmethod = 0
    load_wsd_model(LTR_NROWS_S)    
    
if params==1 or params==2:
    mentionmethod = 1
    svc_nrows, svc_cv, ltr_nrows = get_wikifify_params(params+2)    
    load_mention_model(svc_nrows, svc_cv)
    load_wsd_model(ltr_nrows)


wikify_from_file_api(infn, outfn, mentionmethod)

# Update result file

with open(resultfile,'r') as f:
    html=f.read()

sp = soup(html,'lxml')
resdiv= sp.find("div", {"id": "resultsdiv"})
newhtml="""
            <div class="alert alert-success " role="alert">
                <p>
                	JobId: {0}: Done at {1}, download the results from:  
                </p>
                <a href={2}>{3}
                </a>
           </div>            
        """.format(jobid, datetime.datetime.now(), outfn_rel, outfn_rel);
new_tag = soup(newhtml,'lxml')
new_tag=new_tag.body.next

resdiv.append(new_tag)
with open(resultfile,'w') as f:
    f.write(sp.prettify())

print json.dumps({"redirect":"out/results.html", "jobid":jobid})


