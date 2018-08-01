#!/users/grad/sajadi/backup/anaconda2/envs/wikisim/bin/python
#/home/asajadi/anaconda2/envs/wikisim/bin/python

import json
import cgi, cgitb 

import sys, os

# Apache wanats this
dirname = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(dirname, '..'))

from wikisim.calcsim import *
log('cgi-pairsim started');

# Import modules for CGI handling 

# Create instance of FieldStorage 
form = cgi.FieldStorage() 

# Get data from fields
task=form.getvalue('task')
direction =int(form.getvalue('dir'))
cutoff=form.getvalue('cutoff')
viz=form.getvalue('viz');

c1 = form.getvalue('c1')
c2  = form.getvalue('c2')


log('param %s, %s', c1, c2);

print "Content-type:application/json\r\n\r\n"

if cutoff is not None:
	cutoff = None if cutoff.lower()=="all" else int(cutoff)

if task == 'emb':
	id1=title2id(c1);
	if id1 is None:
		print json.dumps({"err": '%s not found in wikipedia, check if this url exist: "en.wikipedia.org/wiki/%s"' % (c1,c1)});
		exit();	
	em = conceptrep(id1, method='rvspagerank', direction=direction, get_titles=(viz=='true'), cutoff=cutoff)
	print json.dumps({"em": em.to_dict()});
	exit();

id1=title2id(c1);
if id1 is None:
	print json.dumps({"err": '%s not found in wikipedia, check if this url exist: "en.wikipedia.org/wiki/%s"' % (c1,c1)});
	exit();	

id2=title2id(c2);
if id2 is None:
	print json.dumps({"err": '%s not found in wikipedia, check if this url exist: "en.wikipedia.org/wiki/%s"' % (c2,c2)});
	exit();	

sim=getsim(title2id(c1), title2id(c2), 'rvspagerank', direction)


if viz is None or viz=='false':
	print json.dumps({"rel":sim})
 	exit()

cre1 = conceptrep(title2id(c1), method='rvspagerank', direction=direction, get_titles=True, cutoff=cutoff)
cre2 = conceptrep(title2id(c2), method='rvspagerank', direction=direction, get_titles=True, cutoff=cutoff)

print json.dumps({"rel":sim, "em1": cre1.to_dict(), "em2": cre2.to_dict()})

close()

log('finished');
