''' Extract all the anchor text with their target url
'''
import json
import sys
import urllib
import StringIO

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 215, The Wikisim Project"
__credits__ = ["Armin Sajadi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Armin Sajadi"
__email__ = "sajadi@cs.dal.ca"
__status__ = "Development"

fileinput = sys.stdin

#fileinput = StringIO.StringIO(inputstr)
while True:
    line = fileinput.readline()
    if not line:
        break
    line = line.decode('utf-8').strip()
    js = json.loads(line)    
    if "annotation" not in js:
        continue
    for a in js["annotation"]:
        print u"{0}\t{1}".format(a["surface_form"],a["url"].replace(" ","_")).encode('utf-8')
                
    