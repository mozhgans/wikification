'''Converts the output of WikiExtractor.py to a text file with each line a json object
    Each object has the following fields:
    id: Wikipedia id
    title: The title
    opening_text: The first paragraph of the page
    opening_annotation: The annotation of the oppenint text which is itself another json objec
    text: The text
    annotation": The annotation of the text, which is itself another json object 

'''
import re
import json
import sys
import urllib
import sys
from HTMLParser import HTMLParser
sys.path.insert(0,'..')
from wikisim.wikipedia import *

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 215, The Wikisim Project"
__credits__ = ["Armin Sajadi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Armin Sajadi"
__email__ = "sajadi@cs.dal.ca"
__status__ = "Development"


fileinput = sys.stdin

# import StringIO
# fileinput = StringIO.StringIO(inputstr)
def replacelinks(text):
    hp = HTMLParser()
    
    annotations = []
    deltaStringLength = 0
    hrefreg=r'<a href="([^"]+)">([^>]+)</a>'
    ms = re.finditer(hrefreg, text)
    
    
    for m in ms:     

        url = m.group(1)
        # in the parser, url->encode->quote->encode (does nothing, already unicode)->write to file
        # we already have decoded while reading, we have to encode back, unquote, and decode, so that
        # the last encode on writing can make it back to unicode
        url=url.encode('utf-8')
        url =  urllib.unquote(url)
        url = url.decode('utf-8')
        #sometimes we have [[&]] that has been encoded to &amp; 
        #sometimes we have [[&amp]] that has been encoded to &amp;amp; 
        #so we unexcape twice!
        
        url=hp.unescape(url)
        url=hp.unescape(url)
        url=url.replace(u"\xA0"," ")
        
        x = url.find("#")
        if x!=-1:
            url=url[:x]
        #BUG: so what if x==0?  maybe the current approach, empty "dest" is not bad
        antext = m.group(2)
        if '//' not in url:
            annotations.append({
                "url"    :   url, 
                "surface_form" :   antext, 
                "from"  :   m.start() - deltaStringLength,
                "to"    :   m.start() - deltaStringLength+len(antext)
            })

        deltaStringLength += len(m.group(0)) - len(antext)

    #As a second step, replace all links in the article by their label
    text = re.sub(hrefreg, lambda m:m.group(2), text)  
    return annotations, text
def process():
    hp = HTMLParser()
    rstart=r'<doc id="(.*)" url="(.*)" title="(.*)">'
    rend=r'</doc>'

#     open_ann= open ('open_annotation.json', 'w')
#     ann= open ('annotation.json', 'w')
#     wikitext=open('wikitext.json','w')
    state=0
    page=""
    textlist=[]
    while True:
        line = fileinput.readline()
        if not line:
            break
        line = line.decode('utf-8').strip()
        
        if not line:
            continue
        if re.match(rend,line):
            # If you want to check the title, make sure to encode!
            if textlist and (id2title(wid) is not None):
                opening_ann, opening_text = replacelinks(opening_text)
                ann,text = replacelinks("\n".join(textlist))
                
#                 ElasticSearch   
#                 Buggy: convert annotations to text or do something about it
#                 print json.dumps({"index":{"_type":"page","_id":wid}}, ensure_ascii=False).encode('utf-8')
#                 page={"title": wtitle, "opening_text":opening_text, "opening_annotation":opening_ann,
#                       "text":text, "annotation": ann}
                
#                 print json.dumps(page, ensure_ascii=False).encode('utf-8')
                
#               General
                page={"id":wid, "title": wtitle, "opening_text":opening_text, "opening_annotation":opening_ann,
                      "text":text, "annotation": ann}
                
                print json.dumps(page, ensure_ascii=False).encode('utf-8')
                
            textlist=[]    
            state=0
            continue
        if state==0:
            if re.match('<doc', line):
                ms = re.match(rstart, line)
                wid=ms.group(1)
                wtitle=hp.unescape(ms.group(3)).replace(u"\xA0"," ")

                state = 1
            continue    

        
        if state==1:
            state=2
            continue
        textlist.append(line)
            
        if state==2:
            opening_text=line
            state=3
            continue
            
#     wikitext.close()
#     ann.close()
if __name__ == "__main__": process()