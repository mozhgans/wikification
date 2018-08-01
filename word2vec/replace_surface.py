import re
import sys
import urllib
import sys
from HTMLParser import HTMLParser
import time

sys.path.insert(0,'..')
from memapi import memwiki as wiki

fileinput = sys.stdin
skip_line = 1; #LINE 0 is the title

def title2id(title):
    if not title:
        return "NA1"
    wid = wiki.title2id(title)
    if wid is None:
        title=title[0].upper()+title[1:]    
        wid = wiki.title2id(title)
    if wid is None:
        return "NA2"
    return str(wid)
    

def url2id(antext, url):
    hp = HTMLParser()
    
    url=url.encode('utf-8')
    url =  urllib.unquote(url)
    url = url.decode('utf-8')

    url=hp.unescape(url)
    url=hp.unescape(url)
    url=url.replace(u"\xA0"," ")
    x = url.find("#")
    if x!=-1:
        url=url[:x]
    return "id_"+title2id(url)
    
    
def replacelinks(text):
    
    annotations = []
    deltaStringLength = 0
    hrefreg=r'<a href="([^"]+)">([^>]+)</a>'
    
    text = re.sub(hrefreg, lambda m:url2id(m.group(2), m.group(1)), text)  
    return text


def process():
    line_no = -1    
    hp = HTMLParser()
    rstart=r'<doc id="(.*)" url="(.*)" title="(.*)">'
    rend=r'</doc>'
    for line in fileinput.readlines():
        line = line.decode('utf-8').strip()
        if not line:
            continue
            
        ms = re.match(rstart, line)
        if ms is not None:
            wid=ms.group(1)
            wtitle=hp.unescape(ms.group(3)).replace(u"\xA0"," ")
            line_no=0
            #print 'id_'+title2id(wtitle)
            continue
        if line_no == skip_line:
            print ""
            line_no +=1
            continue

        if re.match(rend,line):
            line_no=-1
            print "\n"
            continue    

        text = replacelinks(line).encode('utf-8')
        print text
        line_no += 1
        continue
    
if __name__ == "__main__": 
    #startTime = time.time()
    wiki.load_tables()
    #print 'wiki loaded to memory'    
    #print time.time()-startTime
    #sys.stdout.flush()
    process()