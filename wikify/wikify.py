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

HIGH_PREC_SMALL = 1
HIGH_REC_SMALL = 2
HIGH_PREC_LARGE = 3
HIGH_REC_LARGE = 4
 

def get_wikifify_params(opt):
    if opt == HIGH_PREC_SMALL:
        return SVC_HP_NROWS_S, SVC_HP_CV_S, LTR_NROWS_S
    
    if opt == HIGH_REC_SMALL: 
        return SVC_HR_NROWS_S, SVC_HR_CV_S, LTR_NROWS_S
    
    if opt == HIGH_PREC_LARGE: 
        return SVC_HP_NROWS_L, SVC_HP_CV_L, LTR_NROWS_L
    
    if opt == HIGH_REC_LARGE: 
        return SVC_HR_NROWS_L, SVC_HR_CV_L, LTR_NROWS_L


def wikify_string(line, mentionmethod=CORE_NLP, max_t=20):
    if not isinstance(line, unicode):
        line = line.decode('utf-8')
    
    S,M = detect_mentions(line, mentionmethod)      
    C = generate_candidates(S, M, max_t=max_t, enforce=False)
    E = wsd(S, M, C, method='learned')
    for m,e in zip(M,E[1]):
        m[1]=e
    return S,M

def wikify_a_line(line, mentionmethod=CORE_NLP):
    ''' Annotate a single line 
        Input:
            line: The given string
            mentionmethod: The mention detection method
        Output:
            Annotated Sentence inwhich mentiones are hyper-linked to the Wikipedia concepts
    '''
    S, M = wikify_string(line, mentionmethod) 
    for m in M: 
        S[m[0]]="<a href=https://en.wikipedia.org/wiki/%s>%s</a>"  % (m[1], S[m[0]])
    S_reconcat = " ".join(S)
    return S_reconcat
            
def wikify_api(text, mentionmethod=CORE_NLP):
    outlist=[]
    for line in text.splitlines():
        outlist.append(wikify_a_line(line, mentionmethod))
    return "<br>".join(outlist).decode('utf-8')

def wikify_from_file_api(infilename, outfilename, mentionmethod=CORE_NLP):
    with open(infilename) as infile, open(outfilename, 'w') as outfile:
        for line in infile.readlines():
            wikified = wikify_api(line, mentionmethod)
            outfile.write(wikified + "\n")

            