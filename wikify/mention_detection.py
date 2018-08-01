from mention_detection import *
from sklearn.externals import joblib

from wsd import *

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 215, The Wikisim Project"
__credits__ = ["Armin Sajadi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Armin Sajadi"
__email__ = "sajadi@cs.dal.ca"
__status__ = "Development"


#constants
CORE_NLP=0
LEARNED_MENTION=1

SVC_HP_NROWS_S, SVC_HP_CV_S = 10000,1
SVC_HR_NROWS_S, SVC_HR_CV_S = 10000,20


SVC_HP_NROWS_L, SVC_HP_CV_L = 50000,1
SVC_HR_NROWS_L, SVC_HR_CV_L = 50000,20


mention_model_preprocessor_=None
mention_model_=None

def load_mention_model(nrows, svc):
    global mention_model_preprocessor_, mention_model_
    mention_model_preprocessor_fn = os.path.join(MODELDIR, 'svc_preprocessor.%s.pkl' % (nrows,))
    if os.path.isfile(mention_model_preprocessor_fn): 
        log("[load_mention_model]\tmention_model_preprocessor file (%s) loaded" % (mention_model_preprocessor_fn,))
        mention_model_preprocessor_ = joblib.load(open(mention_model_preprocessor_fn, 'rb'))
    else:
        log("[load_mention_model]\tmention_model_preprocessor file (%s) not found" % (mention_model_preprocessor_fn,))


    mention_model_fn = os.path.join(MODELDIR, 'svc_mentions_unbalanced.%s.%s.pkl' % (nrows,svc))
    if os.path.isfile(mention_model_fn): 
        mention_model_ = joblib.load(open(mention_model_fn, 'rb'))    
        log("[load_mention_model]\tmention_model_ file (%s) loaded" % (mention_model_fn,))
    else:
        log("[load_mention_model]\tmention_model_ file (%s) not found" % (mention_model_fn,))
        

def tokenize_stanford(text):
    addr = 'http://localhost:9001'
    params={'annotators': 'tokenize', 'outputFormat': 'json'}
    r = session.post(addr, params=params, data=text.encode('utf-8'))    
    
    return [token['originalText'] for token in r.json()['tokens']]

def encode_solrtexttagger_result(text,tags):
    """ Convert the solrtext output to our M,S format
        input:
            text: The original text
            tags: The result of the solrtexttagger
        output:
            S,M
            S: segmented sentence [w1, ..., wn]
            M: mensions [m1, ... , mj]
    """
    start=0
    termindex=0
    S=[]
    M=[]
    # pass 1, adjust partial mentions. 
    # approach one, expand (the other could be shrink)
    
    for tag in tags:
        assert text[tag[1]:tag[3]] == tag[5]
        seg = text[start:tag[1]]
        S += seg.strip().split()
        M.append([len(S),'UNKNOWN'])
        S += [" ".join(text[tag[1]:tag[3]].split())]
        start = tag[3]
        
    S += text[start:].strip().split()
    return S, M

def annotate_with_solrtagger(text):
    ''' Annonate a text using solrtexttagger
        Input: 
            text: The input text *must be unicode*
        Output:
            Annotated text
    '''
    addr = 'http://localhost:8983/solr/enwikianchors20160305/tag'
    params={'overlaps':'LONGEST_DOMINANT_RIGHT', 'tagsLimit':'5000', 'fl':'id','wt':'json','indent':'on','matchText':'true'}
    #text=solr_escape(text) Maybe not needed!
    r = session.post(addr, params=params, data=text.encode('utf-8'))    

    S,M = encode_solrtexttagger_result(text,r.json()['tags'])
    return S,M


def encode_corenlp_result(text,annotated):
    """ Convert the corenlp output to our M,S format
        input:
            text: The original text
            mentions: The result of the solrtexttagger
        output:
            S,M
            S: segmented sentence [w1, ..., wn]
            M: mensions [m1, ... , mj]
    """
    #****** Important ****
    #* The indices are not correct if it contains unicode, 
    #* in case you need to work with the indices, decode to utf-8
    #******
    S=[]
    M=[]
    P=[]
    # pass 1, adjust partial mentions. 
    # approach one, expand (the other could be shrink)
    
    for sentence in annotated['sentences']: 
        start=0
        
        for mention in sentence['entitymentions']:
            S += [token['originalText'] for token in sentence['tokens'][start:mention['tokenBegin']]]
            M.append([len(S),'UNKNOWN'])
            mentionstr = mention['text']
            S += [mentionstr]
            start = mention['tokenEnd']

        S += [token['originalText'] for token in sentence['tokens'][start:]]
        P += [[token['originalText'],token['pos']] for token in sentence['tokens']]
    return S, M, P

def annotate_with_corenlp(text):
    ''' Annonate a text using coreNLP
        Input: 
            text: The input text
        Output:
            Annotated text
    '''
    addr = 'http://localhost:9001'
    params={'annotators': 'entitymentions', 'outputFormat': 'json'}
    r = session.post(addr, params=params, data=text.encode('utf-8'))    

    
    S,M, P = encode_corenlp_result(text, r.json())
    return S,M,P

def solrtagger_pos(S,M,P):
    ''' Alligns the tags from corenlp to solrtagger's mentions
        Input:
            S: Sentence 
            M: Mentions
            P: POS of the mentions, from corenlp
        Output:
            Q: POS of solrtagger's mentions
    '''
    Q=[]
    j=0
    for i in range(len(M)):
        #m=tokenize_stanford(solr_unescape(S[M[i][0]]))  I skip escaping for now
        m=tokenize_stanford(S[M[i][0]]) 
        j_backup=j
        q=[]
        while j<len(P):
            if strsimilar(P[j][0], m[0])> .8:
                k=0
                while strsimilar(P[j][0], m[k])>0.8:
                    #q.append(P[j]) #good for debugging
                    q.append(P[j][1]) #good for debugging
                    k=k+1
                    j=j+1
                    if j >= len(P) or k>=len(m):
                        break

                Q.append(" ".join(q))
                break
            j=j+1
        if not q:
            Q.append("OTHER")
            j=j_backup
    return Q

def get_mention_count(s):
    """
    Description:
        Returns the amount of times that the given string appears as a mention in wikipedia.
    Args:
        s: the string (can contain AND, OR, ..)
    Return:
        The amount of times the given string appears as a mention in wikipedia
    """
    
    return sum(c for _,c in anchor2concept(s))  

def mention_prob(text):
    """
    Description:
        Returns the probability that the text is a mention in Wikipedia.
    Args:
        text: 
    Return:
        The probability that the text is a mention in Wikipedia.
    """
    
    total_mentions = get_mention_count(text)
    total_appearances = get_solr_count(text.replace(".", ""))
    if total_appearances == 0:
        return 0 # a mention never used probably is not a good link
    return float(total_mentions)/total_appearances

def get_mention_probs(S,M):
    return [mention_prob(S[m[0]]) for m in M]


def boil_down_candidate_score(score_list):
    return [sum(scores)/len(scores) for scores in scores_list]

def mention_overlap(S1, M1, S2,M2):
    '''Calculates the overlap between two given detected mentions
        Input:
            S1: Source Setnence
            M1: Source Mention
            S2: Destination Sentence
            M2: Destination mention            
        Output: A 0/1 vector of size M1, each element shows whether M1[i] is also in M2
    '''
    is_detected = []
    for m1 in M1:
        found = 0
        for m2 in M2:
            if strsimilar(S1[m1[0]], S2[m2[0]])>0.8:
                found=1
        is_detected.append(found)
    return is_detected

def detect_and_score_mentions(text, max_t=5):
    """Give
        Uses solrtagger to detect mentions, and score them
        Inputs:
            text: Given text
        Output:
            Scores, in this format [[(c111, c11s),...(c1k1, c1ks)],...[(cn11, pn1s),...(c1m1, p1ms)]]
            where cijk is the k-th scores for cij candidate
    """
    text = throw_unicodes(text)
    solr_S, solr_M = annotate_with_solrtagger(text)
    # max_t does not have to equal the number of candidates in wsd, it's just to 
    # get an average relevancy
    solr_C = generate_candidates(solr_S, solr_M, max_t=max_t, enforce=False)
    
    
    wsd_scores = [[sum(sc)/len(sc) for sc in get_scores(solr_S, solr_M, solr_C, method)] for method in \
               ['popularity','entitycontext','mention2entity','context2context','context2profile']]

    mention_scores=[]
    mention_scores.extend(wsd_scores)
    mention_scores.append(get_mention_probs(solr_S, solr_M))
    
    core_S, core_M, core_P = annotate_with_corenlp(text)
    overlap_with_corenlp = mention_overlap(solr_S, solr_M, core_S,core_M)
    mention_scores.append(overlap_with_corenlp)
    
    pos_list = solrtagger_pos(solr_S, solr_M,core_P)
    mention_scores.append(pos_list)
    
    return solr_S, solr_M, zip(*mention_scores)


def get_learned_mentions(text):
    if (mention_model_preprocessor_ is None) or (mention_model_ is None):
        log('[mention_models]\tmodel not loaded')
        raise Exception('model not loaded, try load_mention_model()')
        
    S_solr,M_solr,scores = detect_and_score_mentions(text)
    M_scores=[]
    for sc_vec in scores:
        # Unintuitive: When fitting, the first column was the mention_id, which was ignored!
        # And the preprocessor needs the exact column names!
        sc_frame = pd.DataFrame([sc_vec], columns=[str(i+1) for i in range(len(sc_vec))])
        X = mention_model_preprocessor_.transform(sc_frame)
        M_scores.append(mention_model_.predict(X))
    M = [m for m_s, m in zip(M_scores, M_solr) if m_s==1]
    return S_solr, M
    
def detect_mentions(text, mentionmethod=CORE_NLP):
    if mentionmethod == CORE_NLP:
        S, M, _ = annotate_with_corenlp(text)        
    if mentionmethod == LEARNED_MENTION:
        S, M =  get_learned_mentions(text)
    return S, M
    