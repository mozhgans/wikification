"""Context-based disambiguation and also Learning-To-Rank combination
    of several features.
"""

from __future__ import division

from collections import Counter
import sys
from vsmcoherence import *
from sklearn.externals import joblib
#sys.path.insert(0,'..')

#from wikisim.calcsim import *
#from wsd.wsd import *
# My methods
#from senseembed_train_test.ipynb

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 215, The Wikisim Project"
__credits__ = ["Armin Sajadi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Armin Sajadi"
__email__ = "sajadi@cs.dal.ca"
__status__ = "Development"


LTR_NROWS_S = 10000
LTR_NROWS_L = 50000
wsd_model_preprocessor_ = None
wsd_model_=None
def load_wsd_model(nrows):
    global wsd_model_preprocessor_, wsd_model_
    
    wsd_model_preprocessor_fn = os.path.join(MODELDIR, 'ltr_preprocessor.%s.pkl' %(nrows, ))
    if os.path.isfile(wsd_model_preprocessor_fn): 
        wsd_model_preprocessor_ = joblib.load(open(wsd_model_preprocessor_fn, 'rb'))    
        log("[load_wsd_model]\twsd_model_preprocessor file (%s) loaded" % (wsd_model_preprocessor_fn,))
    else:
        log("[load_wsd_model]\twsd_model_preprocessor file (%s) not found" % (wsd_model_preprocessor_fn,))


    wsd_model_fn_ = os.path.join(MODELDIR, 'ltr.%s.pkl'%(nrows,))
    if os.path.isfile(wsd_model_fn_): 
        wsd_model_ = joblib.load(open(wsd_model_fn_, 'rb'))    
        log("[load_wsd_model]\twsd_model file (%s) loaded" % (wsd_model_fn_,))
    else:
        log("[load_wsd_model]\twsd_model file (%s) not found" % (wsd_model_fn_,))


def get_context(anchor, eid, rows=50000):
    """Returns the context
       Inputs: 
           anchor: the anchor text
           eid: The id of the entity this anchor points to
       Output:
           The context (windows size is, I guess, 20)       
    """
    params={'wt':'json', 'rows':rows}
    anchor = solr_escape(anchor)
    
    q='anchor:"%s" AND entityid:%s' % (anchor, eid)
    params['q']=q
    
#     session = session.Session()
#     http_retries = Retry(total=20,
#                     backoff_factor=.1)
#     http = session.adapters.HTTPAdapter(max_retries=http_retries)
#     session.mount('http://localhost:8983/solr', http)
    
    r = session.get(qstr, params=params).json()
    if 'response' not in r: 
        log("[get_context]\t(terminating)\t%s",(str(r),))
        sys.stdout.flush()
        os._exit(0)
        
    if not r:
        return []
    return r['response']['docs']

#from wsd
def word2vec_context_candidate_scores (S, M, candslist, ws=5):
    '''returns entity scores using the similarity with their context
       Inputs: 
           S: Sentence
           M: Mentions
           candslist: candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
            ws: word size
       Returns:
           Scores [[s11,...s1k],...[sn1,...s1m]] where sij is cij similarity to the key-entity
    '''
    
    candslist_scores=[]
    for i in range(len(candslist)):
        cands = candslist[i]
        pos = M[i][0]
        context = S[max(pos-ws,0):pos]+S[pos+1:pos+ws+1]
        context_vec = sp.zeros(getword2vec_model().vector_size)
        for c in context:
            context_vec += getword2vector(c).as_matrix()
        cand_scores=[]

        for c in cands:
            try:
                # We have zero vectors, so this can rais an exception
                # or return none                
                cand_vector = getentity2vector(encode_entity(c[0],'word2vec', get_id=False))
                d = 1-sp.spatial.distance.cosine(context_vec, cand_vector);
            except:
                d=0                
            if np.isnan(d):
                d=0
            
            cand_scores.append(d)    
        candslist_scores.append(cand_scores) 

    return candslist_scores

#from wsd
def word2vec_context_disambiguate(S, M, candslist):
    '''Disambiguate a sentence using word-context similarity
       Inputs: 
           S: Sentence
           M: Mentions
           candslist: candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
           
       Returns: 
           a list of entity ids and a list of titles
    '''
    
        
    candslist_scores = word2vec_context_candidate_scores (S, M, candslist)
                      
    # Iterate 
    true_entities = []
    for cands, cands_scores in zip(candslist, candslist_scores):
        max_index, max_value = max(enumerate(cands_scores), key= lambda x:x[1])
        true_entities.append(cands[max_index][0])

    titles = ids2title(true_entities)
    return true_entities, titles 



#from wikisim
def get_solr_count(s):
    """ Gets the number of documents the string occurs 
        NOTE: Multi words should be quoted
    Arg:
        s: the string (can contain AND, OR, ..)
    Returns:
        The number of documents
    """
    q='+text:(%s)'%(solr_escape(s),)
    qstr = 'http://localhost:8983/solr/enwiki20160305/select'
    params={'indent':'on', 'wt':'json', 'q':q, 'rows':0}
    r = session.get(qstr, params=params)
    D = r.json()['response']
    return D['numFound']



# Editing Ryan's code
def context_to_profile_sim(mention, context, candidates):
    """
    Description:
        Uses Solr to find the relevancy scores of the candidates based on the context.
    Args:
        mention: The mention as it appears in the text
        context: The words that surround the target word.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
    Return:
        The score for each candidate in the same order as the candidates.
    """
    
    
    # put text in right format
    if not context:
        return [0]*len(candidates)
    context = solr_escape(context)
    mention = solr_escape(mention)
    
    filter_ids = " ".join(['id:' +  str(tid) for tid,_ in candidates])
        

    # select all the docs from Solr with the best scores, highest first.
    qst = 'http://localhost:8983/solr/enwiki20160305/select'
    #q='text:('+context+')^1 title:(' + mention+')^1.35'
    q='text:('+context+')'
    
    params={'fl':'id score', 'fq':filter_ids, 'indent':'on',
            'q':q, 'wt':'json','rows':len(candidates)}
    
    
    r = session.get(qst, params = params).json()['response']['docs']
    id_score_map=defaultdict(float, {long(ri['id']):ri['score'] for ri in r})
    id_score=[id_score_map[c] for c,_ in candidates]
    return id_score

# Important TODO
# This queriy is very much skewed toward popularity, better to replace space with AND
#!!!! I don't like this implementation, instead of retrieving and counting, better to let the 
# solr does the counting, 
def context_to_context_sim(mention, context, candidates, rows=100):
    """
    Description:
        Uses Solr to find the relevancy scores of the candidates based on the context.
    Args:
        mentionStr: The mention as it appears in the text
        context: The words that surround the target word.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
    Return:
        The score for each candidate in the same order as the candidates.
    """
    if not context:
        return [0]*len(candidates)
    
    # put text in right format
    context = solr_escape(context)
    mention = solr_escape(mention)
    
    filter_ids = " ".join(['entityid:' +  str(tid) for tid,_ in candidates])
    
    
    # select all the docs from Solr with the best scores, highest first.
    qstr = 'http://localhost:8983/solr/enwiki20160305_context/select'
    q="_context_:(%s) entity:(%s)" % (context,mention)
    q="_context_:(%s) " % (context)
    
    params={'fl':'entityid', 'fq':filter_ids, 'indent':'on',
            'q':q,'wt':'json', 'rows':rows}
    r = session.get(qstr, params = params)
    cnt = Counter()
    
    for doc in r.json()['response']['docs']:
        cnt[long(doc['entityid'])] += 1
    
    id_score=[cnt[c] for c,_ in candidates]
    return id_score


def context_candidate_scores (S, M, candslist, ws=5, method='c2c', skip_current=1):
    '''returns entity scores using  context seatch
       Inputs: 
           S: Sentence
           M: Mentions
           candslist: candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
            ws: word size
            method: Either 'c2p': for context to profile, or 'c2c' for context to context
            skip_current: Whether or not include the current mention in the context
       Returns:
           Scores [[s11,...s1k],...[sn1,...s1m]] where sij is cij similarity to the key-entity
    '''
    candslist_scores=[]
    for i in range(len(candslist)):
        cands = candslist[i]
        pos = M[i][0]
        mention=S[pos]
        context = S[max(pos-ws,0):pos]+S[pos+skip_current:pos+ws+1]
        context=" ".join(context)
        
        if method == 'c2p':
            cand_scores=context_to_profile_sim(mention, context, cands)
        if method == 'c2c':
            cand_scores=context_to_context_sim(mention, context, cands)
            
        candslist_scores.append(cand_scores) 

    return candslist_scores

def mention_to_title_sim(mention, candidates):
    """
    Description:
        Uses Solr to find the string similarity scores between the mention candidates.
    Args:
        mention: The mention as it appears in the text
        context: The words that surround the target word.
        candidates: A list of candidates that each have the entity id and its frequency/popularity.
    Return:
        The score for each candidate in the same order as the candidates.
    """
    
    
    # put text in right format
    mention = solr_escape(mention)
    
    filter_ids = " ".join(['id:' +  str(tid) for tid,_ in candidates])
        

    # select all the docs from Solr with the best scores, highest first.
    qst = 'http://localhost:8983/solr/enwiki20160305/select'
    q='title:(' + mention+')'
    
    params={'fl':'id score', 'fq':filter_ids, 'indent':'on',
            'q':q, 'wt':'json','rows':len(candidates)}
    
    
    r = session.get(qst, params = params).json()['response']['docs']
    id_score_map=defaultdict(float, {long(ri['id']):ri['score'] for ri in r})
    id_score=[id_score_map[c] for c,_ in candidates]
    return id_score

def mention_candidate_score(S, M, candslist):
    return [mention_to_title_sim(S[m[0]], c) for m,c in zip(M,candslist) ]

def popularity_score(candslist):
    """Retrieves the popularity score from the candslist
    """
    scores=[[s for _, s in cands] for cands in candslist]
    return scores

def normalize(scores_list):
    """Normalize a matrix, row-wise
    """
    normalized_scoreslist=[]
    for scores in scores_list:
        smooth=0
        if 0 in scores:
            smooth=1
        sum_s = sum(s+smooth for s in scores )        
        n_scores = [float(s+smooth)/sum_s for s in scores]
        normalized_scoreslist.append(n_scores)
    return normalized_scoreslist
        
def normalize_minmax(scores_list):
    """Normalize a matrix, row-wise, using minmax technique
    """
    normalized_scoreslist=[]
    for scores in scores_list:
        scores_min = min(scores)        
        scores_max = max(scores)        
        if scores_min == scores_max:
            n_scores = [0]*len(scores)
        else:
            n_scores = [(float(s)-scores_min)/(scores_max-scores_min) for s in scores]
        normalized_scoreslist.append(n_scores)
    return normalized_scoreslist

def find_max(candslist,candslist_scores):
    '''Disambiguate a sentence using a list of candidate-score tuples
       Inputs: 
           candslist: candidate list [[(c11, s11),...(c1k, s1k)],...[(cn1, sn1),...(c1m, s1m)]]
       Returns: 
           a list of entity ids and a list of titles
    '''
            
    true_entities = []
    for cands, cands_scores in zip(candslist, candslist_scores):
        max_index, max_value = max(enumerate(cands_scores), key= lambda x:x[1])
        true_entities.append(cands[max_index][0])
    titles = ids2title(true_entities)
    return true_entities, titles        

#Delete, useless
def disambiguate_random(C):
    '''Disambiguate using the given order (which can be random)
        Input:
            C: Candlist
        Output:
            Disambiguated entities
    '''
    
    ids = [c[0][0] for c in C ]
    titles= ids2title(ids)
    return ids, titles

def get_scores(S, M, C, method):
    """ Disambiguate C list using a disambiguation method 
        Inputs:
            S: Sentence
            M: Metntions
            C: Candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
            method: similarity method
            direction: embedding type
            op_method: disambiguation method 
                        most important ones: ilp (integer linear programming), 
                                             key: Key Entity based method
        
    """
    scores=None
    if method == 'popularity'  :
        scores = popularity_score(C)
    if method == 'keydisamb'  :
        scores = coherence_scores_driver(C, method='rvspagerank', direction=DIR_BOTH, op_method="keydisamb")
    if method == 'entitycontext'  :
        scores = coherence_scores_driver(C, method='rvspagerank', direction=DIR_BOTH, op_method="entitycontext")
    if method == 'mention2entity'  :
        scores = mention_candidate_score (S, M, C)
    if method == 'context2context'  :
        scores = context_candidate_scores (S, M, C, method='c2c')
    if method == 'context2profile'  :
        scores = context_candidate_scores (S, M, C, method='c2p')    
    if method == 'learned'  :
        scores = learned_scores (S, M, C)    
        
    #scores = normalize_minmax(scores)    
    return scores

def formated_scores(scores):
    """Only for pretty-printing
    """
    scores = [['{0:.2f}'.format(s) for s in cand_scores] for cand_scores in scores]
    return scores

def formated_all_scores(scores):
    """Only for pretty-printing
    """
    scores = [[tuple('{0:.2f}'.format(s) for s in sub_scores) for sub_scores in cand_scores] for cand_scores in scores]
    return scores

def get_all_scores(S, M, C):
    """Give all scores as different lists
        Inputs:
            S: segmented sentence [w1, ..., wn]
            M: mensions [m1, ... , mj]
            C: candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]

        Output:
            Scores, in this format [[(c111,.., c1k1),...(cm11,.., cmks)],...[(c1n1,.., pm1s),...(c1m1,.., p1ms)]]
            where cijk is the k-th scores for cij candidate
        
            Scores, in this format [[(c111, c11s),...(c1k1, c1ks)],...[(cn11, pn1s),...(c1m1, p1ms)]]
            where cijk is the k-th scores for cij candidate
    """
    all_scores= [get_scores(S, M, C, method) for method in \
           ['popularity','keydisamb','entitycontext','mention2entity','context2context','context2profile']]
    return [zip(*s) for s in zip(*all_scores)]




def learned_scores (S, M, candslist):
    '''returns entity scores using the learned (learned-to-rank method)
       Inputs: 
           S: Sentence
           M: Mentions
           candslist: candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
       Returns:
           Scores [[s11,...s1k],...[sn1,...s1m]] where sij is cij similarity to the key-entity
    '''
    if (wsd_model_preprocessor_ is None) or (wsd_model_ is None):
        log('[learned_scores]\tmodel not loaded')
        raise Exception('model not loaded, try load_wsd_model()')
    
    all_scores=get_all_scores(S,M,candslist)
    return [wsd_model_.predict(wsd_model_preprocessor_.transform(cand_scores)) for cand_scores in all_scores] 

def wsd(S, M, C, method='learned'):
    '''Gets a sentence, mentions and candslist, and returns disambiguation
       Inputs: 
           S: Sentence
           M: Mentions
           candslist: candidate list [[(c11, p11),...(c1k, p1k)],...[(cn1, pn1),...(c1m, p1m)]]
            method: disambiguation method 
       Returns: 
           A disambiguated list in the form of  (true_entities, titles)
    
    '''
    candslist_scores = get_scores(S, M, C, method)
    return find_max(C,candslist_scores)