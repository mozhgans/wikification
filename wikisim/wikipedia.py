"""A General Class to interact with Wiki datasets"""
# uncomment

import sys;
import os
import scipy as sp
import pandas as pd
import cPickle as pickle
import MySQLdb
from collections import defaultdict

from utils import * # uncomment

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 215, The Wikisim Project"
__credits__ = ["Armin Sajadi", "Evangelo Milios", "Armin Sajadi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Armin Sajadi"
__email__ = "sajadi@cs.dal.ca"
__status__ = "Development"


DISABLE_CACHE=False;
MAX_GRAPH_SIZE=1000000

DIR_IN=0;
DIR_OUT=1;
DIR_BOTH=2;
_db = MySQLdb.connect(host="127.0.0.1",port=3307,user='root',passwd="emilios",db="enwiki20160305")
_cursor = _db.cursor()
#WIKI_SIZE = 10216236;
#WIKI_SIZE = 13670498; #2016
WIKI_SIZE = 5576365; #no redirect, 2016
def close():
    global _db, _cursor;
    if _cursor is not None: 
        _cursor.close();
        _db.close();
    _cursor=_db=None;
def reopen():
    global _db, _cursor;
    if _db is None:
        _db = MySQLdb.connect(host="127.0.0.1",port=3307,user='root',passwd="emilios",db="enwiki20160305")
        _cursor = _db.cursor()

def disable_cache():
    global DISABLE_CACHE
    DISABLE_CACHE=True
def enable_cache():
    global DISABLE_CACHE
    DISABLE_CACHE=False
        
def load_table(tbname, limit=-1):
    """ Returns a list, containing a whole table     
    
    Args: 
        tbname: Table Name
    Returns: 
        The list of rows
    """
    if limit!=-1:
        q = """SELECT * FROM `%s` limit %s""" % (tbname, limit)
    else:
        q = """SELECT * FROM `%s`""" % (tbname,)
        
    _cursor.execute(q)
    rows = _cursor.fetchall();
    return rows
    
    
def id2title(wid):
    """ Returns the title for a given id

    Args: 
        wid: Wikipedia id       
    Returns: 
        The title of the page
    """
    title=None;

    _cursor.execute("""SELECT * FROM `page` where page_id = %s""", (wid,))
    row= _cursor.fetchone();
    if row is not None:
        title=row[2];          
    return title;

def ids2title(wids):
    """ Returns the titles for given list of wikipedia ids 

    Args: 
        wids: A list of Wikipedia ids          
    Returns: 
        The list of titles
    """
    if not wids:
        return []
    wid_list = [str(wid) for wid in wids] ;
    order = ','.join(['page_id'] + wid_list) ;
    wid_str = ",".join(wid_list)
    query = "SELECT page_id, page_title FROM `page` where page_id in ({0})" \
    .format(wid_str, order);
    _cursor.execute(query);
    rows = _cursor.fetchall();
    rows_dict = defaultdict(lambda: None, rows)
    titles = [rows_dict[wid] for wid in wids]
    return titles;

def encode_for_db(instr):
    if isinstance(instr, unicode):
        instr = instr.encode('utf-8')  
    return instr
        
def normalize_str(title):
    
    title = encode_for_db(title)
    title = title.replace(' ','_')
    return title
def title2id(title):
    """ Returns the id for a given title

    Args: 
        wid: Wikipedia id          
    Returns: 
        The title of the page
    """        
    if title is None:
        return None
    wid=None;
    title = normalize_str(title)
    _cursor.execute("""SELECT * FROM `page` where page_title=%s and page_namespace=0""", (title,))
    row= _cursor.fetchone();
    if row is not None:
        wid = getredir_id(row[0]) if row[3] else row[0];
    return wid;

def is_ambiguous(wid):
    _cursor.execute("""SELECT * FROM `categorylinks` WHERE `categorylinks`.cl_from=%s and `categorylinks`.cl_to=19204864;""", (wid,))
    row= _cursor.fetchone();
    return not (row is None)    

def getredir_id(wid):
    """ Returns the target of a redirected page 

    Args:
        wid: wikipedia id of the page
    Returns:
        The id of the target page
    """
    rid=None

    _cursor.execute("""select * from redirect where rd_from=%s;""", (wid,));
    row= _cursor.fetchone();
    if row is not None:
        rid=row[1]
    return rid 

def resolveredir(wid):
    tid = getredir_id(wid);
    if tid is not None:
        wid = tid;    
    return wid

def getredir_title(wid):
    """ Returns the target title of a redirected page 

    Args:
        wid: wikipedia id of the page
    Returns:
        The title of the target page
    """
    
    title=None;
    _cursor.execute(""" select page_title from redirect INNER JOIN page
                  on redirect.rd_to = page.page_id 
                  where redirect.rd_from =%s;""", (wid));
    row=_cursor.fetchone()
    if row is not  None:
        title=row[0];
    return title;

def synonymring_titles(wid):
    """ Returns the synonim ring of a page

    Example: synonymring_titles('USA')={('U.S.A', 'US', 'United_States_of_America', ...)}

    Args:
        wid: the wikipedia id
    Returns:
        all the titles in its synonym ring
    """
    wid = resolveredir(wid)
    _cursor.execute("""(select page_title from page where page_id=%s) union 
                 (select page_title from redirect INNER JOIN page
                    on redirect.rd_from = page.page_id 
                    where redirect.rd_to =%s);""", (wid,wid));
    rows=_cursor.fetchall();
    if rows:
        rows = tuple(r[0] for r in rows)
    return rows;


def anchor2concept(anchor):
    """ Returns the targets of an anchor text

    Args:
        anchor: anchor
        
    Returns:
        The list of the titles of the linked pages
    """
  
    anchor = encode_for_db(anchor)
        
    _cursor.execute("""select anchors.id, anchors.freq from anchors inner join page on anchors.id=page.page_id where anchors.anchor=%s;""", (anchor,))
    rows =_cursor.fetchall()
#     if rows:
#         rows = tuple(r[0] for r in rows)
    return rows


def id2anchor(wid):
    """ Returns the targets of an anchor text

    Args:
        anchor: anchor
        
    Returns:
        The list of the titles of the linked pages
    """
    _cursor.execute("""select anchor , freq from anchors where id=%s""", (wid,))
    rows =_cursor.fetchall()
#     if rows:
#         rows = tuple(r[0] for r in rows)
    return rows


def _getlinkedpages_query(id, direction):
    query="(SELECT {0} as lid FROM pagelinks where ({1} = {2}))"
    if direction == DIR_IN:
        query=query.format("pl_from","pl_to",id);
    elif direction == DIR_OUT:
        query=query.format("pl_to","pl_from",id);
    return query;

def getlinkedpages(wid,direction):
    """ Returns the linkage for a node

    Args:
        id: the wikipedia id
        direction: 0 for in, 1 for out, 2 for all
    Returns:
        The list of the ids of the linked pages
    """
    _cursor.execute(_getlinkedpages_query(wid, direction));
    rows =_cursor.fetchall()
    if rows:
        rows = tuple(r[0] for r in rows)
    return rows

def e2i(wids):
    elist=[];
    edict=dict();
    last=0;    
    for wid in itertools.chain(*iters):
        if wid not in edict:
            edict[wid]=last;
            elist.append(wid);
            last +=1; 
    return elist, edict;

def getneighbors(wid, direction):
    """ Returns the neighborhood for a node

    Args:
        id: the wikipedia id
        direction: 0 for in, 1 for out, 2 for all
    Returns:
        The vector of ids, and the 2d array sparse representation of the graph, in the form of
        array([[row1,col1],[row2, col2]]). This form is flexible for general use or be converted to scipy.sparse 
        formats
    """
    log('[getneighbors started]\twid = %s, direction = %s', wid, direction)
    
    idsquery = """(select  {0} as lid) union {1}""".format(wid,_getlinkedpages_query(wid,direction));

    _cursor.execute(idsquery);


    rows = _cursor.fetchall();
    if len(rows)<2:
        log('[getneighbors]\tERROR: empty')
        return (), sp.array([])
    
    
    neighids = tuple(r[0] for r in rows);
    if len(neighids)>MAX_GRAPH_SIZE:
        log('[getneighbors]\tERROR: too big, %s neighbors', len(neighids))
        return (), sp.array([])

    
    id2row = dict(zip(neighids, range(len(neighids))))

    neighbquery=  """select lid,pl_to as n_l_to from
                     ({0}) a  inner join
                     pagelinks on lid=pl_from""".format(idsquery);

    links=_cursor.execute(neighbquery);

    links = _cursor.fetchall();
    
    #links = tuple((id2row(u), id2row(v)) for u, v in links if (u in id2row) and (v in id2row));
    links = sp.array([[id2row[u], id2row[v]] for u, v in links if (u in id2row) and (v in id2row)]);
    
    log('Graph extracted, %s nodes and %s linkes', len(neighids), len(links) )
    log('[getneighbors]\tfinished')
    return (neighids,links)

def deletefromcache(wid, direction):
    wid = resolveredir(wid)
    if direction in [DIR_IN, DIR_BOTH] : 
        query =    """delete from {0} where cache_id={1}""".format('pagelinksorderedin', wid) 
        _cursor.execute(query);
    if direction in [DIR_OUT, DIR_BOTH]: 
        query =    """delete from {0} where cache_id={1}""".format('pagelinksorderedout', wid) 
        _cursor.execute(query);
    
def clearcache():
    if DISABLE_CACHE:
        return;
    _cursor.execute("delete  from pagelinksorderedin");
    _cursor.execute("delete  from pagelinksorderedout");

def checkcache(wid, direction):
    log('[checkcache started]\twid = %s, direction = %s', wid, direction)
    if DISABLE_CACHE:
        log('[checkcache]\tDisabled')
        return None
    

    
    em=None
    
    if direction == DIR_IN: 
        tablename = 'pagelinksorderedin';
        colname = 'in_neighb'
    elif direction == DIR_OUT: 
        tablename = 'pagelinksorderedout';
        colname = 'out_neighb';
    query =    """select {0} from {1} where cache_id={2}""".format(colname, tablename, wid)
    _cursor.execute(query);
    row = _cursor.fetchone();
    if row is not None:
        values, index = pickle.loads(row[0])
        log('[checkcache]\tfound')
        if not index:        
            log('[checkcache]\tempty embedding')
        em=pd.Series(values, index=index)
    else:
        log('[checkcache]\tnot found')

    log('[checkcache]\tfinished')
    return em


def cachescores(wid, em, direction):
    log('[cachescores started]\twid = %s, direction = %s', wid, direction)
    if DISABLE_CACHE:
        log('[cachescores]\tDisabled')
        return

    if direction == DIR_IN: 
        tablename = 'pagelinksorderedin';
        colname = 'in_neighb'

    elif direction == DIR_OUT: 
        tablename = 'pagelinksorderedout';
        colname = 'out_neighb';
        
    idscstr = pickle.dumps((em.values.tolist(), em.index.values.tolist()), pickle.HIGHEST_PROTOCOL)
    _cursor.execute("""insert into %s values (%s,'%s');""" %(tablename, wid, _db.escape_string(idscstr)));
    
    
    log('cachescores finished')