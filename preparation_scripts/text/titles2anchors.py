''' Add the titles to anchor list
'''
import MySQLdb

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 215, The Wikisim Project"
__credits__ = ["Armin Sajadi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Armin Sajadi"
__email__ = "sajadi@cs.dal.ca"
__status__ = "Development"

_db = MySQLdb.connect(host="127.0.0.1",port=3307,user='root',passwd="emilios",db="enwiki20160305")
_cursor = _db.cursor()
_cursor.execute("select page_title from page where page_namespace=0");
rows = _cursor.fetchall();
for row in rows:
    title=row[0]
    surface_form = title;
    i=surface_form.find('(')
    if i!=-1:
        surface_form = surface_form[:i]
    surface_form = surface_form.replace('_', ' ')
    surface_form = surface_form.strip()
    
    print surface_form+"\t"+title
    