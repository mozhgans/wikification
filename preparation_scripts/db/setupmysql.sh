
#%%system

mysql -u $1 -p$2 -e 'set global key_buffer_size=4*1024*1024*1024;'
mysql -u $1 -p$2 -e 'set global bulk_insert_buffer_size=1*1024*1024*1024;'
mysql -u $1 -p$2 -e 'set global query_cache_size = 4*1024*1024*1024;'
mysql -u $1 -p$2 -e 'set global query_cache_limit = 4*1024*1024*1024;'
mysql -u $1 -p$2 -e 'set global tmp_table_size = 4*1024*1024*1024;'