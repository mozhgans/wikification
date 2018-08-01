
#%%system
gunzip -c $1/enwiki-latest-page.sql.gz		\
		> $1/enwiki-latest-page.sql 		
gunzip -c $1/enwiki-latest-pagelinks.sql.gz 	\
		> $1/enwiki-latest-pagelinks.sql	
gunzip -c $1/enwiki-latest-redirect.sql.gz		\
		> $1/enwiki-latest-redirect.sql	
gunzip -c $1/enwiki-latest-category.sql.gz		\
		> $1/enwiki-latest-category.sql	
gunzip -c $1/enwiki-latest-categorylinks.sql.gz	\
		> $1/enwiki-latest-categorylinks.sql	
