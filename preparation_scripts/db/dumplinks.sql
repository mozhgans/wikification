SELECT *
INTO OUTFILE '~/backup/wikipedia/20160305/edited/enwiki-20160305-pagelinks.dumped..ssv'
CHARACTER SET binary FIELDS TERMINATED BY ' ' OPTIONALLY ENCLOSED BY '\''
FROM pagelinks; 

