SELECT *
INTO OUTFILE '~/backup/wikipedia/20160305/edited/enwiki-20160305-pagelinksorderedout.main.tsv'
CHARACTER SET binary FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\''
FROM pagelinksorderedout; 

