
#%%system
mysql -u $1 -p$2 -e 'CREATE SCHEMA `enwikilast` DEFAULT CHARACTER SET binary;'
./importall  ~/Downloads/wikidumps last $1 $2

