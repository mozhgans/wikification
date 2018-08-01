# This is for extraction
# tail -n +2 MiniMayo.csv | awk -F',' '{gsub(/ /,"_",$5);gsub(/ /,"_",$6);print $5 "\t" $6 "\t" $2}' >> MiniMayo.orig.csv 

# maybe lowecase is better:
for f in *.csv; do
    awk  '{print tolower($0)}' $f > ${f%.*}.lower.csv
done
