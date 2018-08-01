import json
with open('/users/grad/sajadi/backup/wikipedia/20160305/texts/enwiki-20160305-annonated.json') as infile , open('/users/grad/sajadi/backup/wikipedia/20160305/texts/enwiki-20160305-cleantext.json', 'w') as outfile:
    for line in infile.readlines(): 
        doc = json.loads(line)
        outfile.write(doc['title'].encode('utf-8')+'\n')
        outfile.write(doc['text'].encode('utf-8')+'\n\n')
        #out.write(line)
print "done"    