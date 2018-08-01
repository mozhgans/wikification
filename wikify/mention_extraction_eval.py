
"""
This evaluates the quality of mention extraction, macro scores.
"""

from __future__ import division
import requests
import json
import os
from wikification import *
from datetime import datetime
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

pathStrt = '/users/cs/amaral/wsd-datasets'
#pathStrt = 'C:\\Temp\\wsd-datasets'

# the data sets for performing on
datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')},
            {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')},
            {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},
            {'name':'wiki5000', 'path':os.path.join(pathStrt,'wiki-mentions.5000.json')}]

# short for quick tests
#datasets = [{'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}]
#datasets = [{'name':'wiki5000', 'path':os.path.join(pathStrt,'wiki-mentions.5000.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}, {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')}]

performances = {}

verbose = True

# for each dataset, run all methods
for dataset in datasets:
    performances[dataset['name']] = {}
    # get the data from dataset
    dataFile = open(dataset['path'], 'r')
    dataLines = []
    for line in dataFile:
        dataLines.append(json.loads(line.decode('utf-8').strip()))
    
    # reset counters
    totalPrec = 0
    totalRec = 0
    totalF1 = 0
    totalLines = 0

    # each method tests all lines
    for line in dataLines:

        if(verbose):
            print str(totalLines + 1)

        trueMentions = mentionStartsAndEnds(line, True)
        
        """
        output = nlp.annotate(" ".join(line['text']).encode('utf-8'), properties={
            'annotators': 'entitymentions',
            'outputFormat': 'json'
        })
        myMentions = []
        for sentence in output['sentences']:
            for em in sentence['entitymentions']:
                myMentions.append([em['characterOffsetBegin'], em['characterOffsetEnd']])
        
        """
        """"""
        myMentions = mentionExtract(" ".join(line['text']), mthd = 'cls1')['mentions']
        
        # put in right format
        #print trueMentions
        #print myMentions
        for mention in myMentions:
            mention[0] = mention[1]
            mention[1] = mention[2]
        """"""
            
        prec = mentionPrecision(trueMentions, myMentions)
        rec = mentionRecall(trueMentions, myMentions)
        try:
            f1 = (2*prec*rec)/(prec+rec)
        except:
            f1 = 0
        
        if(verbose):
            print str(prec) + ' ' + str(rec) + ' ' + str(f1) + '\n'

        # track results
        totalPrec += prec
        totalRec += rec
        totalF1 += f1
        totalLines += 1

    # record results for this method on this dataset
    performances[dataset['name']] = {'Precision':totalPrec/totalLines, 
                                     'Recall':totalRec/totalLines,
                                     'F1':totalF1/totalLines}
            
with open('/users/cs/amaral/wikisim/wikification/mention_extraction_results.txt', 'a') as resultFile:
    resultFile.write(str(datetime.now()) + '\n')
    resultFile.write('Using cls1 (longest dominant right) with initial filter and limit amount.' + '\n\n')
    for dataset in datasets:
        resultFile.write(dataset['name'] + ':\n')
        resultFile.write('\n    Prec :' + str(performances[dataset['name']]['Precision'])
               + '\n    Rec :' + str(performances[dataset['name']]['Recall'])
               + '\n    F1 :' + str(performances[dataset['name']]['F1']) + '\n')
                
    resultFile.write('\n' + '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~' + '\n')