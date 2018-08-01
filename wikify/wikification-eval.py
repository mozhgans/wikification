"""
This is for testing performance of wikification.

Instruction: Simply change whatever parameters you want in this file, 
some parameters are only changeable in wikification.py, so that will 
have to be edited too. You can edit the 'comment' variable in the
bottom of this script to describe what you are doing in the current
test. Run this script with the command: 'python wikification-eval.py' 
without the quotes. 
"""

from __future__ import division
from wikification import *
from IPython.display import clear_output
import copy
from datetime import datetime
import tagme
import os
import json
from sets import Set

tagme.GCUBE_TOKEN = "f6c2ba6c-751b-4977-a94c-c140c30e9b92-843339462"
pathStrt = '/users/cs/amaral/wsd-datasets'

# the data sets for performing on
datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')},
            {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')},
            {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},
            {'name':'wiki5000', 'path':os.path.join(pathStrt,'wiki-mentions.5000.json')}]

# many different option for combinations of datasets for smaller tests
#datasets = [{'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')}]
datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}]
#datasets = [{'name':'wiki5000', 'path':os.path.join(pathStrt,'wiki-mentions.5000.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}, {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},{'name':'nopop', 'path':os.path.join(pathStrt,'nopop.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}, {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},{'name':'wiki500', 'path':os.path.join(pathStrt,'wiki-mentions.500.json')}]
#datasets = [{'name':'nopop', 'path':os.path.join(pathStrt,'nopop.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}, {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},{'name':'wiki500', 'path':os.path.join(pathStrt,'wiki-mentions.500.json')},{'name':'nopop', 'path':os.path.join(pathStrt,'nopop.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}, {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},{'name':'wiki5000', 'path':os.path.join(pathStrt,'wiki-mentions.5000.json')},{'name':'nopop', 'path':os.path.join(pathStrt,'nopop.json')}]
#datasets = [{'name':'wiki500', 'path':os.path.join(pathStrt,'wiki-mentions.500.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}, {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},{'name':'wiki5000', 'path':os.path.join(pathStrt,'wiki-mentions.5000.json')},{'name':'nopop', 'path':os.path.join(pathStrt,'nopop.json')}]
#datasets = [{'name':'kore', 'path':os.path.join(pathStrt,'kore.json')}, {'name':'AQUAINT', 'path':os.path.join(pathStrt,'AQUAINT.txt.json')}, {'name':'MSNBC', 'path':os.path.join(pathStrt,'MSNBC.txt.json')},{'name':'nopop', 'path':os.path.join(pathStrt,'nopop.json')}]

# 'popular', 'context1', 'context2', 'word2vec', 'coherence', 'tagme', 'multi'
methods = ['multi']
# 'lmart', 'gbr', 'etr', 'rfr'
mlModel = 'lmart' # to be used with method multi
erMethod = 'cls1' # method for entity recognition / mention extraction

if 'word2vec' in methods:
    try:
        word2vec
    except:
        word2vec = gensim_loadmodel('/users/cs/amaral/cgmdir/WikipediaClean5Negative300Skip10.Ehsan/WikipediaClean5Negative300Skip10')
        
# can do both, none would be pointless
doSplit = True # mentions are given
doManual = False # mentions not given

verbose = True # decides how much stuff to ouput

maxCands = 20 # amount of candidates for entity candidate generation (20 prefered)
doHybrid = False # whether to do hybrid candidate generation (False prefered)


performances = {} # record data here

skipped = 0
badThing = []

# for each dataset, run all methods
for dataset in datasets:
    performances[dataset['name']] = {}
    # get the data from dataset
    dataFile = open(dataset['path'], 'r')
    dataLines = []
    
    # get all lines
    for line in dataFile:
        dataLines.append(json.loads(line.decode('utf-8').strip()))
        
    print '\n' + dataset['name'] + '\n'
    
    # run each method on the data set
    for mthd in methods:
        print mthd
        print str(datetime.now()) + '\n'
        
        ## reset counters
        # micro scores
        totalMicroPrecS = 0
        totalMicroPrecM = 0
        totalMicroRecS = 0
        totalMicroRecM = 0
        # macro scores
        totalMacroPrecS = 0
        totalMacroPrecM = 0
        totalMacroRecS = 0
        totalMacroRecM = 0
        # BOT micro scores
        totalBotMicroPrecS = 0
        totalBotMicroPrecM = 0
        totalBotMicroRecS = 0
        totalBotMicroRecM = 0
        # BOT macro scores
        totalBotMacroPrecS = 0
        totalBotMacroPrecM = 0
        totalBotMacroRecS = 0
        totalBotMacroRecM = 0
        # amount of lines done in dataset
        totalLines = 0
        # amount of mentions in dataset
        totalMentions = 0 
        # amount of mentions found
        totalMyMentionsS = 0
        totalMyMentionsM = 0
        
        # each method tests all lines
        for line in dataLines:
            if verbose:
                print str(totalLines + 1)
            
            # get absolute text indexes and entity id of each given mention
            trueEntities = mentionStartsAndEnds(copy.deepcopy(line), forTruth = True) # the ground truth
            
            oData = copy.deepcopy(line) # copy of the line data
            
            # get results for pre split string
            if doSplit and mthd <> 'tagme': # presplit no work on tagme
                # original split string with mentions given
                try:
                    resultS = wikifyEval(copy.deepcopy(line), True, hybridC = doHybrid, maxC = maxCands, 
                                         method = mthd, model = mlModel, erMethod = erMethod)
                except:
                    skipped += 1
                    badThing.append(line)
                    continue
                precS = precision(trueEntities, resultS) # precision of pre-split
                recS = recall(trueEntities, resultS) # recall of pre-split
                
                # micro scores
                totalMicroPrecS += len(resultS) * precS
                totalMicroRecS += len(trueEntities) * recS
                # macro scores
                totalMacroPrecS += precS
                totalMacroRecS += recS
                
                # get bot scores
                trueSet = Set()
                for truEnt in trueEntities:
                    trueSet.add(truEnt[2])
                mySet = Set()
                for res in resultS:
                    mySet.add(res[2])
                    
                try:
                    precS = len(trueSet & mySet)/len(mySet)
                except:
                    precS = 0
                    
                try:
                    recS = len(trueSet & mySet)/len(trueSet)
                except:
                    recS = 0
                
                # BOT micro scores
                totalBotMicroPrecS += len(trueSet) * precS
                totalBotMicroRecS += len(trueSet) * recS
                # BOT macro scores
                totalBotMacroPrecS += precS
                totalBotMacroRecS += recS
                
                totalMyMentionsS += len(resultS)
                
                if verbose:
                    print 'Split: ' + str(precS) + ', ' + str(recS)
                
            # get results for manually split string
            if doManual:
                # tagme has separate way to do things
                if mthd == 'tagme':
                    antns = tagme.annotate(" ".join(line['text']))
                    resultM = []
                    for an in antns.get_annotations(0.005):
                        resultM.append([an.begin,an.end,title2id(an.entity_title)])
                else:
                    # unsplit string to be manually split and mentions found
                    try:
                        resultM = wikifyEval(" ".join(line['text']), False, hybridC = doHybrid, 
                                         maxC = maxCands, method = mthd, model = mlModel, erMethod = erMethod)
                    except:
                        skipped += 1
                        badThing.append(line)
                        continue
                
                precM = precision(trueEntities, resultM) # precision of manual split
                recM = recall(trueEntities, resultM) # recall of manual split
                
                """
                I think the math for micro scores are wrong in manual
                """
                
                # micro scores
                totalMicroPrecM += len(trueEntities) * precM
                totalMicroRecM += len(trueEntities) * recM
                # macro scores
                totalMacroPrecM += precM
                totalMacroRecM += recM
                
                # get bot scores
                trueSet = Set()
                for truEnt in trueEntities:
                    trueSet.add(truEnt[2])
                mySet = Set()
                for res in resultM:
                    mySet.add(res[2])
                    
                try:
                    precM = len(trueSet & mySet)/len(mySet)
                except:
                    precM = 0
                    
                try:
                    recM = len(trueSet & mySet)/len(trueSet)
                except:
                    recM = 0
                
                # BOT micro scores
                totalBotMicroPrecM += len(trueSet) * precM
                totalBotMicroRecM += len(trueSet) * recM
                # BOT macro scores
                totalBotMacroPrecM += precM
                totalBotMacroRecM += recM
                
                totalMyMentionsM += len(resultM)
                
                if verbose:
                    print 'Manual: ' + str(precM) + ', ' + str(recM)
                
            totalLines += 1
            totalMentions += len(trueEntities)
        
        # record results for this method on this dataset
        # all F1 scores are put in later to avoid division by 0 possibility
        
        # to stop errors
        if totalMyMentionsS == 0:
            totalMyMentionsS = -1
        if totalMyMentionsM == 0:
            totalMyMentionsM = -1
        if totalMentions == 0:
            totalMentions = -1
        
        performances[dataset['name']][mthd] = {
                   'S Micro Prec':totalMicroPrecS/totalMyMentionsS, 
                   'M Micro Prec':totalMicroPrecM/totalMyMentionsM,
                   'S Micro Rec':totalMicroRecS/totalMentions, 
                   'M Micro Rec':totalMicroRecM/totalMentions,

                   'S Macro Prec':totalMacroPrecS/totalLines,
                   'M Macro Prec':totalMacroPrecM/totalLines,
                   'S Macro Rec':totalMacroRecS/totalLines, 
                   'M Macro Rec':totalMacroRecM/totalLines,

                   'S BOT Micro Prec':totalBotMicroPrecS/totalMyMentionsS, 
                   'M BOT Micro Prec':totalBotMicroPrecM/totalMyMentionsM,
                   'S BOT Micro Rec':totalBotMicroRecS/totalMentions, 
                   'M BOT Micro Rec':totalBotMicroRecM/totalMentions,

                   'S BOT Macro Prec':totalBotMacroPrecS/totalLines,
                   'M BOT Macro Prec':totalBotMacroPrecM/totalLines,
                   'S BOT Macro Rec':totalBotMacroRecS/totalLines, 
                   'M BOT Macro Rec':totalBotMacroRecM/totalLines
                   }
        
        perf = performances[dataset['name']][mthd]
        # hande divisions by 0 for F1 scores here
        # s and m micro f1
        try:
            performances[dataset['name']][mthd]['S Micro F1'] = (
                    (2*perf['S Micro Prec']*perf['S Micro Rec'])/(perf['S Micro Prec']+perf['S Micro Rec']))
        except:
            performances[dataset['name']][mthd]['S Micro F1'] = 0
            
        try:
            performances[dataset['name']][mthd]['M Micro F1'] = (
                    (2*perf['M Micro Prec']*perf['M Micro Rec'])/(perf['M Micro Prec']+perf['M Micro Rec']))
        except:
            performances[dataset['name']][mthd]['M Micro F1'] = 0
        
        # s and m macro f1
        try:
            performances[dataset['name']][mthd]['S Macro F1'] = (
                    (2*perf['S Macro Prec']*perf['S Macro Rec'])/(perf['S Macro Prec']+perf['S Macro Rec']))
        except:
            performances[dataset['name']][mthd]['S Macro F1'] = 0
            
        try:
            performances[dataset['name']][mthd]['M Macro F1'] = (
                    (2*perf['M Macro Prec']*perf['M Macro Rec'])/(perf['M Macro Prec']+perf['M Macro Rec']))
        except:
            performances[dataset['name']][mthd]['M Macro F1'] = 0
        
        # s and m, micro and macro for BOT f1
        try:
            performances[dataset['name']][mthd]['S BOT Micro F1'] = (
                    (2*perf['S BOT Micro Prec']*perf['S BOT Micro Rec'])/(perf['S BOT Micro Prec']+perf['S BOT Micro Rec']))
        except:
            performances[dataset['name']][mthd]['S BOT Micro F1'] = 0
            
        try:
            performances[dataset['name']][mthd]['M BOT Micro F1'] = (
                    (2*perf['M BOT Micro Prec']*perf['M BOT Micro Rec'])/(perf['M BOT Micro Prec']+perf['M BOT Micro Rec']))
        except:
            performances[dataset['name']][mthd]['M BOT Micro F1'] = 0
        
        try:
            performances[dataset['name']][mthd]['S BOT Macro F1'] = (
                    (2*perf['S BOT Macro Prec']*perf['S BOT Macro Rec'])/(perf['S BOT Macro Prec']+perf['S BOT Macro Rec']))
        except:
            performances[dataset['name']][mthd]['S BOT Macro F1'] = 0
            
        try:
            performances[dataset['name']][mthd]['M BOT Macro F1'] = (
                    (2*perf['M BOT Micro Prec']*perf['M BOT Micro Rec'])/(perf['M BOT Micro Prec']+perf['M BOT Micro Rec']))
        except:
            performances[dataset['name']][mthd]['M BOT Macro F1'] = 0
        
print 'Skipped', skipped
print badThing

with open('/users/cs/amaral/wikisim/wikification/wikification_results.txt', 'a') as resultFile:
    
    resultFile.write('\n' + str(datetime.now()) + '\n' 
                     + 'maxCands: ' + str(maxCands) + '\n'
                     + 'mlModel: ' + mlModel + '\n'
                     + 'erMethod: ' + erMethod + '\n'
                     + 'doHybrid: ' + str(doHybrid) + '\n'
                     + str(datetime.now()) + '\n\n')
    
    comment = 'Please make sure you delete this test.'
    resultFile.write('Comment: ' + comment + '\n\n')
    
    for dataset in datasets:
        resultFile.write(dataset['name'] + ':\n')
        for mthd in methods:
            if doSplit and doManual:
                resultFile.write(mthd + ':'
                       + '\n    S Micro Prec :' + str(performances[dataset['name']][mthd]['S Micro Prec'])
                       + '\n    S Micro Rec :' + str(performances[dataset['name']][mthd]['S Micro Rec']) 
                       + '\n    S Micro F1 :' + str(performances[dataset['name']][mthd]['S Micro F1'])
                       + '\n    S Macro Prec :' + str(performances[dataset['name']][mthd]['S Macro Prec'])
                       + '\n    S Macro Rec :' + str(performances[dataset['name']][mthd]['S Macro Rec']) 
                       + '\n    S Macro F1 :' + str(performances[dataset['name']][mthd]['S Macro F1'])
                       + '\n    S BOT Micro Prec :' + str(performances[dataset['name']][mthd]['S BOT Micro Prec'])
                       + '\n    S BOT Micro Rec :' + str(performances[dataset['name']][mthd]['S BOT Micro Rec']) 
                       + '\n    S BOT Micro F1 :' + str(performances[dataset['name']][mthd]['S BOT Micro F1'])
                       + '\n    S BOT Macro Prec :' + str(performances[dataset['name']][mthd]['S BOT Macro Prec'])
                       + '\n    S BOT Macro Rec :' + str(performances[dataset['name']][mthd]['S BOT Macro Rec']) 
                       + '\n    S BOT Macro F1 :' + str(performances[dataset['name']][mthd]['S BOT Macro F1']) + '\n'
                       + '\n    M Micro Prec :' + str(performances[dataset['name']][mthd]['M Micro Prec'])
                       + '\n    M Micro Rec :' + str(performances[dataset['name']][mthd]['M Micro Rec']) 
                       + '\n    M Micro F1 :' + str(performances[dataset['name']][mthd]['M Micro F1'])
                       + '\n    M Macro Prec :' + str(performances[dataset['name']][mthd]['M Macro Prec'])
                       + '\n    M Macro Rec :' + str(performances[dataset['name']][mthd]['M Macro Rec']) 
                       + '\n    M Macro F1 :' + str(performances[dataset['name']][mthd]['M Macro F1'])
                       + '\n    M BOT Micro Prec :' + str(performances[dataset['name']][mthd]['M BOT Micro Prec'])
                       + '\n    M BOT Micro Rec :' + str(performances[dataset['name']][mthd]['M BOT Micro Rec']) 
                       + '\n    M BOT Micro F1 :' + str(performances[dataset['name']][mthd]['M BOT Micro F1'])
                       + '\n    M BOT Macro Prec :' + str(performances[dataset['name']][mthd]['M BOT Macro Prec'])
                       + '\n    M BOT Macro Rec :' + str(performances[dataset['name']][mthd]['M BOT Macro Rec']) 
                       + '\n    M BOT Macro F1 :' + str(performances[dataset['name']][mthd]['M BOT Macro F1']) + '\n')
            elif doSplit:
                resultFile.write(mthd + ':'
                       + '\n    S Micro Prec :' + str(performances[dataset['name']][mthd]['S Micro Prec'])
                       + '\n    S Micro Rec :' + str(performances[dataset['name']][mthd]['S Micro Rec']) 
                       + '\n    S Micro F1 :' + str(performances[dataset['name']][mthd]['S Micro F1'])
                       + '\n    S Macro Prec :' + str(performances[dataset['name']][mthd]['S Macro Prec'])
                       + '\n    S Macro Rec :' + str(performances[dataset['name']][mthd]['S Macro Rec']) 
                       + '\n    S Macro F1 :' + str(performances[dataset['name']][mthd]['S Macro F1'])
                       + '\n    S BOT Micro Prec :' + str(performances[dataset['name']][mthd]['S BOT Micro Prec'])
                       + '\n    S BOT Micro Rec :' + str(performances[dataset['name']][mthd]['S BOT Micro Rec']) 
                       + '\n    S BOT Micro F1 :' + str(performances[dataset['name']][mthd]['S BOT Micro F1'])
                       + '\n    S BOT Macro Prec :' + str(performances[dataset['name']][mthd]['S BOT Macro Prec'])
                       + '\n    S BOT Macro Rec :' + str(performances[dataset['name']][mthd]['S BOT Macro Rec']) 
                       + '\n    S BOT Macro F1 :' + str(performances[dataset['name']][mthd]['S BOT Macro F1']) + '\n')
            elif doManual:
                resultFile.write(mthd + ':'
                       + '\n    M Micro Prec :' + str(performances[dataset['name']][mthd]['M Micro Prec'])
                       + '\n    M Micro Rec :' + str(performances[dataset['name']][mthd]['M Micro Rec']) 
                       + '\n    M Micro F1 :' + str(performances[dataset['name']][mthd]['M Micro F1'])
                       + '\n    M Macro Prec :' + str(performances[dataset['name']][mthd]['M Macro Prec'])
                       + '\n    M Macro Rec :' + str(performances[dataset['name']][mthd]['M Macro Rec']) 
                       + '\n    M Macro F1 :' + str(performances[dataset['name']][mthd]['M Macro F1'])
                       + '\n    M BOT Micro Prec :' + str(performances[dataset['name']][mthd]['M BOT Micro Prec'])
                       + '\n    M BOT Micro Rec :' + str(performances[dataset['name']][mthd]['M BOT Micro Rec']) 
                       + '\n    M BOT Micro F1 :' + str(performances[dataset['name']][mthd]['M BOT Micro F1'])
                       + '\n    M BOT Macro Prec :' + str(performances[dataset['name']][mthd]['M BOT Macro Prec'])
                       + '\n    M BOT Macro Rec :' + str(performances[dataset['name']][mthd]['M BOT Macro Rec']) 
                       + '\n    M BOT Macro F1 :' + str(performances[dataset['name']][mthd]['M BOT Macro F1']) + '\n')
                
    resultFile.write('\n' + '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~' + '\n')