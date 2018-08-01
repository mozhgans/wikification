import collections
import json

def build_vocab(words, min_count=5):
    count = [['UNK', -1]]
    count.extend([ (w,c) for w,c in collections.Counter(words).items()])
    vocab = dict()
    for word, c in count:
        if c >= min_count:
            vocab[word] = len(vocab)
    return count, vocab

def getwords(*filenames):
    words=[]
    for filename in filenames:
        with open(filename) as infile:
            for line in infile:
                ex = json.loads(line.decode('utf-8').strip())
                words += [str(n[0]) for n in ex["neg"]]
                if "left" in ex["context"]:
                    words += ex["context"]["left"].split()
                if "right" in ex["context"]:
                    words += ex["context"]["right"].split()
                words .append(ex["context"]["entityid"])
    return words
        
def integize(infile_name, outfile_name, vocab):
    with open(infile_name) as infile, open(outfile_name, 'w' ) as outfile:
        for line in infile:
            ex = json.loads(line.decode('utf-8').strip())
            
            neg =  [[vocab[str(n[0])],n[1]] for n in ex["neg"] if str(n[0]) in vocab]                
            if not neg or ex["context"]["entityid"] not in vocab:
                continue
                
            entityid  = vocab[ex["context"]["entityid"]]
                
            if "left" in ex["context"]:
                left = [vocab[w] for w in ex["context"]["left"].split() if w in vocab]
            if "right" in ex["context"]:
                right = [vocab[w] for w in ex["context"]["right"].split() if w in vocab]
            
            
            ex_id = {"neg": neg, 
                     "context": { "left": left, "entityid" : entityid, 
                                 "right": right, "freq":ex["freq"] },                     
                    }
            outfile.write(json.dumps(ex_id, ensure_ascii=False).encode('utf-8')+'\n')
        