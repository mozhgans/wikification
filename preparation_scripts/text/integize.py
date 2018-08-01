import os
from utils import *
home = os.path.expanduser("~")
filepattern='False.0'
train_name_w = os.path.join(home, 'backup/datasets/cmod/train.%s.json'%(filepattern,))
train_name = os.path.join(home, 'backup/datasets/cmod/train.id.%s.json'%(filepattern,))
test_name_w = os.path.join(home, 'backup/datasets/cmod/test.%s.json'%(filepattern,))
test_name = os.path.join(home, 'backup/datasets/cmod/test.id.%s.json'%(filepattern,))

words = getwords(train_name_w, test_name_w)
count, vocab = build_vocab(words, min_count=5)
with open(os.path.join(home, 'backup/datasets/cmod/vocab.%s.tsv'%(filepattern,)), 'w') as out:
    out.write(json.dumps({"orig_size": len(count), "size": len(vocab)})+'\n')
    out.write(json.dumps(count, ensure_ascii=False).encode('utf-8')+'\n')
    out.write(json.dumps(vocab, ensure_ascii=False).encode('utf-8')+'\n')


integize(train_name_w, train_name, vocab)
integize(test_name_w, test_name, vocab)
