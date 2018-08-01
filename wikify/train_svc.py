''' Trains an SVC for mention detection
'''
from mention_detection import *

import numpy as np
import os
import pandas as pd
import sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn_pandas import gen_features
from sklearn.externals import joblib

__author__ = "Armin Sajadi"
__copyright__ = "Copyright 215, The Wikisim Project"
__credits__ = ["Armin Sajadi"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Armin Sajadi"
__email__ = "sajadi@cs.dal.ca"
__status__ = "Development"

def downsample_negatives(X_train, y_train, frac=0.2):

    pos_index = y_train==1
    X_train_pos = X_train[pos_index,:]
    y_train_pos = y_train[pos_index]

    neg_index = y_train==0
    X_train_neg = X_train[neg_index,:]
    y_train_neg = y_train[neg_index]


    X_train_neg, y_train_neg = sklearn.utils.resample(X_train_neg, y_train_neg, 
                                                n_samples = int(frac*len(X_train_neg)), replace=False)    

    X_train_downsampled = np.vstack([X_train_pos, X_train_neg])
    y_train_downsampled = np.hstack([y_train_pos, y_train_neg])

    X_train_downsampled_shuffled, y_train_downsampled_shuffled = sklearn.utils.shuffle(X_train_downsampled, y_train_downsampled)
    return X_train_downsampled_shuffled, y_train_downsampled_shuffled

home = '/users/grad/sajadi'

tr_file_name = os.path.join('../datasets/ner/mentiontrainrepository.5000.30000.tsv')
pos_col=['8']
nrows=50000
data=pd.read_table(tr_file_name, header=None, nrows=nrows)
data.columns = [str(c) for c in data.columns]

# Shuffle, Shuffle and Shuffle!
data = data.sample(frac=1)


num_cols = len(data.columns)
X  = data.iloc[:,1:num_cols-1]
y  = data.iloc[:,num_cols-1]

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

#Preprocess X_train
feature_def = gen_features(
     columns=[[c] for c in X_train.columns[:7]],
     classes=[MinMaxScaler]
 )

feature_def += ((pos_col, [LabelBinarizer()]),)

svc_preprocessor = DataFrameMapper(feature_def)
X_train = svc_preprocessor.fit_transform(X_train)
svc_preprocessor_fn = os.path.join('../model/tmp/svc_preprocessor.%s.pkl' % (nrows,))
joblib.dump(svc_preprocessor, open(svc_preprocessor_fn, 'wb'))
X_test = svc_preprocessor.transform(X_test)
#####

#Didn't help!!
#X_train, y_train = downsample_negatives(X_train, y_train)

for cv in [1,10,20]:
    print "Training, sample_count: %s\tcv:%s" % (nrows, cv)
    clf = svm.SVC(kernel='linear', class_weight={1:cv})
    clf.fit(X_train, y_train)  
    y_pred = clf.predict(X_test)
    measures = metrics.precision_recall_fscore_support(y_test, y_pred, average='binary')
    model_file_name = os.path.join('../model/tmp/svc_mentions_unbalanced.%s.%s.pkl' % (nrows,cv))
    joblib.dump(clf, open(model_file_name, 'wb'))
    print "measures: ", measures
    sys.stdout.flush()




print 'Model saved'