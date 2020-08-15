#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd  
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import  mutual_info_classif
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
#from textvec.vectorizers import TfrfVectorizer
#from textvec.vectorizers import TfIcfVectorizer
import os

from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics.classification import accuracy_score, f1_score
import seaborn as sns
from textvec import vectorizers
#from textvec.vectorizers import BaseBinaryFitter
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")


# In[2]:


class BaseBinaryFitter(TransformerMixin):
    """Base class for supervised methods (supports only binary classification).
    Should not be used as by itself.
    ----------
    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    smooth_df : boolean or int, default=True
        Smooth df weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    References
    ----------
    """

    def __init__(self, norm='l2', smooth_df=False, sublinear_tf=False):
        self.norm = norm
        self.smooth_df = smooth_df
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y):

        n_samples, n_features = X.shape

        pos_samples = sp.spdiags(y, 0, n_samples, n_samples)
        neg_samples = sp.spdiags(1 - y, 0, n_samples, n_samples)

        self._n_pos = np.sum(y)
        self._n_neg = np.sum(1-y)
        
        X_pos = pos_samples * X
        X_neg = neg_samples * X

        tp = np.bincount(X_pos.indices, minlength=n_features)
        fp = np.sum(y) - tp
        tn = np.bincount(X_neg.indices, minlength=n_features)
        fn = np.sum(1 - y) - tn

        self._n_samples = n_samples
        self._n_features = n_features

        self._tp = tp
        self._fp = fp
        self._fn = fn
        self._tn = tn
        self._p = np.sum(y)
        self._n = np.sum(1 - y)

        if self.smooth_df:
            self._n_samples += int(self.smooth_df)
            self._tp += int(self.smooth_df)
            self._fp += int(self.smooth_df)
            self._fn += int(self.smooth_df)
            self._tn += int(self.smooth_df)
        return self


# In[3]:


def readPolarityCross():
    data = pd.read_csv('/Users/gustavoguedes/Temp/polarity/polarity2.arff.csv',sep='##,##')  
   # df = pd.read_csv(path, encoding = "ISO-8859-1")
    #transformando as classes para binários inteiros
    data.loc[data['y']=='pos','y']=int(1)
    data.loc[data['y']=='neg','y']=int(0)
    final=pd.DataFrame({"text": data['text'], "y": data['y'].astype('int')})
    X=final['text']
    y=final['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=0
    )

    return X,y


# In[4]:


def readSarcasmCross():
    data = pd.read_csv('/Users/gustavoguedes/Temp/irony/amazon-sarcarsm-limpo3-raw.arff.csv',sep='##,##')  
    #transformando as classes para binários inteiros
    data.loc[data['y']=='ironic','y']=int(1)
    data.loc[data['y']=='regular','y']=int(0)
    final=pd.DataFrame({"text": data['text'], "y": data['y'].astype('int')})
    X=final['text']
    y=final['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=0
    )

    return X,y


# In[5]:


def readSubjectivityCross():
    data = pd.read_csv('/Users/gustavoguedes/Temp/subjectivity-raw.arff.csv',sep='##,##')  
    #transformando as classes para binários inteiros
    data.loc[data['y']=='subjetivas','y']=int(1)
    data.loc[data['y']=='objetivas','y']=int(0)
    final=pd.DataFrame({"text": data['text'], "y": data['y'].astype('int')})
    X=final['text']
    y=final['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=0
    )

    return X,y


# In[6]:


def readMovieReviewCross():
    data = pd.read_csv('/Users/gustavoguedes/Temp/movie-review-raw.arff.csv',sep='##,##')  
    #transformando as classes para binários inteiros
    data.loc[data['y']=='neg','y']=int(1)
    data.loc[data['y']=='pos','y']=int(0)
    final=pd.DataFrame({"text": data['text'], "y": data['y'].astype('int')})
    X=final['text']
    y=final['y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=0
    )

    return X,y


# In[7]:


def ensure_sparse_format(array, dtype=np.float64):
    if sp.issparse(array):
        if array.dtype != dtype:
            array = array.astype(dtype)
    else:
        array = sp.csr_matrix(array, dtype=dtype)
    return array


# In[1264]:


class TfidfcrfVectorizer1(BaseBinaryFitter):
    def transform(self, X):
        X = ensure_sparse_format(X)
        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1
        np.sqrt(X.data, X.data)     
        tp = self._tp
        fn = self._fn
        tn=self._tn
        fp=self._fp
        
        A=tp
        B=fp
        D=fn
        C=tn
        NP=A+B
        NN=D+C
        f = self._n_features

        #k = np.log2(2 + tp / fn)
      #  N=tp+fn+tn+fp
        #print(N)
        
        #k = np.log(10+(A)/C)* (D/np.log(10+(A)/C))
       # k = np.log(10+(A)/C) * (D+B) *N *(np.log(10+(C)/B+D)) 0.8444
        #k = np.log(10+(A)/C) * (D+B) *N *(np.log(10+(C)/B))0.8435
       # k = np.log(10+(A)/C)* (D/np.log(10+(A)/C)) *(N/np.log(10+(C)/B)) 0.843
       #  k = np.log(10+(A)/C)* (D/np.log(10+(A)/C)) 0.8415
#    k = np.log(10+(A)/C)* (D/np.log(10+(A)/C)) *(N/np.log(10+(B)/D)) 0.8405
       #  k = np.log(10+(A)/C)* (B/np.log(10+(A)/C))  0.8375  
#    k = np.log(10+(A)/C)*np.log(10+(pow(D,2))/B) 0.834 -> facil de explicar
     #   k = np.log2(2 + A / C)/np.log2(2 + A / C) --> 0.8195
         #k = np.log(10+A/C) -->0.818
       # k = np.log2(2 + tp / fn) --> 0.8065
        
     #   k = np.log(10+(A)/C)* (D/np.log(10+(A)/C)) *(N/np.log(10+(C)/B))
        
        #k = np.log(10+(A)/C)* (D/np.log(10+(A)/C))*(N/np.log(10+(C)/B))
        
       # k = np.log(10+(A)/C)* D  *np.log(10+(C)/B)
        #k = np.log(10+(A)/C) * (D+B) *N *(np.log(10+(C)/B+D))
        
        
        #todos os resultados foram gerados com esse abaixo:
        #k = np.log(10+(A)/C) * (D+B) *N *(np.log(10+(C)/B+D))
        
        n_pos=self._n_pos
        n_neg=self._n_neg
        N=n_pos+n_neg
        
        #concatena A com C
        vet1 = np.concatenate(([A], [C]), axis=0)
        #qual o indice da maior classe
        w = np.argmax(vet1,axis=0)
        
        #here, 0 is positive, since A is in the first line and C in the second
        Dtotal_ti=np.where(w == 0, self._n_pos, self._n_neg)
        Dtotal_ti2=np.where(w == 1, self._n_pos, self._n_neg)    
        A=A
        B=B
        C=C
        D=D
        IDF=np.log(N/(A+C)) 
        k=np.log2(2+(np.maximum(A,C)/(2+np.minimum(A,C)))* (np.sqrt(B+D)))#(np.log2(np.maximum(A, B+D)))
        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)
        return X


# In[1261]:


lst=[500,1000,2000,4000,6000,8000,10000,12000,14000]
X,y = readPolarityCross()
resultsTFMEU_svm_Pol_Sub2 = processAll(lst, TfidfcrfVectorizer1(sublinear_tf=False), 5,svm.SVC(kernel='linear', C=1))#MultinomialNB())
print('svm',resultsTFMEU_svm_Pol_Sub2)
resultsTFMEU_mnb_Pol_Sub2 = processAll(lst, TfidfcrfVectorizer1(sublinear_tf=False), 5,MultinomialNB())#MultinomialNB())
print ('mnb',resultsTFMEU_mnb_Pol_Sub2)


# In[1250]:


X,y = readMovieReviewCross()#readSubjectivityCross() #readSarcasmCross() #readPolarityCross()
print('MovieReview - SQRTTfIGMimpVectorizer')
lst=[500,1000,2000,4000,6000,8000,10000,12000,14000]
resultsTFIGMimp_mnb_Pol_Sub2 = processAll(lst, TfidfcrfVectorizer1(sublinear_tf=False), 5,MultinomialNB())#MultinomialNB())
print('1',resultsTFIGMimp_mnb_Pol_Sub2)
print('MovieReview - SQRTTfIGMimpVectorizer2')
resultsTFIGMimp_svm_Pol_Sub2 = processAll(lst, TfidfcrfVectorizer1(sublinear_tf=False), 5,svm.SVC(kernel='linear', C=1))#MultinomialNB())
print(resultsTFIGMimp_svm_Pol_Sub2)


# In[1174]:


lst=[500,1000,2000,14000]#,6000,8000,10000,12000,14000]
X,y = readSubjectivityCross() #readSarcasmCross() #readPolarityCross()

resultsTFMEU_mnb_Pol_Sub2 = processAll(lst, TfidfcrfVectorizer1(sublinear_tf=False), 5,MultinomialNB())#MultinomialNB())
print ('mnb',resultsTFMEU_mnb_Pol_Sub2)
resultsTFMEU_svm_Pol_Sub2 = processAll(lst, TfidfcrfVectorizer1(sublinear_tf=False), 5,svm.SVC(kernel='linear', C=1))#MultinomialNB())
print('svm',resultsTFMEU_svm_Pol_Sub2)


# In[1253]:


lst=[500,1000,2000,4000,14000]#,6000,8000,10000,12000,14000]
X,y = readSarcasmCross()#readSubjectivityCross() #readSarcasmCross() #readPolarityCross()

resultsTFMEU_svm_Pol_Sub2 = processAll(lst, TfidfcrfVectorizer1(sublinear_tf=False), 5,svm.SVC(kernel='linear', C=1))#MultinomialNB())
print('svm',resultsTFMEU_svm_Pol_Sub2)
resultsTFMEU_mnb_Pol_Sub2 = processAll(lst, TfidfcrfVectorizer1(sublinear_tf=False), 5,MultinomialNB())#MultinomialNB())
print ('mnb',resultsTFMEU_mnb_Pol_Sub2)


# In[10]:


class TfIGMVectorizer(BaseBinaryFitter):
    def transform(self, X):
        if self.sublinear_tf:
            X = ensure_sparse_format(X)
            np.log(X.data, X.data)
            X.data += 1
        tp = self._tp
        fn = self._fn
        tn=self._tn
        fp=self._fp 
        A=tp
        B=fp
        D=fn
        C=tn
        NP=A+B
        NN=C+D
        f = self._n_features
        N=tp+fn+tn+fp
        IGM=np.maximum(A,C)/((np.maximum(A,C)*1)+(np.minimum(A,C)*2))
        k=(1 + (7*IGM));
        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)

        return X


# In[ ]:





# In[11]:


class SQRTTfIGMVectorizer(BaseBinaryFitter):
    def transform(self, X):
        X = ensure_sparse_format(X)
        if self.sublinear_tf:
         #   X = ensure_sparse_format(X)
            np.log(X.data, X.data)
            X.data += 1
        np.sqrt(X.data, X.data)   
        tp = self._tp
        fn = self._fn
        tn=self._tn
        fp=self._fp 
        A=tp
        B=fp
        D=fn
        C=tn
        NP=A+B
        NN=C+D
        f = self._n_features
        N=tp+fn+tn+fp
        IGM=np.maximum(A,C)/(((np.maximum(A,C)*1)+(np.minimum(A,C)*2)))
        k=(1 + (7*IGM));
        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)

        return X


# In[941]:


class TfIGMimpVectorizer(BaseBinaryFitter):
    def transform(self, X):
        if self.sublinear_tf:
            X = ensure_sparse_format(X)
            np.log(X.data, X.data)
            X.data += 1 
        tp = self._tp
        fn = self._fn
        tn=self._tn
        fp=self._fp 
        A=tp
        B=fp
        D=fn
        C=tn
        NP=A+B
        NN=C+D
        f = self._n_features
        N=tp+fn+tn+fp
        
        #concatena A com C
        vet1 = np.concatenate(([A], [C]), axis=0)
        #qual o indice da maior classe
        w = np.argmax(vet1,axis=0)
        
        #here, 0 is positive, since A is in the first line and C in the second
        Dtotal_ti=np.where(w == 0, self._n_pos, self._n_neg)
        IGM = np.maximum(A, C) / (((np.maximum(A, C) * 1) + (np.minimum(A, C) * 2) + np.log10(Dtotal_ti / np.maximum(A, C))))
        k=(1 + (7*IGM));
        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)

        return X


# In[934]:


class SQRTTfIGMimpVectorizer(BaseBinaryFitter):
    def transform(self, X):
        X = ensure_sparse_format(X)
        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1 
        np.sqrt(X.data, X.data)      
        tp = self._tp
        fn = self._fn
        tn=self._tn
        fp=self._fp 
        A=tp
        B=fp
        D=fn
        C=tn
        NP=A+B
        NN=C+D
        f = self._n_features
        N=tp+fn+tn+fp
        
        #concatena A com C
        vet1 = np.concatenate(([A], [C]), axis=0)
        #qual o indice da maior classe
        w = np.argmax(vet1,axis=0)
        
        #here, 0 is positive, since A is in the first line and C in the second
        Dtotal_ti=np.where(w == 0, self._n_pos, self._n_neg)
        IGM = np.maximum(A, C) / (((np.maximum(A, C) * 1) + (np.minimum(A, C) * 2) + np.log10(Dtotal_ti / np.maximum(A, C))))
        k=(1 + (7*IGM));
        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)

        return X


# In[17]:


class TfDELTAidf(BaseBinaryFitter):
    def transform(self, X):
        if self.sublinear_tf:
            X = ensure_sparse_format(X)
            np.log(X.data, X.data)
            X.data += 1
            
        tp = self._tp
        fn = self._fn
        tn=self._tn
        fp=self._fp
        
        A=tp
        B=fp
        D=fn
        C=tn
        NP=A+B
        NN=C+D
        f = self._n_features

        #k = np.log2(2 + tp / fn)
        N=tp+fn+tn+fp
        
        k=np.log2(2+((NP+C+0.5)/(A*NN+0.5)))
       # k = np.log(10+(A)/C) * (D+B) *N *(np.log(10+(C)/B+D))
        
#        D+B termo nao ocorre

        #k=np.log(N/(A)+1)*np.log((A+1)/(B+1)+1)
      #  k=np.log((A+1)/(C+1)+1)
        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)

        return X


# In[18]:


class TfrfVectorizer(BaseBinaryFitter, BaseEstimator):
    """Supervised method (supports ONLY binary classification)
    transform a count matrix to a normalized Tfrf representation
    Tf means term-frequency while RF means relevance frequency.
    Parameters
    ----------
    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    smooth_df : boolean or int, default=True
        Smooth df weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    References
    ----------
    .. [M. Lan, C. L. Tan, J. Su, and Y. Lu] `Supervised and traditional
                term weighting methods for automatic text categorization`
    """



    def transform(self, X):
        X = ensure_sparse_format(X)
        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        tp = self._tp
        fn = self._fn
        tn = self._tn
        f = self._n_features

        k = np.log2(2 + (tp / np.maximum(1,tn)))

        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)

        return X


# In[19]:


class TfIcfVectorizer(BaseBinaryFitter,TransformerMixin, BaseEstimator):
    """Supervised method (supports multiclass) to transform
    a count matrix to a normalized Tficf representation
    Tf means term-frequency while ICF means inverse category frequency.
    Parameters
    ----------
    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    References
    ----------
    .. [0] `https://arxiv.org/pdf/1012.2609.pdf`
    """



    def __init__(self, norm=None, sublinear_tf=False, smooth_df=False):
        self.norm = norm
        self.sublinear_tf = sublinear_tf
        self.smooth_df = smooth_df

    def fit(self, X, y):
        n_samples, n_features = X.shape
        pos_samples = sp.spdiags(y, 0, n_samples, n_samples)
        neg_samples = sp.spdiags(1 - y, 0, n_samples, n_samples)

        X_pos = pos_samples * X
        X_neg = neg_samples * X

        tp = np.bincount(X_pos.indices, minlength=n_features)
        fp = np.sum(y) - tp
        tn = np.bincount(X_neg.indices, minlength=n_features)
        fn = np.sum(1 - y) - tn
        if self.smooth_df:
            self._n_samples += int(self.smooth_df)
            self._tp += int(self.smooth_df)
            self._fp += int(self.smooth_df)
            self._fn += int(self.smooth_df)
            self._tn += int(self.smooth_df)

        samples = []
        self.number_of_classes = len(np.unique(y))
        for val in range(self.number_of_classes):
            class_mask = sp.spdiags(y == val, 0, n_samples, n_samples)
            samples.append(np.bincount(
                (class_mask * X).indices, minlength=n_features))
        samples = np.array(samples)
        self.corpus_occurence = np.sum(samples != 0, axis=0)
        self.k = (1+np.log(n_samples/(tp+tn))) * (1 + np.log(self.number_of_classes / self.corpus_occurence))
        self._n_features = n_features
        return self

    def transform(self, X, min_freq=1):
        if self.sublinear_tf:
            X = ensure_sparse_format(X)
            np.log(X.data, X.data)
            X.data += 1
        f = self._n_features
        X = X * sp.spdiags(self.k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)
        return X


# In[14]:


def process (num_attributes, vectorizer, cross_val_num, classifier):
  #  clf = svm.SVC(kernel='linear', C=1)
    clf = classifier
    pipe = Pipeline([('vect', CountVectorizer()),
                     ('vetorizer', vectorizer),
                     ('reduce_dim', SelectKBest(chi2, k=num_attributes)),
                     ('clf', clf)])
    scores = cross_val_score(pipe,X.values,y.values,cv=cross_val_num, n_jobs=8)
#scores = cross_val_score(clf, X.values, y.values, cv=5, scoring='f1_macro')
    scores
    soma=0
    for x in scores:   
        soma=soma+x
    return soma/cross_val_num


# In[15]:


def processAll (array_val, vectorizer, cross_val_num, classifier):
    lst = []
    
    for x in array_val:  
        lst.append(process(x, vectorizer, cross_val_num, classifier))

    return lst


# In[52]:


lst=[500,1000,2000,4000,6000,8000,10000,12000,14000]


# In[ ]:





# In[1162]:


lst=[500,1000,2000,4000,6000,8000,10000,12000,14000]
X,y = readMovieReviewCross()#readSubjectivityCross() #readSarcasmCross() #readPolarityCross()
resultsTFIGM_svm_MR_MR = processAll(lst, TfIGMVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsSQRTTFIGM_svm_MR_MR = processAll(lst, SQRTTfIGMVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsTFIGMimp_svm_MR_MR = processAll(lst, TfIGMimpVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsSQRTTFIGMimp_svm_MR_MR = processAll(lst, SQRTTfIGMimpVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))

resultsICF_svm_MR_MR = processAll(lst, TfIcfVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsDelta_svm_MR_MR = processAll(lst, TfDELTAidf(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsTFMEU_svm_MR_MR = processAll(lst, TfidfcrfVectorizer1(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsTFIDF_svm_MR_MR = processAll(lst, TfidfTransformer(sublinear_tf=False, smooth_idf=False), 5, svm.SVC(kernel='linear', C=1))
resultsTF_svm_MR_MR = processAll(lst, TfidfTransformer(use_idf=False, sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsRF_svm_MR = processAll(lst, TfrfVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))

resultsTFIGM_mnb_MR_MR = processAll(lst, TfIGMVectorizer(sublinear_tf=False), 5,MultinomialNB())
resultsSQRTTFIGM_mnb_MR_MR = processAll(lst, SQRTTfIGMVectorizer(sublinear_tf=False), 5,MultinomialNB())
resultsTFIGMimp_mnb_MR_MR = processAll(lst, TfIGMimpVectorizer(sublinear_tf=False), 5,MultinomialNB())
resultsSQRTTFIGMimp_mnb_MR_MR = processAll(lst, SQRTTfIGMimpVectorizer(sublinear_tf=False), 5,MultinomialNB())

resultsICF_mnb_MR_MR = processAll(lst, TfIcfVectorizer(sublinear_tf=False), 5,MultinomialNB())
resultsDelta_mnb_MR_MR = processAll(lst, TfDELTAidf(sublinear_tf=False), 5,MultinomialNB())
resultsTFMEU_mnb_MR_MR = processAll(lst, TfidfcrfVectorizer1(sublinear_tf=False), 5,MultinomialNB())
resultsTFIDF_mnb_MR_MR = processAll(lst, TfidfTransformer(sublinear_tf=False, smooth_idf=False), 5,MultinomialNB())
resultsTF_mnb_MR_MR = processAll(lst, TfidfTransformer(use_idf=False, sublinear_tf=False), 5,MultinomialNB())
resultsRF_mnb_MR_MR = processAll(lst, TfrfVectorizer(sublinear_tf=False), 5,MultinomialNB())


# In[1185]:


np.set_printoptions(precision=2)
k500=np.array([resultsTF_svm_MR_MR[0], resultsTFIDF_svm_MR_MR[0], resultsDelta_svm_MR_MR[0], resultsICF_svm_MR_MR[0],resultsRF_svm_MR[0],resultsTFIGM_svm_MR_MR[0], resultsSQRTTFIGM_svm_MR_MR[0],resultsTFIGMimp_svm_MR_MR[0],resultsSQRTTFIGMimp_svm_MR_MR[0],resultsTFMEU_svm_MR_MR[0] ])
k500=k500*100
print('500',k500)

np.set_printoptions(precision=2)
k1000=np.array([resultsTF_svm_MR_MR[1], resultsTFIDF_svm_MR_MR[1], resultsDelta_svm_MR_MR[1], resultsICF_svm_MR_MR[1],resultsRF_svm_MR[1],resultsTFIGM_svm_MR_MR[1], resultsSQRTTFIGM_svm_MR_MR[1],resultsTFIGMimp_svm_MR_MR[1],resultsSQRTTFIGMimp_svm_MR_MR[1],resultsTFMEU_svm_MR_MR[1] ])
k1000=k1000*100
print('1000', k1000)

np.set_printoptions(precision=2)
k2000=np.array([resultsTF_svm_MR_MR[2], resultsTFIDF_svm_MR_MR[2], resultsDelta_svm_MR_MR[2], resultsICF_svm_MR_MR[2],resultsRF_svm_MR[2],resultsTFIGM_svm_MR_MR[2], resultsSQRTTFIGM_svm_MR_MR[2],resultsTFIGMimp_svm_MR_MR[2],resultsSQRTTFIGMimp_svm_MR_MR[2],resultsTFMEU_svm_MR_MR[2] ])
k2000=k2000*100
print('2000', k2000)

np.set_printoptions(precision=2)
k4000=np.array([resultsTF_svm_MR_MR[3], resultsTFIDF_svm_MR_MR[3], resultsDelta_svm_MR_MR[3], resultsICF_svm_MR_MR[3],resultsRF_svm_MR[3],resultsTFIGM_svm_MR_MR[3], resultsSQRTTFIGM_svm_MR_MR[3],resultsTFIGMimp_svm_MR_MR[3],resultsSQRTTFIGMimp_svm_MR_MR[3],resultsTFMEU_svm_MR_MR[3] ])
k4000=k4000*100
print('4000', k4000)

np.set_printoptions(precision=2)
k6000=np.array([resultsTF_svm_MR_MR[4], resultsTFIDF_svm_MR_MR[4], resultsDelta_svm_MR_MR[4], resultsICF_svm_MR_MR[4],resultsRF_svm_MR[4],resultsTFIGM_svm_MR_MR[4], resultsSQRTTFIGM_svm_MR_MR[4],resultsTFIGMimp_svm_MR_MR[4],resultsSQRTTFIGMimp_svm_MR_MR[4],resultsTFMEU_svm_MR_MR[4] ])
k6000=k6000*100
print('6000', k6000)

np.set_printoptions(precision=2)
k8000=np.array([resultsTF_svm_MR_MR[5], resultsTFIDF_svm_MR_MR[5], resultsDelta_svm_MR_MR[5], resultsICF_svm_MR_MR[5],resultsRF_svm_MR[5],resultsTFIGM_svm_MR_MR[5], resultsSQRTTFIGM_svm_MR_MR[5],resultsTFIGMimp_svm_MR_MR[5],resultsSQRTTFIGMimp_svm_MR_MR[5],resultsTFMEU_svm_MR_MR[5] ])
k8000 = k8000*100
print('8000', k8000)

np.set_printoptions(precision=2)
k10000=np.array([resultsTF_svm_MR_MR[6], resultsTFIDF_svm_MR_MR[6], resultsDelta_svm_MR_MR[6], resultsICF_svm_MR_MR[6],resultsRF_svm_MR[6],resultsTFIGM_svm_MR_MR[6], resultsSQRTTFIGM_svm_MR_MR[6],resultsTFIGMimp_svm_MR_MR[6],resultsSQRTTFIGMimp_svm_MR_MR[6],resultsTFMEU_svm_MR_MR[6] ])
k10000 = k10000*100
print('10000', k10000)

np.set_printoptions(precision=2)
k12000=np.array([resultsTF_svm_MR_MR[7], resultsTFIDF_svm_MR_MR[7], resultsDelta_svm_MR_MR[7], resultsICF_svm_MR_MR[7],resultsRF_svm_MR[7],resultsTFIGM_svm_MR_MR[7], resultsSQRTTFIGM_svm_MR_MR[7],resultsTFIGMimp_svm_MR_MR[7],resultsSQRTTFIGMimp_svm_MR_MR[7],resultsTFMEU_svm_MR_MR[7] ])
k12000 = k12000*100
print('12000', k12000)

np.set_printoptions(precision=2)
k14000=np.array([resultsTF_svm_MR_MR[8], resultsTFIDF_svm_MR_MR[8], resultsDelta_svm_MR_MR[8], resultsICF_svm_MR_MR[8],resultsRF_svm_MR[8],resultsTFIGM_svm_MR_MR[8], resultsSQRTTFIGM_svm_MR_MR[8],resultsTFIGMimp_svm_MR_MR[8],resultsSQRTTFIGMimp_svm_MR_MR[8],resultsTFMEU_svm_MR_MR[8] ])
k14000 = k14000*100
print('14000', k14000)


# In[1214]:


#grafico Polarity
np.set_printoptions(precision=2)
cont =0
for i in resultsTF_mnb_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1

print('---')    
cont =0
for i in resultsTFIDF_mnb_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1

print('---')    
cont =0
for i in resultsDelta_mnb_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1    
    
print('---')    
cont =0
for i in resultsICF_mnb_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       
    
print('---')    
cont =0
for i in resultsRF_mnb_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       
    
print('---')    
cont =0
for i in resultsTFIGM_mnb_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1 

print('---')    
cont =0
for i in resultsSQRTTFIGM_mnb_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1 

print('---')    
cont =0
for i in resultsTFIGMimp_mnb_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1     

print('---')    
cont =0
for i in resultsSQRTTFIGMimp_mnb_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       

print('---')    
cont =0
for i in resultsTFMEU_mnb_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1      
 


# In[1184]:


np.set_printoptions(precision=2,suppress=False)
k500=np.array([resultsTF_mnb_MR_MR[0], resultsTFIDF_mnb_MR_MR[0], resultsDelta_mnb_MR_MR[0], resultsICF_mnb_MR_MR[0],resultsRF_mnb_MR_MR[0],resultsTFIGM_mnb_MR_MR[0], resultsSQRTTFIGM_mnb_MR_MR[0],resultsTFIGMimp_mnb_MR_MR[0],resultsSQRTTFIGMimp_mnb_MR_MR[0],resultsTFMEU_mnb_MR_MR[0] ])
k500=k500*100
print('500', k500)

np.set_printoptions(precision=2)
k1000=np.array([resultsTF_mnb_MR_MR[1], resultsTFIDF_mnb_MR_MR[1], resultsDelta_mnb_MR_MR[1], resultsICF_mnb_MR_MR[1],resultsRF_mnb_MR_MR[1],resultsTFIGM_mnb_MR_MR[1], resultsSQRTTFIGM_mnb_MR_MR[1],resultsTFIGMimp_mnb_MR_MR[1],resultsSQRTTFIGMimp_mnb_MR_MR[1],resultsTFMEU_mnb_MR_MR[1] ])
k1000=k1000*100
print('1000', k1000)

np.set_printoptions(precision=2)
k2000=np.array([resultsTF_mnb_MR_MR[2], resultsTFIDF_mnb_MR_MR[2], resultsDelta_mnb_MR_MR[2], resultsICF_mnb_MR_MR[2],resultsRF_mnb_MR_MR[2],resultsTFIGM_mnb_MR_MR[2], resultsSQRTTFIGM_mnb_MR_MR[2],resultsTFIGMimp_mnb_MR_MR[2],resultsSQRTTFIGMimp_mnb_MR_MR[2],resultsTFMEU_mnb_MR_MR[2] ])
k2000=k2000*100
print('2000', k2000)

np.set_printoptions(precision=2)
k4000=np.array([resultsTF_mnb_MR_MR[3], resultsTFIDF_mnb_MR_MR[3], resultsDelta_mnb_MR_MR[3], resultsICF_mnb_MR_MR[3],resultsRF_mnb_MR_MR[3],resultsTFIGM_mnb_MR_MR[3], resultsSQRTTFIGM_mnb_MR_MR[3],resultsTFIGMimp_mnb_MR_MR[3],resultsSQRTTFIGMimp_mnb_MR_MR[3],resultsTFMEU_mnb_MR_MR[3] ])
k4000=k4000*100
print('4000', k4000)

np.set_printoptions(precision=2)
k6000=np.array([resultsTF_mnb_MR_MR[4], resultsTFIDF_mnb_MR_MR[4], resultsDelta_mnb_MR_MR[4], resultsICF_mnb_MR_MR[4],resultsRF_mnb_MR_MR[4],resultsTFIGM_mnb_MR_MR[4], resultsSQRTTFIGM_mnb_MR_MR[4],resultsTFIGMimp_mnb_MR_MR[4],resultsSQRTTFIGMimp_mnb_MR_MR[4],resultsTFMEU_mnb_MR_MR[4] ])
k6000=k6000*100
print('6000', k6000)

np.set_printoptions(precision=2)
k8000=np.array([resultsTF_mnb_MR_MR[5], resultsTFIDF_mnb_MR_MR[5], resultsDelta_mnb_MR_MR[5], resultsICF_mnb_MR_MR[5],resultsRF_mnb_MR_MR[5],resultsTFIGM_mnb_MR_MR[5], resultsSQRTTFIGM_mnb_MR_MR[5],resultsTFIGMimp_mnb_MR_MR[5],resultsSQRTTFIGMimp_mnb_MR_MR[5],resultsTFMEU_mnb_MR_MR[5] ])
k8000 = k8000*100
print('8000', k8000)

np.set_printoptions(precision=2)
k10000=np.array([resultsTF_mnb_MR_MR[6], resultsTFIDF_mnb_MR_MR[6], resultsDelta_mnb_MR_MR[6], resultsICF_mnb_MR_MR[6],resultsRF_mnb_MR_MR[6],resultsTFIGM_mnb_MR_MR[6], resultsSQRTTFIGM_mnb_MR_MR[6],resultsTFIGMimp_mnb_MR_MR[6],resultsSQRTTFIGMimp_mnb_MR_MR[6],resultsTFMEU_mnb_MR_MR[6] ])
k10000 = k10000*100
print('10000', k10000)

np.set_printoptions(precision=2)
k12000=np.array([resultsTF_mnb_MR_MR[7], resultsTFIDF_mnb_MR_MR[7], resultsDelta_mnb_MR_MR[7], resultsICF_mnb_MR_MR[7],resultsRF_mnb_MR_MR[7],resultsTFIGM_mnb_MR_MR[7], resultsSQRTTFIGM_mnb_MR_MR[7],resultsTFIGMimp_mnb_MR_MR[7],resultsSQRTTFIGMimp_mnb_MR_MR[7],resultsTFMEU_mnb_MR_MR[7] ])
k12000 = k12000*100
print('12000', k12000)

np.set_printoptions(precision=2)
k14000=np.array([resultsTF_mnb_MR_MR[8], resultsTFIDF_mnb_MR_MR[8], resultsDelta_mnb_MR_MR[8], resultsICF_mnb_MR_MR[8],resultsRF_mnb_MR_MR[8],resultsTFIGM_mnb_MR_MR[8], resultsSQRTTFIGM_mnb_MR_MR[8],resultsTFIGMimp_mnb_MR_MR[8],resultsSQRTTFIGMimp_mnb_MR_MR[8],resultsTFMEU_mnb_MR_MR[8] ])
k14000 = k14000*100
print('14000', k14000)


# In[1215]:


#grafico Polarity
np.set_printoptions(precision=2)
cont =0
for i in resultsTF_svm_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1

print('---')    
cont =0
for i in resultsTFIDF_svm_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1

print('---')    
cont =0
for i in resultsDelta_svm_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1    
    
print('---')    
cont =0
for i in resultsICF_svm_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       
    
print('---')    
cont =0
for i in resultsRF_svm_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       
    
print('---')    
cont =0
for i in resultsTFIGM_svm_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1 

print('---')    
cont =0
for i in resultsSQRTTFIGM_svm_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1 

print('---')    
cont =0
for i in resultsTFIGMimp_svm_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1     

print('---')    
cont =0
for i in resultsSQRTTFIGMimp_svm_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       

print('---')    
cont =0
for i in resultsTFMEU_svm_MR_MR:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1      
 


# In[1159]:


X,y = readSubjectivityCross() #readSarcasmCross() #readPolarityCross()
resultsTFIGM_svm_MR_sub = processAll(lst, TfIGMVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsSQRTTFIGM_svm_MR_sub = processAll(lst, SQRTTfIGMVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsTFIGMimp_svm_MR_sub = processAll(lst, TfIGMimpVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsSQRTTFIGMimp_svm_MR_sub = processAll(lst, SQRTTfIGMimpVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))

resultsICF_svm_MR_sub = processAll(lst, TfIcfVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsDelta_svm_MR_sub = processAll(lst, TfDELTAidf(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsTFMEU_svm_MR_sub = processAll(lst, TfidfcrfVectorizer1(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsTFIDF_svm_MR_sub = processAll(lst, TfidfTransformer(sublinear_tf=False, smooth_idf=False), 5, svm.SVC(kernel='linear', C=1))
resultsTF_svm_MR_sub = processAll(lst, TfidfTransformer(use_idf=False, sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsRF_svm_sub = processAll(lst, TfrfVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))

resultsTFIGM_mnb_MR_sub = processAll(lst, TfIGMVectorizer(sublinear_tf=False), 5,MultinomialNB())
resultsSQRTTFIGM_mnb_MR_sub = processAll(lst, SQRTTfIGMVectorizer(sublinear_tf=False), 5,MultinomialNB())
resultsTFIGMimp_mnb_MR_sub = processAll(lst, TfIGMimpVectorizer(sublinear_tf=False), 5,MultinomialNB())
resultsSQRTTFIGMimp_mnb_MR_sub = processAll(lst, SQRTTfIGMimpVectorizer(sublinear_tf=False), 5,MultinomialNB())

resultsICF_mnb_MR_sub = processAll(lst, TfIcfVectorizer(sublinear_tf=False), 5,MultinomialNB())
resultsDelta_mnb_MR_sub = processAll(lst, TfDELTAidf(sublinear_tf=False), 5,MultinomialNB())
resultsTFMEU_mnb_MR_sub = processAll(lst, TfidfcrfVectorizer1(sublinear_tf=False), 5,MultinomialNB())
resultsTFIDF_mnb_MR_sub = processAll(lst, TfidfTransformer(sublinear_tf=False, smooth_idf=False), 5,MultinomialNB())
resultsTF_mnb_MR_sub = processAll(lst, TfidfTransformer(use_idf=False, sublinear_tf=False), 5,MultinomialNB())
resultsRF_mnb_MR_sub = processAll(lst, TfrfVectorizer(sublinear_tf=False), 5,MultinomialNB())


# In[ ]:





# In[1182]:


np.set_printoptions(precision=2)
k500=np.array([resultsTF_svm_MR_sub[0], resultsTFIDF_svm_MR_sub[0], resultsDelta_svm_MR_sub[0], resultsICF_svm_MR_sub[0],resultsRF_svm_MR[0],resultsTFIGM_svm_MR_sub[0], resultsSQRTTFIGM_svm_MR_sub[0],resultsTFIGMimp_svm_MR_sub[0],resultsSQRTTFIGMimp_svm_MR_sub[0],resultsTFMEU_svm_MR_sub[0] ])
k500=k500*100
print('500', k500)

np.set_printoptions(precision=2)
k1000=np.array([resultsTF_svm_MR_sub[1], resultsTFIDF_svm_MR_sub[1], resultsDelta_svm_MR_sub[1], resultsICF_svm_MR_sub[1],resultsRF_svm_MR[1],resultsTFIGM_svm_MR_sub[1], resultsSQRTTFIGM_svm_MR_sub[1],resultsTFIGMimp_svm_MR_sub[1],resultsSQRTTFIGMimp_svm_MR_sub[1],resultsTFMEU_svm_MR_sub[1] ])
k1000=k1000*100
print('1000', k1000)

np.set_printoptions(precision=2)
k2000=np.array([resultsTF_svm_MR_sub[2], resultsTFIDF_svm_MR_sub[2], resultsDelta_svm_MR_sub[2], resultsICF_svm_MR_sub[2],resultsRF_svm_MR[2],resultsTFIGM_svm_MR_sub[2], resultsSQRTTFIGM_svm_MR_sub[2],resultsTFIGMimp_svm_MR_sub[2],resultsSQRTTFIGMimp_svm_MR_sub[2],resultsTFMEU_svm_MR_sub[2] ])
k2000=k2000*100
print('2000', k2000)

np.set_printoptions(precision=2)
k4000=np.array([resultsTF_svm_MR_sub[3], resultsTFIDF_svm_MR_sub[3], resultsDelta_svm_MR_sub[3], resultsICF_svm_MR_sub[3],resultsRF_svm_MR[3],resultsTFIGM_svm_MR_sub[3], resultsSQRTTFIGM_svm_MR_sub[3],resultsTFIGMimp_svm_MR_sub[3],resultsSQRTTFIGMimp_svm_MR_sub[3],resultsTFMEU_svm_MR_sub[3] ])
k4000=k4000*100
print('4000', k4000)

np.set_printoptions(precision=2)
k6000=np.array([resultsTF_svm_MR_sub[4], resultsTFIDF_svm_MR_sub[4], resultsDelta_svm_MR_sub[4], resultsICF_svm_MR_sub[4],resultsRF_svm_MR[4],resultsTFIGM_svm_MR_sub[4], resultsSQRTTFIGM_svm_MR_sub[4],resultsTFIGMimp_svm_MR_sub[4],resultsSQRTTFIGMimp_svm_MR_sub[4],resultsTFMEU_svm_MR_sub[4] ])
k6000=k6000*100
print('6000', k6000)

np.set_printoptions(precision=2)
k8000=np.array([resultsTF_svm_MR_sub[5], resultsTFIDF_svm_MR_sub[5], resultsDelta_svm_MR_sub[5], resultsICF_svm_MR_sub[5],resultsRF_svm_MR[5],resultsTFIGM_svm_MR_sub[5], resultsSQRTTFIGM_svm_MR_sub[5],resultsTFIGMimp_svm_MR_sub[5],resultsSQRTTFIGMimp_svm_MR_sub[5],resultsTFMEU_svm_MR_sub[5] ])
k8000 = k8000*100
print('8000', k8000)

np.set_printoptions(precision=2)
k10000=np.array([resultsTF_svm_MR_sub[6], resultsTFIDF_svm_MR_sub[6], resultsDelta_svm_MR_sub[6], resultsICF_svm_MR_sub[6],resultsRF_svm_MR[6],resultsTFIGM_svm_MR_sub[6], resultsSQRTTFIGM_svm_MR_sub[6],resultsTFIGMimp_svm_MR_sub[6],resultsSQRTTFIGMimp_svm_MR_sub[6],resultsTFMEU_svm_MR_sub[6] ])
k10000 = k10000*100
print('10000', k10000)

np.set_printoptions(precision=2)
k12000=np.array([resultsTF_svm_MR_sub[7], resultsTFIDF_svm_MR_sub[7], resultsDelta_svm_MR_sub[7], resultsICF_svm_MR_sub[7],resultsRF_svm_MR[7],resultsTFIGM_svm_MR_sub[7], resultsSQRTTFIGM_svm_MR_sub[7],resultsTFIGMimp_svm_MR_sub[7],resultsSQRTTFIGMimp_svm_MR_sub[7],resultsTFMEU_svm_MR_sub[7] ])
k12000 = k12000*100
print('12000', k12000)

np.set_printoptions(precision=2)
k14000=np.array([resultsTF_svm_MR_sub[8], resultsTFIDF_svm_MR_sub[8], resultsDelta_svm_MR_sub[8], resultsICF_svm_MR_sub[8],resultsRF_svm_MR[8],resultsTFIGM_svm_MR_sub[8], resultsSQRTTFIGM_svm_MR_sub[8],resultsTFIGMimp_svm_MR_sub[8],resultsSQRTTFIGMimp_svm_MR_sub[8],resultsTFMEU_svm_MR_sub[8] ])
k14000 = k14000*100
print('14000', k14000)


# In[1209]:


#grafico Polarity
np.set_printoptions(precision=2)
cont =0
for i in resultsTF_svm_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1

print('---')    
cont =0
for i in resultsTFIDF_svm_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1

print('---')    
cont =0
for i in resultsDelta_svm_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1    
    
print('---')    
cont =0
for i in resultsICF_svm_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       
    
print('---')    
cont =0
for i in resultsRF_svm_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       
    
print('---')    
cont =0
for i in resultsTFIGM_svm_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1 

print('---')    
cont =0
for i in resultsSQRTTFIGM_svm_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1 

print('---')    
cont =0
for i in resultsTFIGMimp_svm_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1     

print('---')    
cont =0
for i in resultsSQRTTFIGMimp_svm_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       

print('---')    
cont =0
for i in resultsTFMEU_svm_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1      
 


# In[1183]:


np.set_printoptions(precision=2)
k500=np.array([resultsTF_mnb_MR_sub[0], resultsTFIDF_mnb_MR_sub[0], resultsDelta_mnb_MR_sub[0], resultsICF_mnb_MR_sub[0],resultsRF_mnb_MR_sub[0],resultsTFIGM_mnb_MR_sub[0], resultsSQRTTFIGM_mnb_MR_sub[0],resultsTFIGMimp_mnb_MR_sub[0],resultsSQRTTFIGMimp_mnb_MR_sub[0],resultsTFMEU_mnb_MR_sub[0] ])
k500=k500*100
print('500', k500)

np.set_printoptions(precision=2)
k1000=np.array([resultsTF_mnb_MR_sub[1], resultsTFIDF_mnb_MR_sub[1], resultsDelta_mnb_MR_sub[1], resultsICF_mnb_MR_sub[1],resultsRF_mnb_MR_sub[1],resultsTFIGM_mnb_MR_sub[1], resultsSQRTTFIGM_mnb_MR_sub[1],resultsTFIGMimp_mnb_MR_sub[1],resultsSQRTTFIGMimp_mnb_MR_sub[1],resultsTFMEU_mnb_MR_sub[1] ])
k1000=k1000*100
print('1000', k1000)

np.set_printoptions(precision=2)
k2000=np.array([resultsTF_mnb_MR_sub[2], resultsTFIDF_mnb_MR_sub[2], resultsDelta_mnb_MR_sub[2], resultsICF_mnb_MR_sub[2],resultsRF_mnb_MR_sub[2],resultsTFIGM_mnb_MR_sub[2], resultsSQRTTFIGM_mnb_MR_sub[2],resultsTFIGMimp_mnb_MR_sub[2],resultsSQRTTFIGMimp_mnb_MR_sub[2],resultsTFMEU_mnb_MR_sub[2] ])
k2000=k2000*100
print('2000', k2000)

np.set_printoptions(precision=2)
k4000=np.array([resultsTF_mnb_MR_sub[3], resultsTFIDF_mnb_MR_sub[3], resultsDelta_mnb_MR_sub[3], resultsICF_mnb_MR_sub[3],resultsRF_mnb_MR_sub[3],resultsTFIGM_mnb_MR_sub[3], resultsSQRTTFIGM_mnb_MR_sub[3],resultsTFIGMimp_mnb_MR_sub[3],resultsSQRTTFIGMimp_mnb_MR_sub[3],resultsTFMEU_mnb_MR_sub[3] ])
k4000=k4000*100
print('4000', k4000)

np.set_printoptions(precision=2)
k6000=np.array([resultsTF_mnb_MR_sub[4], resultsTFIDF_mnb_MR_sub[4], resultsDelta_mnb_MR_sub[4], resultsICF_mnb_MR_sub[4],resultsRF_mnb_MR_sub[4],resultsTFIGM_mnb_MR_sub[4], resultsSQRTTFIGM_mnb_MR_sub[4],resultsTFIGMimp_mnb_MR_sub[4],resultsSQRTTFIGMimp_mnb_MR_sub[4],resultsTFMEU_mnb_MR_sub[4] ])
k6000=k6000*100
print('6000', k6000)

np.set_printoptions(precision=2)
k8000=np.array([resultsTF_mnb_MR_sub[5], resultsTFIDF_mnb_MR_sub[5], resultsDelta_mnb_MR_sub[5], resultsICF_mnb_MR_sub[5],resultsRF_mnb_MR_sub[5],resultsTFIGM_mnb_MR_sub[5], resultsSQRTTFIGM_mnb_MR_sub[5],resultsTFIGMimp_mnb_MR_sub[5],resultsSQRTTFIGMimp_mnb_MR_sub[5],resultsTFMEU_mnb_MR_sub[5] ])
k8000 = k8000*100
print('8000', k8000)

np.set_printoptions(precision=2)
k10000=np.array([resultsTF_mnb_MR_sub[6], resultsTFIDF_mnb_MR_sub[6], resultsDelta_mnb_MR_sub[6], resultsICF_mnb_MR_sub[6],resultsRF_mnb_MR_sub[6],resultsTFIGM_mnb_MR_sub[6], resultsSQRTTFIGM_mnb_MR_sub[6],resultsTFIGMimp_mnb_MR_sub[6],resultsSQRTTFIGMimp_mnb_MR_sub[6],resultsTFMEU_mnb_MR_sub[6] ])
k10000 = k10000*100
print('10000', k10000)

np.set_printoptions(precision=2)
k12000=np.array([resultsTF_mnb_MR_sub[7], resultsTFIDF_mnb_MR_sub[7], resultsDelta_mnb_MR_sub[7], resultsICF_mnb_MR_sub[7],resultsRF_mnb_MR_sub[7],resultsTFIGM_mnb_MR_sub[7], resultsSQRTTFIGM_mnb_MR_sub[7],resultsTFIGMimp_mnb_MR_sub[7],resultsSQRTTFIGMimp_mnb_MR_sub[7],resultsTFMEU_mnb_MR_sub[7] ])
k12000 = k12000*100
print('12000', k12000)

np.set_printoptions(precision=2)
k14000=np.array([resultsTF_mnb_MR_sub[8], resultsTFIDF_mnb_MR_sub[8], resultsDelta_mnb_MR_sub[8], resultsICF_mnb_MR_sub[8],resultsRF_mnb_MR_sub[8],resultsTFIGM_mnb_MR_sub[8], resultsSQRTTFIGM_mnb_MR_sub[8],resultsTFIGMimp_mnb_MR_sub[8],resultsSQRTTFIGMimp_mnb_MR_sub[8],resultsTFMEU_mnb_MR_sub[8] ])
k14000 = k14000*100
print('14000', k14000)


# In[1211]:


#grafico Polarity
np.set_printoptions(precision=2)
cont =0
for i in resultsTF_mnb_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1

print('---')    
cont =0
for i in resultsTFIDF_mnb_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1

print('---')    
cont =0
for i in resultsDelta_mnb_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1    
    
print('---')    
cont =0
for i in resultsICF_mnb_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       
    
print('---')    
cont =0
for i in resultsRF_mnb_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       
    
print('---')    
cont =0
for i in resultsTFIGM_mnb_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1 

print('---')    
cont =0
for i in resultsSQRTTFIGM_mnb_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1 

print('---')    
cont =0
for i in resultsTFIGMimp_mnb_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1     

print('---')    
cont =0
for i in resultsSQRTTFIGMimp_mnb_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       

print('---')    
cont =0
for i in resultsTFMEU_mnb_MR_sub:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1      
 


# In[83]:


X,y = readSubjectivityCross()
resultsRF_svm_sub = processAll(lst, TfrfVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsRF_mnb_MR_sub = processAll(lst, TfrfVectorizer(sublinear_tf=False), 5,MultinomialNB())


# In[1155]:


X,y = readSarcasmCross() #readPolarityCross()
resultsTFIGM_svm_MR_sar = processAll(lst, TfIGMVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsSQRTTFIGM_svm_MR_sar = processAll(lst, SQRTTfIGMVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsTFIGMimp_svm_MR_sar = processAll(lst, TfIGMimpVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsSQRTTFIGMimp_svm_MR_sar = processAll(lst, SQRTTfIGMimpVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))

resultsICF_svm_MR_sar = processAll(lst, TfIcfVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsDelta_svm_MR_sar = processAll(lst, TfDELTAidf(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsTFMEU_svm_MR_sar = processAll(lst, TfidfcrfVectorizer1(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsTFIDF_svm_MR_sar = processAll(lst, TfidfTransformer(sublinear_tf=False, smooth_idf=False), 5, svm.SVC(kernel='linear', C=1))
resultsTF_svm_MR_sar = processAll(lst, TfidfTransformer(use_idf=False, sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsRF_svm_sar = processAll(lst, TfrfVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))

resultsTFIGM_mnb_MR_sar = processAll(lst, TfIGMVectorizer(sublinear_tf=False), 5,MultinomialNB())
resultsSQRTTFIGM_mnb_MR_sar = processAll(lst, SQRTTfIGMVectorizer(sublinear_tf=False), 5,MultinomialNB())
resultsTFIGMimp_mnb_MR_sar = processAll(lst, TfIGMimpVectorizer(sublinear_tf=False), 5,MultinomialNB())
resultsSQRTTFIGMimp_mnb_MR_sar = processAll(lst, SQRTTfIGMimpVectorizer(sublinear_tf=False), 5,MultinomialNB())

resultsICF_mnb_MR_sar = processAll(lst, TfIcfVectorizer(sublinear_tf=False), 5,MultinomialNB())
resultsDelta_mnb_MR_sar = processAll(lst, TfDELTAidf(sublinear_tf=False), 5,MultinomialNB())
resultsTFMEU_mnb_MR_sar = processAll(lst, TfidfcrfVectorizer1(sublinear_tf=False), 5,MultinomialNB())
resultsTFIDF_mnb_MR_sar = processAll(lst, TfidfTransformer(sublinear_tf=False, smooth_idf=False), 5,MultinomialNB())
resultsTF_mnb_MR_sar = processAll(lst, TfidfTransformer(use_idf=False, sublinear_tf=False), 5,MultinomialNB())
resultsRF_mnb_MR_sar = processAll(lst, TfrfVectorizer(sublinear_tf=False), 5,MultinomialNB())


# In[ ]:





# In[ ]:





# In[82]:


X,y = readSarcasmCross()
resultsRF_svm_sar = processAll(lst, TfrfVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsRF_mnb_MR_sar = processAll(lst, TfrfVectorizer(sublinear_tf=False), 5,MultinomialNB())


# In[1181]:


np.set_printoptions(precision=2)
k500=np.array([resultsTF_mnb_MR_sar[0], resultsTFIDF_mnb_MR_sar[0], resultsDelta_mnb_MR_sar[0], resultsICF_mnb_MR_sar[0],resultsRF_mnb_MR_sar[0],resultsTFIGM_mnb_MR_sar[0], resultsSQRTTFIGM_mnb_MR_sar[0],resultsTFIGMimp_mnb_MR_sar[0],resultsSQRTTFIGMimp_mnb_MR_sar[0],resultsTFMEU_mnb_MR_sar[0] ])
k500=k500*100
print('500', k500)

np.set_printoptions(precision=2)
k1000=np.array([resultsTF_mnb_MR_sar[1], resultsTFIDF_mnb_MR_sar[1], resultsDelta_mnb_MR_sar[1], resultsICF_mnb_MR_sar[1],resultsRF_mnb_MR_sar[1],resultsTFIGM_mnb_MR_sar[1], resultsSQRTTFIGM_mnb_MR_sar[1],resultsTFIGMimp_mnb_MR_sar[1],resultsSQRTTFIGMimp_mnb_MR_sar[1],resultsTFMEU_mnb_MR_sar[1] ])
k1000=k1000*100
print('1000', k1000)

np.set_printoptions(precision=2)
k2000=np.array([resultsTF_mnb_MR_sar[2], resultsTFIDF_mnb_MR_sar[2], resultsDelta_mnb_MR_sar[2], resultsICF_mnb_MR_sar[2],resultsRF_mnb_MR_sar[2],resultsTFIGM_mnb_MR_sar[2], resultsSQRTTFIGM_mnb_MR_sar[2],resultsTFIGMimp_mnb_MR_sar[2],resultsSQRTTFIGMimp_mnb_MR_sar[2],resultsTFMEU_mnb_MR_sar[2] ])
k2000=k2000*100
print('2000', k2000)

np.set_printoptions(precision=2)
k4000=np.array([resultsTF_mnb_MR_sar[3], resultsTFIDF_mnb_MR_sar[3], resultsDelta_mnb_MR_sar[3], resultsICF_mnb_MR_sar[3],resultsRF_mnb_MR_sar[3],resultsTFIGM_mnb_MR_sar[3], resultsSQRTTFIGM_mnb_MR_sar[3],resultsTFIGMimp_mnb_MR_sar[3],resultsSQRTTFIGMimp_mnb_MR_sar[3],resultsTFMEU_mnb_MR_sar[3] ])
k4000=k4000*100
print('4000', k4000)

np.set_printoptions(precision=2)
k6000=np.array([resultsTF_mnb_MR_sar[4], resultsTFIDF_mnb_MR_sar[4], resultsDelta_mnb_MR_sar[4], resultsICF_mnb_MR_sar[4],resultsRF_mnb_MR_sar[4],resultsTFIGM_mnb_MR_sar[4], resultsSQRTTFIGM_mnb_MR_sar[4],resultsTFIGMimp_mnb_MR_sar[4],resultsSQRTTFIGMimp_mnb_MR_sar[4],resultsTFMEU_mnb_MR_sar[4] ])
k6000=k6000*100
print('6000', k6000)

np.set_printoptions(precision=2)
k8000=np.array([resultsTF_mnb_MR_sar[5], resultsTFIDF_mnb_MR_sar[5], resultsDelta_mnb_MR_sar[5], resultsICF_mnb_MR_sar[5],resultsRF_mnb_MR_sar[5],resultsTFIGM_mnb_MR_sar[5], resultsSQRTTFIGM_mnb_MR_sar[5],resultsTFIGMimp_mnb_MR_sar[5],resultsSQRTTFIGMimp_mnb_MR_sar[5],resultsTFMEU_mnb_MR_sar[5] ])
k8000 = k8000*100
print('8000', k8000)

np.set_printoptions(precision=2)
k10000=np.array([resultsTF_mnb_MR_sar[6], resultsTFIDF_mnb_MR_sar[6], resultsDelta_mnb_MR_sar[6], resultsICF_mnb_MR_sar[6],resultsRF_mnb_MR_sar[6],resultsTFIGM_mnb_MR_sar[6], resultsSQRTTFIGM_mnb_MR_sar[6],resultsTFIGMimp_mnb_MR_sar[6],resultsSQRTTFIGMimp_mnb_MR_sar[6],resultsTFMEU_mnb_MR_sar[6] ])
k10000 = k10000*100
print('10000', k10000)

np.set_printoptions(precision=2)
k12000=np.array([resultsTF_mnb_MR_sar[7], resultsTFIDF_mnb_MR_sar[7], resultsDelta_mnb_MR_sar[7], resultsICF_mnb_MR_sar[7],resultsRF_mnb_MR_sar[7],resultsTFIGM_mnb_MR_sar[7], resultsSQRTTFIGM_mnb_MR_sar[7],resultsTFIGMimp_mnb_MR_sar[7],resultsSQRTTFIGMimp_mnb_MR_sar[7],resultsTFMEU_mnb_MR_sar[7] ])
k12000 = k12000*100
print('12000', k12000)

np.set_printoptions(precision=2)
k14000=np.array([resultsTF_mnb_MR_sar[8], resultsTFIDF_mnb_MR_sar[8], resultsDelta_mnb_MR_sar[8], resultsICF_mnb_MR_sar[8],resultsRF_mnb_MR_sar[8],resultsTFIGM_mnb_MR_sar[8], resultsSQRTTFIGM_mnb_MR_sar[8],resultsTFIGMimp_mnb_MR_sar[8],resultsSQRTTFIGMimp_mnb_MR_sar[8],resultsTFMEU_mnb_MR_sar[8] ])
k14000 = k14000*100
print('14000', k14000)


# In[1207]:


#grafico Polarity
np.set_printoptions(precision=2)
cont =0
for i in resultsTF_mnb_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1

print('---')    
cont =0
for i in resultsTFIDF_mnb_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1

print('---')    
cont =0
for i in resultsDelta_mnb_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1    
    
print('---')    
cont =0
for i in resultsICF_mnb_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       
    
print('---')    
cont =0
for i in resultsRF_mnb_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       
    
print('---')    
cont =0
for i in resultsTFIGM_mnb_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1 

print('---')    
cont =0
for i in resultsSQRTTFIGM_mnb_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1 

print('---')    
cont =0
for i in resultsTFIGMimp_mnb_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1     

print('---')    
cont =0
for i in resultsSQRTTFIGMimp_mnb_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       

print('---')    
cont =0
for i in resultsTFMEU_mnb_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1      
 


# In[1180]:


np.set_printoptions(precision=2)
k500=np.array([resultsTF_svm_MR_sar[0], resultsTFIDF_svm_MR_sar[0], resultsDelta_svm_MR_sar[0], resultsICF_svm_MR_sar[0],resultsRF_svm_sar[0],resultsTFIGM_svm_MR_sar[0], resultsSQRTTFIGM_svm_MR_sar[0],resultsTFIGMimp_svm_MR_sar[0],resultsSQRTTFIGMimp_svm_MR_sar[0],resultsTFMEU_svm_MR_sar[0] ])
k500=k500*100
print('500', k500)

np.set_printoptions(precision=2)
k1000=np.array([resultsTF_svm_MR_sar[1], resultsTFIDF_svm_MR_sar[1], resultsDelta_svm_MR_sar[1], resultsICF_svm_MR_sar[1],resultsRF_svm_sar[1],resultsTFIGM_svm_MR_sar[1], resultsSQRTTFIGM_svm_MR_sar[1],resultsTFIGMimp_svm_MR_sar[1],resultsSQRTTFIGMimp_svm_MR_sar[1],resultsTFMEU_svm_MR_sar[1] ])
k1000=k1000*100
print('1000', k1000)

np.set_printoptions(precision=2)
k2000=np.array([resultsTF_svm_MR_sar[2], resultsTFIDF_svm_MR_sar[2], resultsDelta_svm_MR_sar[2], resultsICF_svm_MR_sar[2],resultsRF_svm_sar[2],resultsTFIGM_svm_MR_sar[2], resultsSQRTTFIGM_svm_MR_sar[2],resultsTFIGMimp_svm_MR_sar[2],resultsSQRTTFIGMimp_svm_MR_sar[2],resultsTFMEU_svm_MR_sar[2] ])
k2000=k2000*100
print('2000', k2000)

np.set_printoptions(precision=2)
k4000=np.array([resultsTF_svm_MR_sar[3], resultsTFIDF_svm_MR_sar[3], resultsDelta_svm_MR_sar[3], resultsICF_svm_MR_sar[3],resultsRF_svm_sar[3],resultsTFIGM_svm_MR_sar[3], resultsSQRTTFIGM_svm_MR_sar[3],resultsTFIGMimp_svm_MR_sar[3],resultsSQRTTFIGMimp_svm_MR_sar[3] ,resultsTFMEU_svm_MR_sar[3]])
k4000=k4000*100
print('4000', k4000)

np.set_printoptions(precision=2)
k6000=np.array([resultsTF_svm_MR_sar[4], resultsTFIDF_svm_MR_sar[4], resultsDelta_svm_MR_sar[4], resultsICF_svm_MR_sar[4],resultsRF_svm_sar[4],resultsTFIGM_svm_MR_sar[4], resultsSQRTTFIGM_svm_MR_sar[4],resultsTFIGMimp_svm_MR_sar[4],resultsSQRTTFIGMimp_svm_MR_sar[4],resultsTFMEU_svm_MR_sar[4] ])
k6000=k6000*100
print('6000', k6000)

np.set_printoptions(precision=2)
k8000=np.array([resultsTF_svm_MR_sar[5], resultsTFIDF_svm_MR_sar[5], resultsDelta_svm_MR_sar[5], resultsICF_svm_MR_sar[5],resultsRF_svm_sar[5],resultsTFIGM_svm_MR_sar[5], resultsSQRTTFIGM_svm_MR_sar[5],resultsTFIGMimp_svm_MR_sar[5],resultsSQRTTFIGMimp_svm_MR_sar[5],resultsTFMEU_svm_MR_sar[5] ])
k8000 = k8000*100
print('8000', k8000)

np.set_printoptions(precision=2)
k10000=np.array([resultsTF_svm_MR_sar[6], resultsTFIDF_svm_MR_sar[6], resultsDelta_svm_MR_sar[6], resultsICF_svm_MR_sar[6],resultsRF_svm_sar[6],resultsTFIGM_svm_MR_sar[6], resultsSQRTTFIGM_svm_MR_sar[6],resultsTFIGMimp_svm_MR_sar[6],resultsSQRTTFIGMimp_svm_MR_sar[6],resultsTFMEU_svm_MR_sar[6] ])
k10000 = k10000*100	
print('10000', k10000)

np.set_printoptions(precision=2)
k12000=np.array([resultsTF_svm_MR_sar[7], resultsTFIDF_svm_MR_sar[7], resultsDelta_svm_MR_sar[7], resultsICF_svm_MR_sar[7],resultsRF_svm_sar[7],resultsTFIGM_svm_MR_sar[7], resultsSQRTTFIGM_svm_MR_sar[7],resultsTFIGMimp_svm_MR_sar[7],resultsSQRTTFIGMimp_svm_MR_sar[7],resultsTFMEU_svm_MR_sar[7] ])
k12000 = k12000*100
print('12000', k12000)

np.set_printoptions(precision=2)
k14000=np.array([resultsTF_svm_MR_sar[8], resultsTFIDF_svm_MR_sar[8], resultsDelta_svm_MR_sar[8], resultsICF_svm_MR_sar[8],resultsRF_svm_sar[8],resultsTFIGM_svm_MR_sar[8], resultsSQRTTFIGM_svm_MR_sar[8],resultsTFIGMimp_svm_MR_sar[8],resultsSQRTTFIGMimp_svm_MR_sar[8],resultsTFMEU_svm_MR_sar[8] ])
k14000 = k14000*100
print('14000', k14000)


# In[1208]:


#grafico Polarity
np.set_printoptions(precision=2)
cont =0
for i in resultsTF_svm_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1

print('---')    
cont =0
for i in resultsTFIDF_svm_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1

print('---')    
cont =0
for i in resultsDelta_svm_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1    
    
print('---')    
cont =0
for i in resultsICF_svm_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       
    
print('---')    
cont =0
for i in resultsRF_svm_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       
    
print('---')    
cont =0
for i in resultsTFIGM_svm_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1 

print('---')    
cont =0
for i in resultsSQRTTFIGM_svm_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1 

print('---')    
cont =0
for i in resultsTFIGMimp_svm_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1     

print('---')    
cont =0
for i in resultsSQRTTFIGMimp_svm_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       

print('---')    
cont =0
for i in resultsTFMEU_svm_MR_sar:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1      
 


# In[1143]:


X,y = readPolarityCross()
resultsTFIGM_svm_MR_Pol = processAll(lst, TfIGMVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsSQRTTFIGM_svm_MR_Pol = processAll(lst, SQRTTfIGMVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsTFIGMimp_svm_MR_Pol = processAll(lst, TfIGMimpVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsSQRTTFIGMimp_svm_MR_Pol = processAll(lst, SQRTTfIGMimpVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))

resultsICF_svm_MR_Pol = processAll(lst, TfIcfVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsDelta_svm_MR_Pol = processAll(lst, TfDELTAidf(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsTFMEU_svm_MR_Pol = processAll(lst, TfidfcrfVectorizer1(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsTFIDF_svm_MR_Pol = processAll(lst, TfidfTransformer(sublinear_tf=False, smooth_idf=False), 5, svm.SVC(kernel='linear', C=1))
resultsTF_svm_MR_Pol = processAll(lst, TfidfTransformer(use_idf=False, sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsRF_svm_Pol = processAll(lst, TfrfVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))

resultsTFIGM_mnb_MR_Pol = processAll(lst, TfIGMVectorizer(sublinear_tf=False), 5,MultinomialNB())
resultsSQRTTFIGM_mnb_MR_Pol = processAll(lst, SQRTTfIGMVectorizer(sublinear_tf=False), 5,MultinomialNB())
resultsTFIGMimp_mnb_MR_Pol = processAll(lst, TfIGMimpVectorizer(sublinear_tf=False), 5,MultinomialNB())
resultsSQRTTFIGMimp_mnb_MR_Pol = processAll(lst, SQRTTfIGMimpVectorizer(sublinear_tf=False), 5,MultinomialNB())

resultsICF_mnb_MR_Pol = processAll(lst, TfIcfVectorizer(sublinear_tf=False), 5,MultinomialNB())
resultsDelta_mnb_MR_Pol = processAll(lst, TfDELTAidf(sublinear_tf=False), 5,MultinomialNB())
resultsTFMEU_mnb_MR_Pol = processAll(lst, TfidfcrfVectorizer1(sublinear_tf=False), 5,MultinomialNB())
resultsTFIDF_mnb_MR_Pol = processAll(lst, TfidfTransformer(sublinear_tf=False, smooth_idf=False), 5,MultinomialNB())
resultsTF_mnb_MR_Pol = processAll(lst, TfidfTransformer(use_idf=False, sublinear_tf=False), 5,MultinomialNB())
resultsRF_mnb_MR_Pol = processAll(lst, TfrfVectorizer(sublinear_tf=False), 5,MultinomialNB())


# In[1140]:





# In[1142]:


k500=0.123123
print(np.round(0.565435*100, 2))


# In[81]:


X,y = readPolarityCross()
resultsRF_svm_Pol = processAll(lst, TfrfVectorizer(sublinear_tf=False), 5, svm.SVC(kernel='linear', C=1))
resultsRF_mnb_MR_Pol = processAll(lst, TfrfVectorizer(sublinear_tf=False), 5,MultinomialNB())


# In[1178]:


#np.set_printoptions(precision=2)
k500=np.array([resultsTF_svm_MR_Pol[0], resultsTFIDF_svm_MR_Pol[0], resultsDelta_svm_MR_Pol[0], resultsICF_svm_MR_Pol[0],resultsRF_svm_Pol[0],resultsTFIGM_svm_MR_Pol[0], resultsSQRTTFIGM_svm_MR_Pol[0],resultsTFIGMimp_svm_MR_Pol[0],resultsSQRTTFIGMimp_svm_MR_Pol[0],resultsTFMEU_svm_MR_Pol[0] ])
k500=np.round(k500*100,2)
print('500',k500)

np.set_printoptions(precision=2)
k1000=np.array([resultsTF_svm_MR_Pol[1], resultsTFIDF_svm_MR_Pol[1], resultsDelta_svm_MR_Pol[1], resultsICF_svm_MR_Pol[1],resultsRF_svm_Pol[1],resultsTFIGM_svm_MR_Pol[1], resultsSQRTTFIGM_svm_MR_Pol[1],resultsTFIGMimp_svm_MR_Pol[1],resultsSQRTTFIGMimp_svm_MR_Pol[1],resultsTFMEU_svm_MR_Pol[1] ])
k1000=np.round(k1000*100,2)
print('1000',k1000)

np.set_printoptions(precision=2)
k2000=np.array([resultsTF_svm_MR_Pol[2], resultsTFIDF_svm_MR_Pol[2], resultsDelta_svm_MR_Pol[2], resultsICF_svm_MR_Pol[2],resultsRF_svm_Pol[2],resultsTFIGM_svm_MR_Pol[2], resultsSQRTTFIGM_svm_MR_Pol[2],resultsTFIGMimp_svm_MR_Pol[2],resultsSQRTTFIGMimp_svm_MR_Pol[2],resultsTFMEU_svm_MR_Pol[2] ])
k2000=np.round(k2000*100,2)
print('2000',k2000)

np.set_printoptions(precision=2)
k4000=np.array([resultsTF_svm_MR_Pol[3], resultsTFIDF_svm_MR_Pol[3], resultsDelta_svm_MR_Pol[3], resultsICF_svm_MR_Pol[3],resultsRF_svm_Pol[3],resultsTFIGM_svm_MR_Pol[3], resultsSQRTTFIGM_svm_MR_Pol[3],resultsTFIGMimp_svm_MR_Pol[3],resultsSQRTTFIGMimp_svm_MR_Pol[3],resultsTFMEU_svm_MR_Pol[3] ])
k4000=np.round(k4000*100,2)
print('4000',k4000)

np.set_printoptions(precision=2)
k6000=np.array([resultsTF_svm_MR_Pol[4], resultsTFIDF_svm_MR_Pol[4], resultsDelta_svm_MR_Pol[4], resultsICF_svm_MR_Pol[4],resultsRF_svm_Pol[4],resultsTFIGM_svm_MR_Pol[4], resultsSQRTTFIGM_svm_MR_Pol[4],resultsTFIGMimp_svm_MR_Pol[4],resultsSQRTTFIGMimp_svm_MR_Pol[4],resultsTFMEU_svm_MR_Pol[4] ])
k6000=np.round(k6000*100,2)
print('6000',k6000)

np.set_printoptions(precision=2)
k8000=np.array([resultsTF_svm_MR_Pol[5], resultsTFIDF_svm_MR_Pol[5], resultsDelta_svm_MR_Pol[5], resultsICF_svm_MR_Pol[5],resultsRF_svm_Pol[5],resultsTFIGM_svm_MR_Pol[5], resultsSQRTTFIGM_svm_MR_Pol[5],resultsTFIGMimp_svm_MR_Pol[5],resultsSQRTTFIGMimp_svm_MR_Pol[5],resultsTFMEU_svm_MR_Pol[5] ])
k8000 = np.round(k8000*100,2)
print('8000',k8000)

np.set_printoptions(precision=2)
k10000=np.array([resultsTF_svm_MR_Pol[6], resultsTFIDF_svm_MR_Pol[6], resultsDelta_svm_MR_Pol[6], resultsICF_svm_MR_Pol[6],resultsRF_svm_Pol[6],resultsTFIGM_svm_MR_Pol[6], resultsSQRTTFIGM_svm_MR_Pol[6],resultsTFIGMimp_svm_MR_Pol[6],resultsSQRTTFIGMimp_svm_MR_Pol[6],resultsTFMEU_svm_MR_Pol[6] ])
k10000 = np.round(k10000*100,2)	
print('10000',k10000)

np.set_printoptions(precision=2)
k12000=np.array([resultsTF_svm_MR_Pol[7], resultsTFIDF_svm_MR_Pol[7], resultsDelta_svm_MR_Pol[7], resultsICF_svm_MR_Pol[7],resultsRF_svm_Pol[7],resultsTFIGM_svm_MR_Pol[7], resultsSQRTTFIGM_svm_MR_Pol[7],resultsTFIGMimp_svm_MR_Pol[7],resultsSQRTTFIGMimp_svm_MR_Pol[7],resultsTFMEU_svm_MR_Pol[7] ])
k12000 = np.round(k12000*100,2)
print('12000',k12000)

np.set_printoptions(precision=2)
k14000=np.array([resultsTF_svm_MR_Pol[8], resultsTFIDF_svm_MR_Pol[8], resultsDelta_svm_MR_Pol[8], resultsICF_svm_MR_Pol[8],resultsRF_svm_Pol[8],resultsTFIGM_svm_MR_Pol[8], resultsSQRTTFIGM_svm_MR_Pol[8],resultsTFIGMimp_svm_MR_Pol[8],resultsSQRTTFIGMimp_svm_MR_Pol[8],resultsTFMEU_svm_MR_Pol[8] ])
k14000 = np.round(k14000*100,2)
print('14000',k14000)


# In[1203]:


#grafico Polarity
np.set_printoptions(precision=2)
cont =0
for i in resultsTF_svm_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1

print('---')    
cont =0
for i in resultsTFIDF_svm_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1

print('---')    
cont =0
for i in resultsDelta_svm_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1    
    
print('---')    
cont =0
for i in resultsICF_svm_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       
    
print('---')    
cont =0
for i in resultsRF_svm_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       
    
print('---')    
cont =0
for i in resultsTFIGM_svm_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1 

print('---')    
cont =0
for i in resultsSQRTTFIGM_svm_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1 

print('---')    
cont =0
for i in resultsTFIGMimp_svm_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1     

print('---')    
cont =0
for i in resultsSQRTTFIGMimp_svm_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       

print('---')    
cont =0
for i in resultsTFMEU_svm_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1      
 
    


# In[1205]:


#grafico Polarity
np.set_printoptions(precision=2)
cont =0
for i in resultsTF_mnb_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1

print('---')    
cont =0
for i in resultsTFIDF_mnb_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1

print('---')    
cont =0
for i in resultsDelta_mnb_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1    
    
print('---')    
cont =0
for i in resultsICF_mnb_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       
    
print('---')    
cont =0
for i in resultsRF_mnb_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       
    
print('---')    
cont =0
for i in resultsTFIGM_mnb_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1 

print('---')    
cont =0
for i in resultsSQRTTFIGM_mnb_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1 

print('---')    
cont =0
for i in resultsTFIGMimp_mnb_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1     

print('---')    
cont =0
for i in resultsSQRTTFIGMimp_mnb_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1       

print('---')    
cont =0
for i in resultsTFMEU_mnb_MR_Pol:
    print('(',lst[cont], ',', np.round(i*100,2), ')',sep="")
    cont=cont+1      
 


# In[1179]:


np.set_printoptions(precision=2)
k500=np.array([resultsTF_mnb_MR_Pol[0], resultsTFIDF_mnb_MR_Pol[0], resultsDelta_mnb_MR_Pol[0], resultsICF_mnb_MR_Pol[0],resultsRF_mnb_MR_Pol[0],resultsTFIGM_mnb_MR_Pol[0], resultsSQRTTFIGM_mnb_MR_Pol[0],resultsTFIGMimp_mnb_MR_Pol[0],resultsSQRTTFIGMimp_mnb_MR_Pol[0],resultsTFMEU_mnb_MR_Pol[0] ])
k500=k500*100
print('500', k500)

np.set_printoptions(precision=2)
k1000=np.array([resultsTF_mnb_MR_Pol[1], resultsTFIDF_mnb_MR_Pol[1], resultsDelta_mnb_MR_Pol[1], resultsICF_mnb_MR_Pol[1],resultsRF_mnb_MR_Pol[1],resultsTFIGM_mnb_MR_Pol[1], resultsSQRTTFIGM_mnb_MR_Pol[1],resultsTFIGMimp_mnb_MR_Pol[1],resultsSQRTTFIGMimp_mnb_MR_Pol[1],resultsTFMEU_mnb_MR_Pol[1] ])
k1000=k1000*100
print('1000', k1000)

np.set_printoptions(precision=2)
k2000=np.array([resultsTF_mnb_MR_Pol[2], resultsTFIDF_mnb_MR_Pol[2], resultsDelta_mnb_MR_Pol[2], resultsICF_mnb_MR_Pol[2],resultsRF_mnb_MR_Pol[2],resultsTFIGM_mnb_MR_Pol[2], resultsSQRTTFIGM_mnb_MR_Pol[2],resultsTFIGMimp_mnb_MR_Pol[2],resultsSQRTTFIGMimp_mnb_MR_Pol[2],resultsTFMEU_mnb_MR_Pol[2] ])
k2000=k2000*100
print('2000', k2000)

np.set_printoptions(precision=2)
k4000=np.array([resultsTF_mnb_MR_Pol[3], resultsTFIDF_mnb_MR_Pol[3], resultsDelta_mnb_MR_Pol[3], resultsICF_mnb_MR_Pol[3],resultsRF_mnb_MR_Pol[3],resultsTFIGM_mnb_MR_Pol[3], resultsSQRTTFIGM_mnb_MR_Pol[3],resultsTFIGMimp_mnb_MR_Pol[3],resultsSQRTTFIGMimp_mnb_MR_Pol[3],resultsTFMEU_mnb_MR_Pol[3] ])
k4000=k4000*100
print('4000', k4000)

np.set_printoptions(precision=2)
k6000=np.array([resultsTF_mnb_MR_Pol[4], resultsTFIDF_mnb_MR_Pol[4], resultsDelta_mnb_MR_Pol[4], resultsICF_mnb_MR_Pol[4],resultsRF_mnb_MR_Pol[4],resultsTFIGM_mnb_MR_Pol[4], resultsSQRTTFIGM_mnb_MR_Pol[4],resultsTFIGMimp_mnb_MR_Pol[4],resultsSQRTTFIGMimp_mnb_MR_Pol[4],resultsTFMEU_mnb_MR_Pol[4] ])
k6000=k6000*100
print('6000', k6000)

np.set_printoptions(precision=2)
k8000=np.array([resultsTF_mnb_MR_Pol[5], resultsTFIDF_mnb_MR_Pol[5], resultsDelta_mnb_MR_Pol[5], resultsICF_mnb_MR_Pol[5],resultsRF_mnb_MR_Pol[5],resultsTFIGM_mnb_MR_Pol[5], resultsSQRTTFIGM_mnb_MR_Pol[5],resultsTFIGMimp_mnb_MR_Pol[5],resultsSQRTTFIGMimp_mnb_MR_Pol[5],resultsTFMEU_mnb_MR_Pol[5] ])
k8000 = k8000*100
print('8000', k8000)

np.set_printoptions(precision=2)
k10000=np.array([resultsTF_mnb_MR_Pol[6], resultsTFIDF_mnb_MR_Pol[6], resultsDelta_mnb_MR_Pol[6], resultsICF_mnb_MR_Pol[6],resultsRF_mnb_MR_Pol[6],resultsTFIGM_mnb_MR_Pol[6], resultsSQRTTFIGM_mnb_MR_Pol[6],resultsTFIGMimp_mnb_MR_Pol[6],resultsSQRTTFIGMimp_mnb_MR_Pol[6],resultsTFMEU_mnb_MR_Pol[6] ])
k10000 = k10000*100	
print('10000', k10000)

np.set_printoptions(precision=2)
k12000=np.array([resultsTF_mnb_MR_Pol[7], resultsTFIDF_mnb_MR_Pol[7], resultsDelta_mnb_MR_Pol[7], resultsICF_mnb_MR_Pol[7],resultsRF_mnb_MR_Pol[7],resultsTFIGM_mnb_MR_Pol[7], resultsSQRTTFIGM_mnb_MR_Pol[7],resultsTFIGMimp_mnb_MR_Pol[7],resultsSQRTTFIGMimp_mnb_MR_Pol[7],resultsTFMEU_mnb_MR_Pol[7] ])
k12000 = k12000*100
print('12000', k12000)

np.set_printoptions(precision=2)
k14000=np.array([resultsTF_mnb_MR_Pol[8], resultsTFIDF_mnb_MR_Pol[8], resultsDelta_mnb_MR_Pol[8], resultsICF_mnb_MR_Pol[8],resultsRF_mnb_MR_Pol[8],resultsTFIGM_mnb_MR_Pol[8], resultsSQRTTFIGM_mnb_MR_Pol[8],resultsTFIGMimp_mnb_MR_Pol[8],resultsSQRTTFIGMimp_mnb_MR_Pol[8],resultsTFMEU_mnb_MR_Pol[8] ])
k14000 = k14000*100
print('14000', k14000)


# In[ ]:





# In[310]:



X,y = readPolarityCross()
resultsTFIGM_svm_MR_Pol = processAll(lst, TfIGMVectorizer(sublinear_tf=True), 5, svm.SVC(kernel='linear', C=1))
#resultsSQRTTFIGM_svm_MR_Pol = processAll(lst, SQRTTfIGMVectorizer(sublinear_tf=True), 5, svm.SVC(kernel='linear', C=1))
#resultsTFIGMimp_svm_MR_Pol = processAll(lst, TfIGMimpVectorizer(sublinear_tf=True), 5, svm.SVC(kernel='linear', C=1))
#resultsSQRTTFIGMimp_svm_MR_Pol = processAll(lst, SQRTTfIGMimpVectorizer(sublinear_tf=True), 5, svm.SVC(kernel='linear', C=1))

#resultsTFIGM_mnb_MR_Pol = processAll(lst, TfIGMVectorizer(sublinear_tf=True), 5,MultinomialNB())
#resultsSQRTTFIGM_mnb_MR_Pol = processAll(lst, SQRTTfIGMVectorizer(sublinear_tf=True), 5,MultinomialNB())
#resultsTFIGMimp_mnb_MR_Pol = processAll(lst, TfIGMimpVectorizer(sublinear_tf=True), 5,MultinomialNB())
#resultsSQRTTFIGMimp_mnb_MR_Pol = processAll(lst, SQRTTfIGMimpVectorizer(sublinear_tf=True), 5,MultinomialNB())

