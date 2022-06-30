import numpy as np
# from mne.decoding import CSP
from sklearn import metrics
import sklearn.feature_selection
from sklearn.feature_selection import mutual_info_classif
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP


class FBCSP(BaseEstimator, TransformerMixin):
    def __init__(self,band:list,fs:int,k=8,n_components=4,metric='logeuclid'):
        '''
         param:
         n_components: n_components in CSP
         band : freq point when constructing filter bank, e.g. [8,12,16,20,24,28,32]
         fs: the sample freq
         k: how many feature to select using MIBIF(there will be n_components*band features in total)
        '''
        self.csp = CSP(nfilter=n_components,log=True,metric=metric)
        self.band = band
        self.fs = fs
        self.k = k
        self.n_components = n_components
        self.metric = metric

    def __cheby2_bandpass(self,lowcut,highcut,fs):
        sos = signal.cheby2(10, 50, [lowcut, highcut], analog=False, btype="band", output="sos", fs=fs) # 阶次为10，在阻带处的最小衰减为50，带通频率为wp，返回sos形式的传递函数（一系列二阶tf串联成最终的tf）
        return sos
    
    def __cheby2_bandpass_filter(self,data,lowcut,highcut,fs):
        sos = self.__cheby2_bandpass(lowcut,highcut,fs)
        return signal.sosfilt(sos, data)

    def fit(self,X,y):
        '''
         param:
         X: (trail,channels,time)
         y: (trail,) the labels w.r.t. each trail
        '''
        # construct the filter bank
        for freq_cnt in range(len(self.band)):
            lower = self.band[freq_cnt]
            if lower==self.band[-1]:
                break
            higher=self.band[freq_cnt+1]
            filtered = self.__cheby2_bandpass_filter(X,lower,higher,self.fs)
            cov_filtered = Covariances().transform(filtered)
            tmp_feature = self.csp.fit_transform(cov_filtered,y)
            if freq_cnt==0:
                features = tmp_feature
            else:
                features = np.concatenate((features,tmp_feature),axis=1)
        
        # get the best k features based on MIBIF algorithm
        self.select_K = sklearn.feature_selection.SelectKBest(mutual_info_classif,k=self.k).fit(features,y)
        return self

    def transform(self,X):
        '''
         param:
         X: (trail,channels,time)
        '''
        # construct the filter bank
        for freq_cnt in range(len(self.band)):
            lower = self.band[freq_cnt]
            if lower==self.band[-1]:
                break
            higher=self.band[freq_cnt+1]
            filtered = self.__cheby2_bandpass_filter(X,lower,higher,self.fs)
            cov_filtered = Covariances().transform(filtered)
            tmp_feature = self.csp.transform(cov_filtered)# diff
            if freq_cnt==0:
                features = tmp_feature
            else:
                features = np.concatenate((features,tmp_feature),axis=1)
        return self.select_K.transform(features)# diff       
    

if __name__ == '__main__':

    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score,KFold
    from sklearn.linear_model import LogisticRegression

    data = np.load('data/train/data.npz')

    epochs_data_train = data['X']
    labels = data['y']

    # cross validation
    # cv = KFold(n_splits=10, shuffle=True, random_state=42)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    # Assemble a classifier
    fbcsp = FBCSP(n_components=4,band=[8,16,24,32],fs=250,k=4)
    pipe = Pipeline([('FBCSP',fbcsp),
                    ('classify',LogisticRegression())])
    scores = cross_val_score(pipe, epochs_data_train, labels, cv=cv, n_jobs=-1)
    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("FBCSP + LDA Classification accuracy: %f / Chance level: %f" %
        (np.mean(scores), class_balance))




