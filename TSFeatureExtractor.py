import numpy as np
from sklearn.feature_selection import SelectFdr
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.tangentspace import TangentSpace

class TSFeatureExtractor(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.TS = TangentSpace()
        self.fdr = SelectFdr()

    def fit(self,X,y):
        '''
         param:
         X: (trials, n_channels, n_channels) 
         y: (trials,) the labels w.r.t each trial
        '''
        S = self.TS.fit_transform(X)
        self.u,sigma,v = np.linalg.svd(S.T,compute_uv=True)
        S_0 = (self.u.T@(S.T)).T
        self.fdr.fit(S_0,y)
        return self

    def transform(self,X):
        S = self.TS.transform(X)
        S_0 = (self.u.T@(S.T)).T
        new_features = self.fdr.transform(S_0)
        return new_features

if __name__=='__main__':
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score,KFold
    from pyriemann.estimation import Covariances

    data = np.load('data/train/data.npz')
    data_train = data['X']
    labels = data['y']
    cov_data_train = Covariances().transform(data_train)


    # cross validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    # Assemble a classifier
    pipe = make_pipeline(TSFeatureExtractor(),SVC())# 69% dim=8
    # pipe = make_pipeline(TangentSpace(),SVC())# 74% dim=91
    scores = cross_val_score(pipe, cov_data_train, labels, cv=cv, n_jobs=1)
    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("TS+ANOVA feature select Classification accuracy: %f(+/-%f) / Chance level: %f" %
        (scores.mean(),scores.std(), class_balance))