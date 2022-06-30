# just a test method, not pretty good, so not include in readme and report
# generic import
import logging
import numpy as np
import pandas as pd
from sklearn import ensemble
from tabulate import tabulate
# import seaborn as sns
from matplotlib import pyplot as plt
import argparse

# sklearn imports
from sklearn.model_selection import cross_val_score,GridSearchCV,LeavePGroupsOut
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,SVR
from sklearn import neural_network
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,VotingClassifier

# self implement
from fbcsp import FBCSP
##############################################################################\
# logging and timestamp
import logging
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def logger_config(log_path,logging_name):
    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

logger = logger_config('log/'+timestamp+'.log','test')
parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str,help='data:orig_data,ra:ra_data,ea:ea_data')
args = parser.parse_args()
###############################################################################
# Set parameters and read data
if args.data=='ra':
    logger.info('# ra & Leave1Out')
    data = np.load('data/train/ra_data.npz')
elif args.data=='ea':
    logger.info('# ea & Leave1Out')
    data = np.load('data/train/ea_data.npz')
else:
    logger.info('# orig_data & Leave1Out')
    data = np.load('data/train/data.npz')


data_train = data['X']
labels = data['y']
groups = [1]*200+[2]*200+[3]*200+[4]*200

# cross validation
cv = LeavePGroupsOut(n_groups=1)

# compute covariance matrices
# cov_data_train = Covariances().transform(data_train)


#################################################################
from sklearn.base import TransformerMixin,BaseEstimator
class Reshape(TransformerMixin,BaseEstimator):
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self,X):
        return np.reshape(X,(X.shape[0],13,750))#FIXME: I hard code this (-1,channels,time) 
##################################################################
logger.info('## hard Bagging FBCSP clf')

clf_list = [
make_pipeline(
    Reshape(),
    FBCSP(n_components=6,k=8,band=[8,12,16],metric='euclid',fs=250),
    RandomForestClassifier(n_estimators=10,max_depth=5)
),
make_pipeline(
    Reshape(),
    FBCSP(n_components=6,k='all',band=[8,12,16],metric='euclid',fs=250),
    SVC(C=0.5,probability=True)
),
make_pipeline(
    Reshape(),
    FBCSP(band=[8,12,16],k=8,metric='logeuclid',n_components=6,fs=250),
    RandomForestClassifier(n_estimators=20,max_depth=3)
),
make_pipeline(
    Reshape(),
    FBCSP(band=[8,12,16],k='all',metric='riemann',n_components=4,fs=250),
    RandomForestClassifier(n_estimators=20,max_depth=3)
),
make_pipeline(
    Reshape(),
    FBCSP(band=[8,16],k='all',metric='euclid',n_components=4,fs=250),
    RandomForestClassifier(n_estimators=20,max_depth=3)
),
make_pipeline(
    Reshape(),
    FBCSP(band=[8,16],k=4,metric='euclid',n_components=8,fs=250),
    RandomForestClassifier(n_estimators=20,max_depth=3)
)
]

ensemble_clf = VotingClassifier(
    estimators=[(str(i),clf) for i,clf in enumerate(clf_list)],
    voting='hard'
)

param_grid=[
    {
        '0__fbcsp__n_components':[6],
    }
]

data_train = np.reshape(data_train,(data_train.shape[0],-1))# !!! because bagging classifier need the input to be (n,features)

# search = GridSearchCV(estimator=BaggingClassifier(base_estimator=clf_1),param_grid=param_grid,cv=cv,n_jobs=-1,return_train_score=True)
search = GridSearchCV(estimator=ensemble_clf,param_grid=param_grid,cv=cv,n_jobs=-1,return_train_score=True)
search.fit(data_train,labels,groups=groups)

results_df = pd.DataFrame(search.cv_results_)
results_df = results_df.sort_values(by=["rank_test_score"])
results_df = results_df.set_index(
    results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
).rename_axis("abbr.")
results_df['test_score']=["%0.3f (+/-%0.03f)" % (mean, std) for mean,std in zip(results_df['mean_test_score'],results_df['std_test_score'])]
results_df['train_score']=["%0.3f (+/-%0.03f)" % (mean, std) for mean,std in zip(results_df['mean_train_score'],results_df['std_train_score'])]
logger.info(tabulate(results_df[["test_score","train_score","params"]],headers='keys',tablefmt="github"))

# 效果不好，弃用