# interface on the test set

# generic import
import logging
import numpy as np
import pandas as pd
from tabulate import tabulate
# import seaborn as sns
from matplotlib import pyplot as plt
import argparse


# pyriemann import
from pyriemann.classification import MDM, TSclassifier,FgMDM
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP,SPoC
from pyriemann.tangentspace import TangentSpace,FGDA

# sklearn imports
from sklearn.model_selection import cross_val_score, KFold,RepeatedStratifiedKFold,GridSearchCV,LeavePGroupsOut
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,SVR
from sklearn import neural_network
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier

# self implement
from fbcsp import FBCSP
from TSFeatureExtractor import TSFeatureExtractor
from BrainNetv1 import BrainNetv1
from BrainNetv2 import BrainNetv2
from BrainNetv3 import BrainNetv3
from utils.preprocess import rs_align,es_align
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

logger.info('## FBCSP + classifier train')

param_grid = [
    {
        'FBCSP__n_components':[2,4,6,8],
        'FBCSP__k':[2,4,8,'all'],
        'FBCSP__band':[[8,12,16]],#FIXME:主要就是8-12这个频段最有用，再细分成8-10-12就会变差，同时稍微附加一点12-16会比较好,[8,14]的效果都不如[8,12,16]效果好
        'FBCSP__metric':['euclid','logeuclid','riemann'],
        'classify':[RandomForestClassifier(n_estimators=10,max_depth=5),SVC(C=0.5)]# FIXME:不对随机森林算法做约束的话会严重的过拟合
    }
]

pipe = Pipeline([('FBCSP',FBCSP(band=[8,12,16,20,24,28,32],fs=250)),
                ('classify',SVC())])
search = GridSearchCV(estimator=pipe,param_grid=param_grid,cv=cv,n_jobs=-1,return_train_score=True)
search.fit(data_train,labels,groups=groups)

results_df = pd.DataFrame(search.cv_results_)
results_df = results_df.sort_values(by=["rank_test_score"])
results_df = results_df.set_index(
    results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
).rename_axis("abbr.")
results_df['test_score']=["%0.3f (+/-%0.03f)" % (mean, std) for mean,std in zip(results_df['mean_test_score'],results_df['std_test_score'])]
results_df['train_score']=["%0.3f (+/-%0.03f)" % (mean, std) for mean,std in zip(results_df['mean_train_score'],results_df['std_train_score'])]
logger.info(tabulate(results_df[["test_score","train_score","params"]],headers='keys',tablefmt="github"))

logger.info('## FBCSP predict')
y_pred=[]
for i in range(5,9):
    s = np.load('data/test/S'+str(i)+'.npz')
    x = s['X']
    if args.data=='ra':
        x = rs_align(x)
        y_pred.append(search.predict(x))
    elif args.data=='ea':
        x = es_align(x)
        y_pred.append(search.predict(x))
    else:
        y_pred.append(search.predict(x))

if args.data=='ra':
    np.savetxt('ra_res.csv',y_pred,delimiter=',',fmt='%i')
elif args.data=='ea':
    np.savetxt('ea_res.csv',y_pred,delimiter=',',fmt='%i')
else:
    np.savetxt('res.csv',y_pred,delimiter=',',fmt='%i')

