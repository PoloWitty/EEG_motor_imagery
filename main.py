"""
====================================================================
Motor imagery classification
====================================================================

Classify Motor imagery data with Riemannian Geometry.
"""
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
cov_data_train = Covariances().transform(data_train)

###############################################################################
# # Classification with Minimum distance to mean
# mdm = MDM(metric=dict(mean='riemann', distance='riemann'))

# # Use scikit-learn Pipeline with cross_val_score function
# scores = cross_val_score(mdm, cov_data_train, labels, groups=groups, cv=cv, n_jobs=1)

# # Printing the results
# class_balance = np.mean(labels == labels[0])
# class_balance = max(class_balance, 1. - class_balance)
# logger.info("MDM Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
#                                                               class_balance))

###############################################################################
# Classification with Tangent Space Logistic Regression

#----------------
# testing
#----------------

# # # clf = TSclassifier()
# clf = make_pipeline(
#     TangentSpace(metric="riemann"),
#    # LogisticRegression(),
#     SVC()
# )
# # Use scikit-learn Pipeline with cross_val_score function
# scores = cross_val_score(clf, cov_data_train, labels, groups=groups,cv=cv, n_jobs=-1)

# # Printing the results
# class_balance = np.mean(labels == labels[0])
# class_balance = max(class_balance, 1. - class_balance)
# logger.info("Tangent space Classification accuracy: %f(+/- %f) / Chance level: %f" %
#       (scores.mean(),scores.std(), class_balance))

# #--------------------
# # grid search
# #--------------------
# logger.info('## TangentSpace Mapping')
# param_grid = [
#     # {
#     #     # 'TangentSpace__tsupdate':[False,True],# FIXME:做不做update不具有显著性差异
#     #     'classify':[LogisticRegression(),SVC(),GaussianNB(priors=[.5, .5])]
#     # },
#     # {'classify__kernel':['rbf','poly']}#FIXME:rbf的效果略好于poly
#     {
#         'TangentSpace':[TSFeatureExtractor(),TangentSpace()],
#         'classify':[SVC(),LogisticRegression(),LinearDiscriminantAnalysis(),SVC(kernel='poly')]
#         # FIXME: 直接用TangentSpace不做特征选择，如果后续的分类器比较复杂的话就会过拟合；如果分类器是像逻辑斯蒂回归这样比较简单的就不会。
#         # 如果用TSFeatureExtractor做了特征选择，基本不管分类器怎样，效果都还行，但都在70左右，不管是训练集还是测试集
#     }
# ]

# pipe = Pipeline([('TangentSpace',TangentSpace(metric="riemann")),
#                 ('classify',SVC())])
# search = GridSearchCV(estimator=pipe,param_grid=param_grid,cv=cv,n_jobs=-1,return_train_score=True)
# search.fit(cov_data_train,labels,groups=groups)

# results_df = pd.DataFrame(search.cv_results_)
# results_df = results_df.sort_values(by=["rank_test_score"])
# results_df = results_df.set_index(
#     results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
# ).rename_axis("abbr.")
# results_df['test_score']=["%0.3f (+/-%0.03f)" % (mean, std) for mean,std in zip(results_df['mean_test_score'],results_df['std_test_score'])]
# results_df['train_score']=["%0.3f (+/-%0.03f)" % (mean, std) for mean,std in zip(results_df['mean_train_score'],results_df['std_train_score'])]
# logger.info(tabulate(results_df[["test_score","train_score","params"]],headers='keys',tablefmt="github"))

###############################################################################
# # Classification with CSP + logistic regression

#-------------------
# testing
#-------------------

# Assemble a classifier
# lr = LogisticRegression()
# # csp = mne.decoding.CSP(n_components=4, log=True)# mne's CSP
# csp = CSP(nfilter=4,log=True,metric='riemann') # pyriemann's CSP
# clf = Pipeline([('CSP',csp),
#                 ('classify',lr)])
# scores = cross_val_score(clf, cov_data_train, labels, groups=groups, cv=cv, n_jobs=-1)
# # Printing the results
# class_balance = np.mean(labels == labels[0])
# class_balance = max(class_balance, 1. - class_balance)
# logger.info("CSP + LDA Classification accuracy: %f(+/-%f) / Chance level: %f" %
#       (scores.mean(),scores.std(), class_balance))

# #-------------------
# # grid search
# #-------------------

# logger.info('## CSP + classifier')

# param_grid = [
#       {
#             "CSP__nfilter":[6],
#             "CSP__metric":['riemann', 'logeuclid', 'euclid'],
#             "classify":[SVC(C=0.5)] 
#       },
#     #   {
#     #         "CSP__nfilter":[4,5,6],
#     #         "CSP__metric":['logeuclid', 'euclid'],
#     #         "classify":[GaussianNB(priors=[.5, .5]),SVC(C=.8, gamma="scale", kernel="rbf"),LinearDiscriminantAnalysis()] 
#     #   },
#       # {
#       #       "CSP__nfilter":[4,5,6],
#       #       "CSP__metric":['logeuclid', 'euclid'],
#       #      # "classify":[neural_network.MLPClassifier(hidden_layer_sizes=(10), activation="relu",
#       #                                           solver='adam', alpha=0.0001,batch_size='auto',
#       #                                           learning_rate_init=0.001, power_t=0.5, max_iter=200 ,tol=1e-4)] 
#       # }
#       ]


# pipe = Pipeline([('CSP', CSP()), 
#                   ('classify', LogisticRegression())])
# search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv,n_jobs=-1,return_train_score=True)
# search.fit(cov_data_train, labels,groups=groups)


# results_df = pd.DataFrame(search.cv_results_)
# results_df = results_df.sort_values(by=["rank_test_score"])
# results_df = results_df.set_index(
#     results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
# ).rename_axis("abbr.")
# results_df['test_score']=["%0.3f (+/-%0.03f)" % (mean, std) for mean,std in zip(results_df['mean_test_score'],results_df['std_test_score'])]
# results_df['train_score']=["%0.3f (+/-%0.03f)" % (mean, std) for mean,std in zip(results_df['mean_train_score'],results_df['std_train_score'])]
# logger.info(tabulate(results_df[["test_score","train_score","params"]],headers='keys',tablefmt="github"))

##############################################################################
# classify with FBCSP+logisticRegression


#--------
# testing
#--------

# Assemble a classifier
fbcsp = FBCSP(n_components=4,band=[8,12,16,20,24,28,32],fs=250,k=16)
pipe = Pipeline([('FBCSP',fbcsp),
                ('classify',LogisticRegression())])
scores = cross_val_score(pipe, data_train, labels,groups=groups, cv=cv, n_jobs=-1)
# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
logger.info("FBCSP + LDA Classification accuracy: %f+/-%f / Chance level: %f" %
      (scores.mean(), scores.std() , class_balance))

#---------
# grid search
#---------

# logger.info('## FBCSP + classifier\n')

# param_grid = [
#     # {
#     #     'FBCSP__n_components':[4,6,8],
#     #     'FBCSP__k':[16,24,32],
#     #     'classify':[LogisticRegression(),SVC()]
#     # },
#     # {
#     #     'FBCSP__n_components':[4],
#     #     'FBCSP__k':[16],
#     #     'FBCSP__band':[[8,12,16,20,24,28,32],[8,16,24,32],[8,20,32]],
#     #     'classify':[SVC()]
#     # },
#     # {
#     #     'FBCSP__n_components':[4],
#     #     'FBCSP__k':[8],
#     #     'FBCSP__band':[[8,16,24,32],[8,20,32]],
#     #     'classify':[SVC(),SVR()]# FIXME:res：SVR效果不好，0.5都达不到，不用再试了
#     # },
#     # {
#     #     'FBCSP__n_components':[4,6,8],
#     #     'FBCSP__k':[2,4,8],
#     #     'FBCSP__band':[[8,32],[8,20,32]],# FIXME:res：不构建filter bank的效果最好，分的band越多，效果越差
#     #     'classify':[SVC(kernel='rbf'),LogisticRegression(),SVC(kernel='poly')]
#     # },
#     # {
#     #     'FBCSP__n_components':[4,6,8],
#     #     'FBCSP__k':[4,'all'],
#     #     'FBCSP__band':[[8,12,16],[8,32]],
#     #     'FBCSP__metric':['logeuclid','euclid'],
#     #     'classify':[SVC()]
#     # }
#     # {
#     #     'FBCSP__n_components':[6],
#     #     'FBCSP__k':[2,4,8,'all'],
#     #     'FBCSP__band':[[8,12,16],[8,10,12,14,16]],
#     #     'FBCSP__metric':['euclid'],
#     #     'classify':[SVC(kernel='rbf',C=0.5),SVC(C=0.3),SVC(C=0.8)]
#     # },
#     {
#         'FBCSP__n_components':[6],
#         'FBCSP__k':[2,4,8,'all'],
#         'FBCSP__band':[[8,12,16]],#FIXME:主要就是8-12这个频段最有用，再细分成8-10-12就会变差，同时稍微附加一点12-16会比较好,[8,14]的效果都不如[8,12,16]效果好
#         'FBCSP__metric':['euclid','logeuclid'],
#         'classify':[RandomForestClassifier(n_estimators=10,max_depth=5),SVC(C=0.5)]# FIXME:不对随机森林算法做约束的话会严重的过拟合
#     }
# ]

# pipe = Pipeline([('FBCSP',FBCSP(band=[8,12,16,20,24,28,32],fs=250)),
#                 ('classify',SVC())])
# search = GridSearchCV(estimator=pipe,param_grid=param_grid,cv=cv,n_jobs=-1,return_train_score=True)
# search.fit(data_train,labels,groups=groups)

# results_df = pd.DataFrame(search.cv_results_)
# results_df = results_df.sort_values(by=["rank_test_score"])
# results_df = results_df.set_index(
#     results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
# ).rename_axis("abbr.")
# results_df['test_score']=["%0.3f (+/-%0.03f)" % (mean, std) for mean,std in zip(results_df['mean_test_score'],results_df['std_test_score'])]
# results_df['train_score']=["%0.3f (+/-%0.03f)" % (mean, std) for mean,std in zip(results_df['mean_train_score'],results_df['std_train_score'])]
# logger.info(tabulate(results_df[["test_score","train_score","params"]],headers='keys',tablefmt="github"))

##############################################################################
# BrainNetv1

#--------------
# testing
#--------------
# model = BrainNetv1(verbose=True,epochs=200,batch_size=100)

# scores = cross_val_score(model, data_train, labels, groups=groups, cv=cv, n_jobs=1)

# # Printing the results
# class_balance = np.mean(labels == labels[0])
# class_balance = max(class_balance, 1. - class_balance)
# logger.info("BrainNetv1 Classification accuracy: %f(+/-%f) / Chance level: %f" % (scores.mean(),scores.std(),
#                                     class_balance))

#---------------
# grid search
#---------------

# logger.info('## BrainNetv1 classify')
# param_grid = [
#     {
#         'epochs':[150,200,250,300,350,400],
#         'batch_size':[100,150,200],
#         'dropout_p':[0.2,0.4,0.6],
#         'data':[args.data]
#     }
# ]

# model = BrainNetv1(verbose=True)

# search = GridSearchCV(estimator=model,param_grid=param_grid,cv=cv,n_jobs=1,return_train_score=True)# 不能多线程
# search.fit(data_train,labels,groups=groups)

# results_df = pd.DataFrame(search.cv_results_)
# results_df = results_df.sort_values(by=["rank_test_score"])
# results_df = results_df.set_index(
#     results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
# ).rename_axis("abbr.")
# results_df['test_score']=["%0.3f (+/-%0.03f)" % (mean, std) for mean,std in zip(results_df['mean_test_score'],results_df['std_test_score'])]
# results_df['train_score']=["%0.3f (+/-%0.03f)" % (mean, std) for mean,std in zip(results_df['mean_train_score'],results_df['std_train_score'])]
# logger.info(tabulate(results_df[["test_score","train_score","params"]],headers='keys',tablefmt="github"))

##############################################################################
# BrainNetv2

#--------------
# testing
#--------------
# model = BrainNetv2(verbose=True,epochs=100,batch_size=50)

# scores = cross_val_score(model, data_train, labels, groups=groups, cv=cv, n_jobs=1)

# # Printing the results
# class_balance = np.mean(labels == labels[0])
# class_balance = max(class_balance, 1. - class_balance)
# logger.info("BrainNetv2 Classification accuracy: %f(+/-%f) / Chance level: %f" % (scores.mean(),scores.std(),
#                                     class_balance))

# #---------------
# # grid search
# #---------------

# logger.info('## BrainNetv2 classify')
# param_grid = [
#     # {
#     #     'epochs':[50,100,150,200],
#     #     'batch_size':[50,100,150],
#     #     'dropout_p':[0.2,0.4,0.6],
#     #     'data':[args.data],
#     # },
#     # {
#     #     'data':[args.data],
#     #     'd_ratio':[10,15,25,30,50,150,250],# 750=2*5*5*3*5
#     #     'head': [3,5],# FIXEME: 实验感觉好像3/5会比较好
#     # }
#     {
#         'data':[args.data],
#         'd_ratio':[15,30],
#         'head':[5],
#         'epochs':[500,1000]
#     }
# ]

# model = BrainNetv2(verbose=False,dropout_p=0.25,batch_size=100,h=6)# FIXEME:实验结果感觉跟batch_size,dropout_p关系不大，10个epoch能有70几，20个就80几了，所以平衡一下15可能会比较好，实验感觉h6个就挺好的，其实差别不是很大

# search = GridSearchCV(estimator=model,param_grid=param_grid,cv=cv,n_jobs=-1,return_train_score=True)# 用wandb的话就不能多线程
# search.fit(data_train,labels,groups=groups)

# results_df = pd.DataFrame(search.cv_results_)
# results_df = results_df.sort_values(by=["rank_test_score"])
# results_df = results_df.set_index(
#     results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
# ).rename_axis("abbr.")
# results_df['test_score']=["%0.3f (+/-%0.03f)" % (mean, std) for mean,std in zip(results_df['mean_test_score'],results_df['std_test_score'])]
# results_df['train_score']=["%0.3f (+/-%0.03f)" % (mean, std) for mean,std in zip(results_df['mean_train_score'],results_df['std_train_score'])]
# logger.info(tabulate(results_df[["test_score","train_score","params"]],headers='keys',tablefmt="github"))
##############################################################################
# BrainNetv3

#--------------
# testing
#--------------
# model = BrainNetv3(verbose=True,epochs=100,batch_size=50)

# scores = cross_val_score(model, data_train, labels, groups=groups, cv=cv, n_jobs=1)

# # Printing the results
# class_balance = np.mean(labels == labels[0])
# class_balance = max(class_balance, 1. - class_balance)
# logger.info("BrainNetv3 Classification accuracy: %f(+/-%f) / Chance level: %f" % (scores.mean(),scores.std(),
#                                     class_balance))

#---------------
# grid search
#---------------

# logger.info('## BrainNetv3 classify')
# param_grid = [
#     {
#         'T':[2,4,8,16,32,64],
#         'tau':[2.0,4.0,8.0,16.0],
#         'epochs':[1000]
#     }
# ]

# model = BrainNetv3(verbose=False,batch_size=100)

# search = GridSearchCV(estimator=model,param_grid=param_grid,cv=cv,n_jobs=-1,return_train_score=True)# 用wandb的话就不能多线程
# search.fit(data_train,labels,groups=groups)

# results_df = pd.DataFrame(search.cv_results_)
# results_df = results_df.sort_values(by=["rank_test_score"])
# results_df = results_df.set_index(
#     results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))
# ).rename_axis("abbr.")
# results_df['test_score']=["%0.3f (+/-%0.03f)" % (mean, std) for mean,std in zip(results_df['mean_test_score'],results_df['std_test_score'])]
# results_df['train_score']=["%0.3f (+/-%0.03f)" % (mean, std) for mean,std in zip(results_df['mean_train_score'],results_df['std_train_score'])]
# logger.info(tabulate(results_df[["test_score","train_score","params"]],headers='keys',tablefmt="github"))


###############################################################################
# Display MDM centroid

# mdm = MDM()
# mdm.fit(cov_data_train, labels)

# fig, axes = plt.subplots(1, 2, figsize=[8, 4])
# ch_names = [ch.replace('.', '') for ch in epochs.ch_names]

# df = pd.DataFrame(data=mdm.covmeans_[0], index=ch_names, columns=ch_names)
# g = sns.heatmap(
#     df, ax=axes[0], square=True, cbar=False, xticklabels=2, yticklabels=2)
# g.set_title('Mean covariance - hands')

# df = pd.DataFrame(data=mdm.covmeans_[1], index=ch_names, columns=ch_names)
# g = sns.heatmap(
#     df, ax=axes[1], square=True, cbar=False, xticklabels=2, yticklabels=2)
# plt.xticks(rotation='vertical')
# plt.yticks(rotation='horizontal')
# g.set_title('Mean covariance - feets')

# # dirty fix
# plt.sca(axes[0])
# plt.xticks(rotation='vertical')
# plt.yticks(rotation='horizontal')
# plt.show()
