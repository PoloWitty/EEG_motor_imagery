import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance

def preprocess(path):
    '''
    对原始数据进行预处理
    '''
    X = None
    Y = None
    for i in range(1,5):
        s = np.load(path+'/S'+str(i)+'.npz')
        # x = re_refer(s['X'])
        x =es_align(s['X'])
        # x = s['X']
        y =s['y']
        if X is None:
            X = x
            Y = y
        else:
            X = np.concatenate((X,x))# 把所有用户的数据拼到一块儿
            Y = np.concatenate((Y,y))
    np.savez(path+'/ea_data',X=X,y=Y)
    return X,Y

def re_refer(data):
    '''
    做每个用户分别做平均重参考 
    '''
    # TODO: 一起做平均重参考
    # Digitally re-reference to common mode average
    ref = np.mean(data,axis = 1, keepdims=True)
    data = data - ref
    return data

def es_align(data):
    '''
    对每个用户使用欧几里得对齐方法
    '''
    cov_data = Covariances().transform(data)
    mean = mean_covariance(cov_data,metric='euclid')
    V_,P = np.linalg.eig(mean)
    V = np.diag(V_**(-0.5))
    R = P@V@np.linalg.inv(P)
    data = R @ data
    return data
    
def rs_align(data):
    '''
    对每个用户在黎曼空间进行对齐
    '''
    cov_data = Covariances().transform(data)
    mean = mean_covariance(cov_data,metric='riemann')
    V_,P = np.linalg.eig(mean)
    V = np.diag(V_**(-0.5))
    R = P@V@np.linalg.inv(P)
    data = R @ data
    return data

def preprocess_test_data(path):
    '''
    对原始数据进行预处理
    '''
    X = None
    for i in range(5,9):
        s = np.load(path+'/S'+str(i)+'.npz')
        # x = re_refer(s['X'])
        # x =es_align(s['X'])
        # x = s['X']
        x =rs_align(s['X'])
        if X is None:
            X = x
        else:
            X = np.concatenate((X,x))# 把所有用户的数据拼到一块儿
    np.savez(path+'/ra_data',X=X)
    return X

if __name__=='__main__':
    import os
    path = './data/test'
    if not os.path.exists(path):
        print('data path is not correct')
        print('cur path is '+os.path.abspath('.'))
    # X,Y = preprocess(path)
    X = preprocess_test_data(path)
    print(X.shape)
    # print(Y.shape)
