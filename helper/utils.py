import logzero
import logging
import os
import pickle
import pandas as pd
import numpy as np
from scipy import sparse

def custome_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logger = logzero.setup_logger(formatter=formatter,
                                  level=logging.INFO,
                                  name=name)
    return logger

logger = custome_logger('amazon')

def compress_datatype(data):
    def compress_int(x):
        if x.min() > np.iinfo(np.int8).min and x.max() < np.iinfo(np.int8).max:
            x = x.astype(np.int8)
        elif x.min() > np.iinfo(np.int16).min and x.max() < np.iinfo(np.int16).max:
            x = x.astype(np.int16)
        elif x.min() > np.iinfo(np.int32).min and x.max() < np.iinfo(np.int32).max:
            x = x.astype(np.int32)
        else:
            x = x.astype(np.int64)
        return x 
    
    def compress_float(x):
        if x.min() > np.finfo(np.float16).min and x.max() < np.finfo(np.float16).max:
            x = x.astype(np.float16)
        elif x.min() > np.finfo(np.float32).min and x.max() < np.finfo(np.float32).max:
            x = x.astype(np.float32)
        else:
            x = x.astype(np.float64)
        return x
        
    if isinstance(data,np.ndarray):
        data = pd.DataFrame(data)
    
    for col in range(data.shape[1]):
        d = data.iloc[:,col]
        if np.issubdtype(d,np.integer):
            data.iloc[:,col] = compress_int(d)
        elif np.issubdtype(d,np.floating):
            data.iloc[:,col] = compress_float(d)
    
    return data

def load_data(path,convert_to_numpy=False):
    train = os.path.join(path,'train.csv')
    test = os.path.join(path,'test.csv')
    
    train = pd.read_csv(train)
    test = pd.read_csv(test)
    
    if convert_to_numpy:
        train = np.array(train)
        test = np.array(test)
    
    return train,test

def save_data(X,X_te,filename,feat=None,feat_te=None):
    if feat is not None:
        assert feat.shape[1] == feat_te.shape[1]
        assert feat is not None and feat_te is not None
    
        if sparse.issparse(X):
            feat = sparse.lil_matrix(feat)
            X = sparse.hstack([X,feat]).tocsr()
            feat_te = sparse.lil_matrix(feat_te)
            X_te = sparse.hstack([X_te,feat_te]).tocsr()
        else:
            X = np.hstack([X,feat])
            X_te = np.hstack([X_te,feat_te])
    
    save_path = 'interim_data_store/model_features'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    p = os.path.join(save_path,f'{filename}.pkl')
    with open(p,'wb') as f:
        pickle.dump((X,X_te),f)

def get_dataset(fset,train,cv):
    from scipy import sparse
    with open(f'interim_data_store/model_features/{fset}.pkl','rb') as f:
        X_tr,X_te = pickle.load(f)
    
    dim = X_tr.shape[1]
    if sparse.issparse(X_tr):
        X_tr = sparse.hstack([X_tr,X_tr]).tocsr()[train,:dim]
        X_te = sparse.hstack([X_te,X_te]).tocsr()[cv,:dim]
    else:
        X_tr = X_tr[train]
        X_te = X_tr[cv]
    
    return X_tr,X_te
    

#%%

