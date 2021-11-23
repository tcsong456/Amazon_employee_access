import logzero
import logging
import os
import pandas as pd
import numpy as np

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

#%%

