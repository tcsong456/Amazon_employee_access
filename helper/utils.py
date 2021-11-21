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

def load_data(path):
    train = os.path.join(path,'train.csv')
    test = os.path.join(path,'test.csv')
    
    train = np.array(train)
    test = np.array(test)
    
    return train,test

#%%

