import numpy as np
from itertools import combinations

def comb_data(data,num):
    rows,cols = data.shape
    combs = combinations(range(cols),num)
    
    hash_data = []
    for comb in combs:
        hash_data.append([hash(tuple(v)) for v in data[:,comb]])
    return np.array(hash_data).T

def create_features(train,test):
    all_data = np.vstack([train[:,1:-1],test[:,:-1]])
    train_rows = train.shape[0]
    
    double_hash = comb_data(all_data,2)
    tripple_hash = comb_data(all_data,3)
    
    y = train[:,0]
    X = train[:,1:-1]
    X_double_tr = double_hash[:train_rows]
    X_tripple_tr = tripple_hash[:train_rows]
    X_tr = np.hstack([X,X_double_tr,X_tripple_tr])
    
    Xt = test[:,:-1]
    X_double_te = double_hash[train_rows:]
    X_tripple_te = tripple_hash[train_rows:]
    X_te = np.hstack([Xt,X_double_te,X_tripple_te])
    
    num_features = X_tr.shape[1]
    
    
    

#%%
import numpy as np
import pandas as pd
train = pd.read_csv(r'data/train.csv')
#z = comb_data(np.array(train),2)