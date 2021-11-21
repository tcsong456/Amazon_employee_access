import numpy as np
from itertools import combinations
from dataclasses import dataclass,field
from tasks import register_task
from models import build_model

def comb_data(data,num):
    rows,cols = data.shape
    combs = combinations(range(cols),num)
    
    hash_data = []
    for comb in combs:
        hash_data.append([hash(tuple(v)) for v in data[:,comb]])
    return np.array(hash_data).T

@dataclass
class GreedySearchConfig:
    cv_splits: int = field(default=10,metadata='number of splits for cv search')
    good_features: int = field(default=4,metadata='number of tries to find good features')

@register_task('greedy_search',dataclass=GreedySearchConfig)
class GreedySearch:
    def __init__(self,
                 args,
                 train,
                 test
                 ):
        self.train = train
        self.test = test
        self.model = build_model(args)
        print(self.model.get_params)
        
    @classmethod
    def setup_task(cls,cfg,**kwargs):
        return cls(cfg,**kwargs)
    
    def create_features(self,train,test):
        all_data = np.vstack([self.train[:,1:-1],self.test[:,:-1]])
        train_rows = self.train.shape[0]
        
        double_hash = comb_data(all_data,2)
        tripple_hash = comb_data(all_data,3)
        
        y = self.train[:,0]
        X = self.train[:,1:-1]
        X_double_tr = double_hash[:train_rows]
        X_tripple_tr = tripple_hash[:train_rows]
        X_tr = np.hstack([X,X_double_tr,X_tripple_tr])
        
        Xt = self.test[:,:-1]
        X_double_te = double_hash[train_rows:]
        X_tripple_te = tripple_hash[train_rows:]
        X_te = np.hstack([Xt,X_double_te,X_tripple_te])
        
        num_features = X_tr.shape[1]
        return X_tr,X_te
        
    
#%%
