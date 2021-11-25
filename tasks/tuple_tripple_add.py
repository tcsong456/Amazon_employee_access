import numpy as np
import pickle
from dataclasses import dataclass,field
from tasks.base_task import BaseTask
from dataclass.configs import BaseDataClass
from dataclass.choices import TUPLETRIPPLE_CHOICES
from tasks import register_task
from helper.utils import compress_datatype
from sklearn.base import TransformerMixin
from scipy import sparse
from typing import Optional

@dataclass
class BuildTupleTrippleConfig(BaseDataClass):
    task: str = field(default='build_tuple_tripple_add')
    option: Optional[TUPLETRIPPLE_CHOICES] = field(default='tuple',metadata={'help':'option of effect data group to build'})

def create_tuple(data):
    new_data = []
    for i in range(data.shape[1]):
        for j in range(i+1,data.shape[1]):
            new_data.append((data[:,i] + data[:,j])[:,None])
    new_data = np.hstack(new_data)
    return new_data

def create_tripple(data):
    new_data = []
    num_cols = data.shape[1]
    for i in range(num_cols):
        for j in range(i+1,num_cols):
            for k in range(j+1,num_cols):
                d = data[:,i] + data[:,j] + data[:,k]
                new_data.append(d[:,None])
    new_data = np.hstack(new_data)
    return new_data

def sparcify(train,test):
    X = np.vstack([train,test])
    fitter = OneHotEncoder()
    fitter.fit(X)
    X_train = fitter.transform(train)
    X_test = fitter.transform(test)
    return X_train,X_test

class OneHotEncoder(TransformerMixin):    
    def fit(self,X):
        self.keymap = []
        datas = list(X.T)
        for data in datas:
            unique_v = np.unique(data)
            self.keymap.append(dict((v,i) for i,v in enumerate(unique_v)))
        return self
    
    def transform(self,X):
        sparse_data = []
        for ind,x in enumerate(list(X.T)):
            km = self.keymap[ind]
            num_cols = len(km)
            spm = sparse.lil_matrix((len(x),num_cols))
            for i,v in enumerate(x):
                spm[i,km[v]] = 1
            sparse_data.append(spm)
        output = sparse.hstack(sparse_data).tocsr()
        return output  

@register_task('build_tuple_tripple',dataclass=BuildTupleTrippleConfig)
class BuildTupleTripple(BaseTask):
    def __init__(self,
                 args,
                 train,
                 test,
                 model,
                 arch,
                 **kwargs):
        self.args = args
        self.train = train[:,1:-1]
        self.test = test[:,1:-1]
    
    def create_features(self):
        if self.args.option == 'tuple':
            feat_tr = create_tuple(self.train)
            feat_te = create_tuple(self.test)
        elif self.args.option == 'tripple':
            feat_tr = create_tripple(self.train)
            feat_te = create_tripple(self.test)
        else:
            feat_tr = self.train
            feat_te = self.test
        
        feat_tr = compress_datatype(feat_tr)
        feat_te = compress_datatype(feat_te)
        
        filename = self.args.option
        with open(f'interim_data_store/{filename}.pkl','wb') as f:
            pickle.dump((feat_tr,feat_te),f)
        
#%%

