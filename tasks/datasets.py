import os
import numpy as np
import pickle
from dataclasses import dataclass,field
from tasks.base_task import BaseTask
from dataclass.configs import BaseDataClass
from tasks import register_task
from sklearn.preprocessing import StandardScaler
from helper.utils import save_data,logger
from sklearn.base import TransformerMixin
from scipy import sparse

@dataclass
class BuildDatasetConfig(BaseDataClass):
    task: str = field(default='build_dataset')

def process(data,log_transform=False,normalize=False,create_divs=False):
    if not isinstance(data,np.ndarray):
        data = np.array(data)
        
    d = list(data.T)
    features = [np.array(f) for f in d]
    
    if create_divs:
        b = features[0]
        features.extend([a / (b + 1) for a in features])
        features.extend([a * b for a in features])
    
    if log_transform:
        features.extend([a**2 for a in features])
        features.extend([np.log(a + 1) for a in features])
    
    features = np.array(features).T
    
    if normalize:
        fitter = StandardScaler()
        fitter.fit(features)
        features = fitter.transform(features)
    
    return features

def check_data(files):
    dirs = os.listdir('interim_data_store')
    file_list = [f[:f.find('.pkl')] for f in dirs]
    bools = [(f in file_list) for f in files]
    assert np.all(bools)

def check_var(dtr,dte):
    keep = []
    num_rows = dtr.shape[0]
    d = np.vstack([dtr,dte])
    
    for col in range(d.shape[1]):
        var = d[:,col].var()
        if var > 0:
            keep.append(col)
    d = d[:,keep]
    
    dtr = d[:num_rows]
    dte = d[num_rows:]
    return dtr,dte

def sparcify(train,test):
    X = np.vstack([train,test])
    fitter = OneHotEncoder()
    fitter.fit(X)
    X_train = fitter.transform(train).astype(np.float16)
    X_test = fitter.transform(test).astype(np.float16)
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

@register_task('build_dataset',dataclass=BuildDatasetConfig)
class BuildDataset(BaseTask):
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
        self.arch = arch
        self.mother_path = 'interim_data_sotre/model_features'
        
        self.DATASETS = ['greedy1remove','greedy2remove','greedy3remove','greedy4remove']
    
    def load_data(self,ds):
        with open(f'interim_data_store/{ds}.pkl','rb') as f:
            data = pickle.load(f)
            if len(data) == 2:
                X_tr,X_te = data
            elif len(data) == 3:
                X_tr,X_te,_ = data
        
        if not isinstance(X_tr,np.ndarray):
            X_tr = np.array(X_tr)
            X_te = np.array(X_te)
        
        return X_tr,X_te
    
    def create_features(self):
        check_data(self.DATASETS)
        
        bsfeats_tr,bsfeats_te = self.load_data('bsfeats')
        basefeats_tr,basefeats_te = self.load_data('base_data')
        extrafeats_tr,extrafeats_te = self.load_data('extra_data')
        metafeats_tr,metafeats_te = self.load_data('meta_data')
        treefeats_tr,treefeats_te = self.load_data('tree_data')
        basic_tr,basic_te = self.load_data('basic')
        tuple_tr,tuple_te = self.load_data('tuple')
        tripple_tr,tripple_te = self.load_data('tripple')

        X_tup,X_tup_te = sparcify(tuple_tr,tuple_te)
        save_data(X_tup,X_tup_te,'lr_tuple')
        
        X_trp,X_trp_te = sparcify(tripple_tr,tripple_te)
        save_data(X_trp,X_trp_te,'lr_tripple')
        
        save_data(X_tup,X_tup_te,'lr_tuple_tripple',X_trp,X_trp_te)
        
        tree_cd_tr = process(treefeats_tr,create_divs=True)
        tree_cd_te = process(treefeats_te,create_divs=True)
        tree_cd_tr,tree_cd_te = check_var(tree_cd_tr,tree_cd_te)
        save_data(tree_cd_tr,tree_cd_te,'tree_div')
        
        tree_log_tr = process(treefeats_tr,log_transform=True)
        tree_log_te = process(treefeats_te,log_transform=True)
        tree_log_tr,tree_log_te = check_var(tree_log_tr,tree_log_te)
        save_data(tree_log_tr,tree_log_te,'tree_log')
        
        extra_log_tr = process(extrafeats_tr,log_transform=True)
        extra_log_te = process(extrafeats_te,log_transform=True)
        extra_log_tr,extra_log_te = check_var(extra_log_tr,extra_log_te)
        save_data(extra_log_tr,extra_log_te,'extra_log')
        
        meta_log_tr = process(metafeats_tr,log_transform=True)
        meta_log_te = process(metafeats_te,log_transform=True)
        meta_log_tr,meta_log_te = check_var(meta_log_tr,meta_log_te)
        save_data(meta_log_tr,meta_log_te,'meta_log')
        
        for i,ds in enumerate(self.DATASETS):
            logger.info(f'creating features with {ds} dataset used')
            X,X_te = self.load_data(ds)
            
            if 'greedy' in ds:
                X,X_te = sparcify(X,X_te)
                save_data(X,X_te,f'lr_gr{i}')
                save_data(X,X_te,f'lr_gr{i}tuple',X_tup,X_tup_te)
                save_data(X,X_te,f'lr_gr{i}tripple',X_trp,X_trp_te)
            
                

#%%
