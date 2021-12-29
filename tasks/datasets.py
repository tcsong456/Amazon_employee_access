import os
import numpy as np
import pickle
from dataclasses import dataclass,field
from tasks.base_task import BaseTask
from dataclass.configs import BaseDataClass
from tasks import register_task
from sklearn.preprocessing import StandardScaler
from helper.utils import save_data,logger,check_var
from sklearn.base import TransformerMixin
from scipy import sparse

@dataclass
class BuildDatasetConfig(BaseDataClass):
    task: str = field(default='build_dataset')

def process(data,log_transform=False,normalize=False,create_divs=False):
    if not isinstance(data,np.ndarray):
        data = np.array(data)
        
    d = list(data.T)
    feats = [np.array(f) for f in d]
    features = []
    features.extend(feats)
    
    if create_divs:
        b = feats[0]
        features.extend([a / (b + 1) for a in feats])
        features.extend([a * b for a in feats])
    
    if log_transform:
        features.extend([a**2 for a in feats])
        features.extend([np.log(a + 1) for a in feats])
    
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

def sparcify(train,test):
    X = np.vstack([train,test])
    fitter = OneHotEncoder()
    fitter.fit(X)
    X_train = fitter.transform(train)
    X_test = fitter.transform(test)
    return X_train,X_test

def concat(*args):
    args = args[0]
    data = [d for d in args]
    data = np.hstack(data)
    return data
    
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
        self.test = test[:,:-1]
        self.arch = arch
        self.mother_path = 'interim_data_store/model_features'
        self.lr_path = os.path.join(self.mother_path,'logistic_regression')
        if not os.path.exists(self.lr_path):
            os.makedirs(self.lr_path)
        
        self.DATASETS = ['greedy1remove','greedy2remove']
        
    
    def load_data(self,ds,suffix=None,convert_to_numpy=True):
        suffix = suffix if suffix is not None else ''
        with open(f'interim_data_store/{suffix}{ds}.pkl','rb') as f:
            data = pickle.load(f)
            if len(data) == 2:
                X_tr,X_te = data
            elif len(data) == 3:
                X_tr,X_te,_ = data
        
        if convert_to_numpy:
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
        save_data(X_tup,X_tup_te,'lr_tuple','logistic_regression')
        
        X_trp,X_trp_te = sparcify(tripple_tr,tripple_te)
        save_data(X_trp,X_trp_te,'lr_tripple','logistic_regression')
        
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
            
            X,X_te = sparcify(X,X_te)
            save_data(X,X_te,f'lr_gr{i}','logistic_regression')
            save_data(X,X_te,f'lr_gr{i}tuple','logistic_regression',X_tup,X_tup_te)
            save_data(X,X_te,f'lr_gr{i}tripple','logistic_regression',X_trp,X_trp_te)
            
            X,X_te = self.load_data(f'lr_gr{i}','model_features/logistic_regression/',convert_to_numpy=False)
            save_data(X,X_te,f'lr_gr{i}base','logistic_regression',basefeats_tr,basefeats_te)
            
            X,X_te = self.load_data(f'lr_gr{i}tuple','model_features/logistic_regression/',convert_to_numpy=False)
            save_data(X,X_te,f'lr_gr{i}tripple_base','logistic_regression',basefeats_tr,basefeats_te)
            
            X,X_te = self.load_data(f'lr_gr{i}tripple','model_features/logistic_regression/',convert_to_numpy=False)
            save_data(X,X_te,f'lr_gr{i}tuple_base','logistic_regression',basefeats_tr,basefeats_te)
        
        base_basic_tr = concat([basefeats_tr,basic_tr])
        base_basic_te = concat([basefeats_te,basic_te])
        save_data(base_basic_tr,base_basic_te,'lgb_base_basic','lightgbm')
        
        baseextrbs_tr = concat([basefeats_tr,treefeats_tr,extrafeats_tr,basic_tr])
        baseextrbs_te = concat([basefeats_te,treefeats_te,extrafeats_te,basic_te])
        save_data(baseextrbs_tr,baseextrbs_te,'lgb_baseextrbs','lightgbm')
        
        bsbatrlmelex_tr = concat([bsfeats_tr,basic_tr,tree_log_tr,meta_log_tr,extrafeats_tr])
        bsbatrlmelex_te = concat([bsfeats_te,basic_te,tree_log_te,meta_log_te,extrafeats_te])
        save_data(bsbatrlmelex_tr,bsbatrlmelex_te,'lgb_bsbatrlmelex','lightgbm')
        
        exbasic_tr = concat([extrafeats_tr,basic_tr])
        exbasic_te = concat([extrafeats_te,basic_te])
        save_data(exbasic_tr,exbasic_te,'lgb_exbasic','lightgbm')
        
        trbsme_tr = concat([treefeats_tr,basic_tr,metafeats_tr])
        trbsme_te = concat([treefeats_te,basic_te,metafeats_te])
        save_data(trbsme_tr,trbsme_te,'lgb_trbsme','lightgbm')
        
        base_tr = concat([basefeats_tr])
        base_te = concat([basefeats_te])
        save_data(base_tr,base_te,'rf_base','random_forest')
        
        babsme_tr = concat([basefeats_tr,basic_tr,metafeats_tr])
        babsme_te = concat([basefeats_te,basic_te,metafeats_te])
        save_data(babsme_tr,babsme_te,'rf_babsme','random_forest')
        
        trmelba_tr = concat([treefeats_tr,meta_log_tr,basic_tr])
        trmelba_te = concat([treefeats_te,meta_log_te,basic_te])
        save_data(trmelba_tr,trmelba_te,'rf_trmelba','random_forest')

#%%
#import pickle
#with open('interim_data_store/tuple.pkl','rb') as f:
#    data = pickle.load(f)



