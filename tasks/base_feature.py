import os
import pickle
import numpy as np
import pandas as pd
from helper.utils import compress_datatype,logger
from dataclass.configs import BaseDataClass
from dataclasses import dataclass,field
from tasks import register_task
from tasks.base_task import BaseTask

@dataclass
class BaseFeatureConfig(BaseDataClass):
    task: str = field(default='base_feature')
    log: bool = field(default=False,metadata={'help':'if result is log processed'})

@register_task('base_feature',dataclass=BaseFeatureConfig)
class BaseFeature(BaseTask):
    def __init__(self,
                 args,
                 train,
                 test,
                 model,
                 arch,
                 **kwargs):
        self.train = train
        self.test = test
        self.X_all = pd.concat([self.train.iloc[:,1:-1],self.test.iloc[:,1:-1]])
        self.model = model
        self.args = args
        self.arch = arch
    
    def save_data(self,Xt,Xte):
        f_path = 'interim_data_store'
        
        Xt = compress_datatype(Xt)
        Xte = compress_datatype(Xte)
        with open(os.path.join(f_path,'bsfeats.pkl'),'wb') as f:
            pickle.dump((Xt,Xte),f)
            logger.info(f'data saved as bsfeats at {f_path}')
    
    def role_group_cnt(self,X):
        for c in X.columns:
            X[c+'_cnt'] = 0
            group = X.groupby([c]).size()
            X[c+'_cnt'] = X[c].map(group)
        
            if self.args.log:
                X[c+'_cnt'] = X[c+'_cnt'].apply(np.log)
            X = X.reset_index(drop=True)
        
        return X
    
    def role_resource_cnt(self,X):
        def apply_mapping(row,dc_s,dc_f):
            key_1 = (row[0],row[1])
            h1 = dc_s[key_1]
            key_2 = row[0]
            h2 = dc_f[key_2]
            return np.round(h1 / h2,5)
        
        d = []
        apply_cols = self.train.columns[2:-1]
        for col in apply_cols:
            map_son = X.groupby([col,'RESOURCE']).size()
            map_father = X.groupby([col]).size()
            
            p = X[[col,'RESOURCE']]
            output = p.apply(lambda row:apply_mapping(row,map_son,map_father),axis=1)
            d.append(output.to_frame(col+'_resource_cnt'))
        d = pd.concat(d,axis=1).reset_index(drop=True)
        return d
    
    def mgr_unique_cnt(self,X):
        grp = X.groupby(['MGR_ID'])['RESOURCE'].unique()
        grp_dc = grp.map(len)
        x = X['MGR_ID'].map(grp_dc).to_frame('mgr_rs_unique_cnt')
        x = x.reset_index(drop=True)
        return x
    
    def create_features(self):
        self.X_all['ROLE_ROLLUP'] = self.X_all['ROLE_ROLLUP_1'] + self.X_all['ROLE_ROLLUP_2'] * 1000000
        self.X_all['FAMILY_TITLE'] = self.X_all['ROLE_FAMILY'] * 1000000 + self.X_all['ROLE_TITLE']

        X_cnt = self.role_group_cnt(self.X_all)
        X_rs = self.role_resource_cnt(self.X_all)
        x_mgr_rs_unique = self.mgr_unique_cnt(self.X_all)
        X = pd.concat([X_cnt,X_rs,x_mgr_rs_unique],axis=1)
        X = X.iloc[:,8:]
        
        train_rows = self.train.shape[0]
        Xt = X.iloc[:train_rows]
        Xte = X.iloc[train_rows:]

        self.save_data(Xt,Xte)
        
        

#%%
