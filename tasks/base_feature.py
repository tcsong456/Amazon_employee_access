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
        self.X_all = pd.concat([self.train,self.test]).iloc[:,1:]
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

@register_task('base_feature_sql',dataclass=BaseFeatureConfig)
class BaseFeatureSql(BaseFeature):  
    def __init__(self,
                 args,
                 train,
                 test,
                 model,
                 arch,
                 **kwargs):
        self.train = train
        self.test = test
        self.X_all = pd.concat([train,test]).fillna(0)
        self.model = model
        self.args = args
        self.arch = arch
        
    def role_group_cnt(self,cursor):
        X = pd.DataFrame()
        for col in self.X_all.columns[1:-1]:
            logger.info(f'counting by category in column:{col}')
            new_col = col + '_cnt'
            cursor.execute('select %s,count(%s) from amazon_all group by %s'%(col,col,col))
            d = cursor.fetchall()
            d = pd.DataFrame(d,columns=[col,new_col]).set_index(col)
            d = pd.Series(d[new_col])
            X[new_col] = self.X_all[col].map(d)
            
            if self.args.log:
                X[new_col] = X[new_col].apply(np.log)  
        X = X.reset_index(drop=True)
        return X
    
    def role_resource_cnt(self,cursor):
        def apply_mapping(row,dc_s,dc_f):
            key_1 = (row[0],row[1])
            h1 = dc_s[key_1]
            key_2 = row[1]
            h2 = dc_f[key_2]
            return np.round(h1 / h2,5)
        
        sets = []
        for col in self.X_all.columns[2:-1]:
            logger.info(f'getting count by {col},resource')
            cursor.execute('select %s,%s,count(*) from amazon_all group by %s,%s'%('resource',col,'resource',col))
            data = cursor.fetchall()
            d = pd.DataFrame(data,columns=['resource',col,'cnt']).set_index(['resource',col])
            d = pd.Series(d['cnt'])
            
            cursor.execute('select %s,count(*) from amazon_all group by %s'%(col,col))
            data = cursor.fetchall()
            ds = pd.DataFrame(data,columns=[col,'cnt']).set_index(col)
            ds = pd.Series(ds['cnt'])
            
            output = self.X_all[['RESOURCE',col]].apply(lambda row:apply_mapping(row,d,ds),axis=1)
            sets.append(output.to_frame(col+'_resource_cnt'))
        out = pd.concat(sets,axis=1).reset_index(drop=True)
        return out
    
    def mgr_unique_cnt(self,cursor):
        cursor.execute('select mgr_id,group_concat(distinct resource) from amazon_all group by mgr_id')
        data = cursor.fetchall()
        d = [(d[0],len(d[1].split(','))) for d in data]
        d = pd.DataFrame(d,columns=['mgr_id','cnt']).set_index('mgr_id')
        d = pd.Series(d['cnt'])
        x = self.X_all['MGR_ID'].map(d).to_frame('mgr_rs_unique_cnt')
        x = x.reset_index(drop=True)
        return x
    
    def create_features(self):
        from mysql_preparation import build_connection
        db = build_connection('111',
                              host='mysqldb',
                              database='kaggle')
        cursor = db.cursor()
        
        cursor.execute('select role_rollup_1 from amazon_all')
        role_rollup_1 = cursor.fetchall()
        cursor.execute('select role_rollup_2 from amazon_all')
        role_rollup_2 = cursor.fetchall()
        role_rollup = pd.DataFrame([r1[0]+r2[0]*1000000 for r1,r2 in zip(role_rollup_1,role_rollup_2)],columns=['role_rollup'])
        
        cursor.execute('select role_family from amazon_all')
        role_family = cursor.fetchall()
        cursor.execute('select role_title from amazon_all')
        role_title = cursor.fetchall()
        family_title = pd.DataFrame([r1[0]*1000000+r2[0] for r1,r2 in zip(role_family,role_title)],columns=['family_title'])
        
        cnt_stats = self.role_group_cnt(cursor)
        gp_resource_cnt = self.role_resource_cnt(cursor)
        mgr_resource = self.mgr_unique_cnt(cursor)
        
        import pickle
        with open('interim_data_store/tmp.pkl','wb') as f:
            x = [role_rollup,family_title,cnt_stats,gp_resource_cnt,mgr_resource]
            pickle.dump(x,f)
        
        X = pd.concat([role_rollup,family_title,cnt_stats,gp_resource_cnt,mgr_resource],axis=1)
        train_rows = self.train.shape[0]
        Xt = X.iloc[:train_rows]
        Xte = X.iloc[train_rows:]
        self.save_data(Xt,Xte)


#%%
