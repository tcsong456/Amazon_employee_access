import os
import numpy as np
import pickle
from tasks.base_task import BaseTask
from dataclass.configs import BaseDataClass
from tasks import register_task
from helper.utils import logger,get_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import sparse

class StackedClassifier(BaseTask):
    def __init__(self,
                 models,
                 arch,
                 stacking_inner_splits=4,
                 model_selection=False,
                 stack=True,
                 use_cached_models=False,
                 show_step=True):
        self.model_selection = model_selection
        self.stack = stack
        self.use_cached_models = use_cached_models
        self.models = models
        self.stacking_inner_splits = stacking_inner_splits
        self.show_step = show_step
        self.arch = arch
        
        stacking_path = 'interim_data_store/stacking'
        if not os.path.exists(stacking_path):
            os.makedirs(stacking_path)
        self.stacking_path = stacking_path
        self.mode_preds_path = os.path.join(self.stacking_path,f'{self.arch}_model_preds.pkl')
    
    def _get_model_preds(self,X_train,X_predict,y_train,model): 
        if os.path.exists(self.mode_preds_path):
            model_preds = pickle.load(self.mode_preds_path)
            return model_preds
        
        model.fit(X_train,X_predict)
        model_preds = model.predict_proba(X_predict)[:,1]
        with open(self.mode_preds_path,'wb') as f:
            pickle.dump(model_preds,f)
        return model_preds
    
    def _get_model_cv_preds(self,model,X_train,y_train):
        skf = StratifiedKFold(n_splits=self.stacking_inner_splits,random_state=1587,shuffle=True)
        num_rows = X_train.shape[0]
        for ind_tr,ind_te in skf.split(range(num_rows),y_train):
            if sparse.issparse(X_train):
                
    
    def fit_predict(self,y,train,predict):
        y_train = y[train]
        
        for model,feature_set in self.models:
            X_train,X_predict = get_dataset(feature_set,train,predict)
            
            model_preds = self._get_model_preds(X_train,X_predict,y_train,model)

#%%
from sklearn.linear_model import LogisticRegression
skf = StratifiedKFold(n_splits=4,random_state=2398,shuffle=True)
model = LogisticRegression()
with open(r'interim_data_store/model_features/lr_gr0.pkl.pkl','rb') as f:
    d,_ = pickle.load(f)

for ind_tr,ind_te in skf.split(range(d.shape[0]),y):
    print(ind_tr,ind_te)
    break
#%%
from helper.utils import get_dataset
models = ['lr-lr_gr0']
model_mapping = {'lr':LogisticRegression()}
for m in models:
    model,ds = m.split('-')
    model = model_mapping[model]
    X_train,X_predict = get_dataset(ds,ind_tr,ind_te)
    y_tr,y_te = y[ind_tr],y[ind_te]
    model.fit(X_train,y_tr)
    pred = model.predict_proba(X_predict)[:,1]
    
    
#%%
X_predict