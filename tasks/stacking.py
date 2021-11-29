import os
import numpy as np
import pickle
import pandas as pd
from tasks.base_task import BaseTask
from helper.utils import logger,get_dataset,compress_datatype
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import sparse
from sklearn.linear_model import RidgeCV
from dataclasses import dataclass,field
from dataclass.configs import BaseDataClass
from tasks import register_task
import warnings
warnings.filterwarnings(action='ignore')

@dataclass
class StackingConfig(BaseDataClass):
    stack: bool = field(default=True,metadata={'help':'if use stacking to produce output'})
    stacking_splits: int = field(default=5,metadata={'help':'num of splits used for stacking'})
    use_cached_preds: bool = field(default=False,metadata={'help':'load preds from cache'})
    show_step: bool = field(default=True,metadata={'help':'to show detail steps during stacking'})
    gen_linespace: int = field(default=200,metadata={'help':'linespace upper limit for stacking generalizer'})
    gen_cv: int = field(default=100,metadata={'help':'num of cv for stacking generalizer'})
    num_iter: int = field(default=3,metadata={'help':'number of iterations for running stacking'})
    valid_size: float = field(default=0.2,metadata={'help':'portion of data used for validation'})
    seed: int = field(default=7159,metadata={'help':'seed used for shuffling train test split of data'})
    save_data: bool = field(default=False,metadata={'help':'if stacking result will be saved'})
    task: str = field(default='stacking')

@register_task('stacking',dataclass=StackingConfig)
class StackedClassifier(BaseTask):
    def __init__(self,
                 args,
                 models,
                 ):
        self.args = args
        self.models = models
        
        stacking_path = 'interim_data_store/stacking'
        if not os.path.exists(stacking_path):
            os.makedirs(stacking_path)
        self.stacking_path = stacking_path
        
        self.generalizer = RidgeCV(alphas=np.linspace(0, self.args.gen_linespace), cv=self.args.gen_cv)
    
    def _get_model_cv_preds(self,model,X_train,X_predict,y_train):
        skf = StratifiedKFold(n_splits=self.args.stacking_splits,random_state=1587,shuffle=True)
        num_rows,num_cols = X_train.shape
        stack_preds = np.zeros(num_rows)
        model_preds = np.zeros(X_predict.shape[0])
        
        for i,(ind_tr,ind_te) in enumerate(skf.split(range(num_rows),y_train)):
            logger.info(f'running stacking model cv: {i}')
            if sparse.issparse(X_train):
                xtr = sparse.hstack([X_train,X_train]).tocsr()[ind_tr,:num_cols]
                xte = sparse.hstack([X_train,X_train]).tocsr()[ind_te,:num_cols]
            else:
                xtr = X_train[ind_tr]
                xte = X_train[ind_te]
            
            ytr = y_train[ind_tr]
            model.fit(xtr,ytr)
            stack_pred = model.predict(xte)
            stack_preds[ind_te] = stack_pred

            model_pred = model.predict(X_predict)
            model_preds += model_pred
            
        model_preds /= self.args.stacking_splits
        
        return stack_preds,model_preds
    
    def _combine_preds(self,X_stack,X_predict,y_train,y_te=None):
        mean_preds = np.mean(X_predict,axis=1)
        mean_score = roc_auc_score(y_te,mean_preds) if y_te is not None else 0
        stack_preds = None
        
        if self.args.stack:
            self.generalizer.fit(X_stack,y_train)
            stack_preds = self.generalizer.predict(X_predict)
            stack_score = roc_auc_score(y_te,stack_preds) if y_te is not None else 0
        
        return mean_score,stack_score,mean_preds,stack_preds
    
    def fit_predict(self,y,train=None,predict=None):
        y_train = y[train] if train is not None else y
        if train is not None and predict is None:
            predict = [i for i in range(len(y)) if i not in train]
        y_te = y[predict] if predict is not None else None
        
        stack_train,stack_predict = [],[]
        best_score = 0
        keep_features,featuer_ind = [],[]
        for i,(model,feature_set,model_name) in enumerate(self.models):
            logger.info(f'predicting feature:{feature_set} with {model_name}')
            
            X_train,X_predict = get_dataset(feature_set,model_name,train,predict)

            if self.args.stack:
                stack_preds,model_preds = self._get_model_cv_preds(model,X_train,X_predict,y_train)
                stack_train.append(stack_preds)
                stack_predict.append(model_preds)

            if self.args.show_step and train is not None:
                mean_score,stack_score,mean_preds,stack_preds = self._combine_preds(np.array(stack_train).T,
                                                                                    np.array(stack_predict).T,
                                                                                    y_train,
                                                                                    y_te)
                                        
                model_auc = roc_auc_score(y_te,stack_predict[-1])
                mean_auc = roc_auc_score(y_te,mean_preds)
                stack_auc = roc_auc_score(y_te,stack_preds) if self.args.stack else 0
                
                if stack_auc > best_score:
                    best_score = stack_auc
                    keep_features.append(feature_set)
                    featuer_ind.append(i)
                else:
                    stack_train = stack_train[:-1]
                    stack_predict = stack_predict[:-1]
                
                logger.info('model auc:%.5f,mean auc:%.5f,stack auc:%.5f' % (
                            model_auc,mean_auc,stack_auc))
        
        stack_train = np.array(stack_train).T
        stack_valid = np.array(stack_predict).T
        stack_mean,stack_score,mean_preds,stack_preds = self._combine_preds(stack_train,
                                                                            stack_valid,
                                                                            y_train,
                                                                            y_te)
        logger.info(f'final best score:{stack_score:.5f}')
        
        if train is None:
            stack_preds = pd.DataFrame(stack_preds,columns=['Action'])
            test = pd.read_csv('data/test.csv')
            stack_preds['id'] = test.id
            stack_preds.to_csv(r'submission.csv',index=False)
        
        if self.args.save_data:
            stack_tr = compress_datatype(stack_train)
            stack_valid = compress_datatype(stack_valid)
            
            with open(os.path.join(self.stacking_path,'stack_train.pkl'),'wb') as f:
                pickle.dump(stack_tr,f)
            with open(os.path.join(self.stacking_path,'stack_valid.pkl'),'wb') as f:
                pickle.dump(stack_valid,f)
            with open(os.path.join(self.stacking_path,'best_features.pkl'),'wb') as f:
                pickle.dump(keep_features,f)
            
        if self.args.stack:
            score = stack_score
        else:
            score = mean_score
        
        return score
        
#%%
