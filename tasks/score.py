import os
import numpy as np
import pickle
from dataclasses import dataclass,field
from tasks.base_task import BaseTask
from dataclass.configs import BaseDataClass
from tasks import register_task
from helper.utils import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import sparse

@dataclass
class ScoreConfig(BaseDataClass):
    task: str = field(default='score')
    cv_splits: int = field(default=5,metadata={'help':'number of splits for cv search'})

def cv_search(x,y,model,n=10,seed=555):
    mean_score = 0
    skf = StratifiedKFold(n_splits=n,random_state=seed,shuffle=True)
    dimention = x.shape[1]
    for ind_tr,ind_te  in skf.split(x,y):
        try:
            x_tr = x[ind_tr]
            x_te = x[ind_te]
        except ValueError:
            if sparse.issparse(x):
                x_tr = sparse.hstack([x,x]).tocsr()[ind_tr,:dimention]
                x_te = sparse.hstack([x,x]).tocsr()[ind_te,:dimention]
                
        y_tr = y[ind_tr]
        y_te = y[ind_te]
        
        model.fit(x_tr,y_tr)
        pred = model.predict(x_te)
        score = roc_auc_score(y_te,pred)
        mean_score += score
    mean_score /= n
    return mean_score

ABBRV = {'logistic_regression_gs':'lr'}

@register_task('score',dataclass=ScoreConfig)
class Score(BaseTask):
    def __init__(self,
                 args,
                 train,
                 test,
                 arch,
                 model,
                 **kwargs):
        self.args = args
        self.model = model
        self.train = train
        self.arch = arch
        self.data_path = 'interim_data_store/model_features'
        
        assert self.arch in ABBRV
        self.key_word = ABBRV[self.arch]
    
    def create_features(self):
        logger.info(f'testing features with {self.arch}')
        
        score_result = {}
        y = self.train[:,0]
        for file in os.listdir(self.data_path):
            key = file[:file.find('.pkl')]
            if self.key_word in file:
                logger.info(f'scoring for {key}')
                path = os.path.join(self.data_path,file)
                with open(path,'rb') as f:
                    X,_ = pickle.load(f)
                score = cv_search(X,y,self.model,n=self.args.cv_splits)
                score_result[key] = np.round(score,5)
         
        with open(f'interim_data_store/stats/{self.arch}_score.pkl','wb') as f:
            pickle.dump(score_result,f)


#%%
