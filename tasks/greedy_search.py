import numpy as np
import pickle
from itertools import combinations
from dataclasses import dataclass,field
from dataclass.configs import BaseDataClass
from tasks import register_task
from tasks.base_task import BaseTask
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from helper.utils import logger

def comb_data(data,num):
    rows,cols = data.shape
    combs = combinations(range(cols),num)
    
    hash_data = []
    for comb in combs:
        hash_data.append([hash(tuple(v)) for v in data[:,comb]])
    return np.array(hash_data).T

def convert_to_sparsematrix(data,keymap=None):
    if keymap is None:
        unique_values = set(list(data.T[0]))
        keymap = dict((k,i) for i,k in enumerate(unique_values))
        
    num_rows = data.shape[0]
    num_labels = len(unique_values)
    spmat = sparse.lil_matrix((num_rows,num_labels))
    for i,v in enumerate(data.T[0]):
        col_num = keymap[v]
        spmat[i,col_num] = 1
    spmat = spmat.tocsr()
    return spmat

def cv_search(x,y,model,n=10,seed=555):
    mean_score = 0
    for i in range(n):
        Xt,Xv,yt,yv = train_test_split(x,y,test_size=0.2,random_state=i*seed)
        model.fit(Xt,yt)
        pred = model.predict(Xv)
        score = roc_auc_score(yv,pred)
        mean_score += score
    mean_score /= n
    
    return mean_score

@dataclass
class GreedySearchConfig(BaseDataClass):
    cv_splits: int = field(default=10,metadata={'help':'number of splits for cv search'})
    min_good_features: int = field(default=4,metadata={'help':'number of tries to find good features'})
    seed: int = field(default=1058,metadata={'help':'seed for shuffling data into train test split'})
    task: str = field(default='greedy_search')

@register_task('greedy_search',dataclass=GreedySearchConfig)
class GreedySearch(BaseTask):
    def __init__(self,
                 args,
                 train,
                 test,
                 model,
                 arch
                 ):
        self.args = args
        self.train = train
        self.test = test
        self.model = model
        self.arch = arch

    def create_features(self):
        all_data = np.vstack([self.train[:,1:-1],self.test[:,1:-1]])
        train_rows = self.train.shape[0]
        
        double_hash = comb_data(all_data,2)
        tripple_hash = comb_data(all_data,3)
        
        y = self.train[:,0]
        X = self.train[:,1:-1]
        X_double_tr = double_hash[:train_rows]
        X_tripple_tr = tripple_hash[:train_rows]
        X_tr = np.hstack([X,X_double_tr,X_tripple_tr])
        
        Xt = self.test[:,1:-1]
        X_double_te = double_hash[train_rows:]
        X_tripple_te = tripple_hash[train_rows:]
        X_te = np.hstack([Xt,X_double_te,X_tripple_te])
        
        num_features = X_tr.shape[1]
        Xts = [convert_to_sparsematrix(X_tr[:,[i]]) for i in range(num_features)]
        
        i = 0
        score_list = []
        good_feature_list = []
        
        while len(score_list) < self.args.min_good_features or (score_list[-1] > score_list[-2]):
            good_features = []
            scores = []
            for f in range(len(Xts)):
                if f not in good_features:
                    good_features.append(f)
                    X = sparse.hstack([Xts[j] for j in good_features])
                    score = cv_search(X,y,self.model,self.args.cv_splits,self.args.seed*(i+1))
                    logger.info('current features:{} with auc_score:{:.5f}'.format(good_features,score))
                    scores.append((score,f))
            good_feature_list.append(sorted(scores)[-1][1])
            score_list.append(sorted(scores)[-1][0])
        
        good_feature_list = good_feature_list[:-1]
        good_feature_list = list(sorted(good_feature_list))
        logger.info(f'arch:{self.arch} round:{i} best features:{good_feature_list}')
    
        X_train = X_tr[:,good_feature_list]
        X_test = X_te[:,good_feature_list]
        suffix = str(i + 1) if i else ''
        with open(f'greedy{suffix}.pkl','wb') as f:
            pickle.dump((X_train,X_test),f)
            logger.info(f'successfully saved greed{suffix}')
        i += 1
        
        return Xts
        
    
#%%

#'{:.5f}'.format(2.12313123)