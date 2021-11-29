import os
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
from helper.utils import logger,compress_datatype

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
    num_of_tries: int = field(default=4,metadata={'help':'number of tries to find good features'})
    seed: int = field(default=1058,metadata={'help':'seed for shuffling data into train test split'})
    task: str = field(default='greedy_search')
    instant_remove: bool = field(default=False,metadata={'help':'if item is removed instantly during search'})
    trail_feat_len: int = field(default=-1,metadata={'help':'fast way to test code feasibility'})

@register_task('greedy_search',dataclass=GreedySearchConfig)
class GreedySearch(BaseTask):
    def __init__(self,
                 args,
                 train,
                 test,
                 model,
                 arch,
                 **kwargs):
        self.args = args
        self.train = train
        self.test = test
        self.model = model
        self.arch = arch
        
    def save_data(self,X_tr,X_te,good_feature_list,i):
        X_train = X_tr[:,good_feature_list]
        X_test = X_te[:,good_feature_list]
        suffix = str(i + 1)
        
        X_train = compress_datatype(X_train)
        X_test = compress_datatype(X_test)
        
        str_suffix = 'remove' if self.args.instant_remove else ''
        with open(f'interim_data_store/greedy{suffix}{str_suffix}.pkl','wb') as f:
            pickle.dump((X_train,X_test,good_feature_list),f)
            logger.info(f"successfully saved greedy{suffix}{str_suffix}")
    
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
        
        if os.path.exists('greedy_helper.pkl'):
            with open('greedy_helper.pkl','rb') as f:
                greedy_index = pickle.load(f)
                gr1_ind,gr2_ind = greedy_index
            self.save_data(X_tr,X_te,gr1_ind,0)
            self.save_data(X_tr,X_te,gr2_ind,1)
            return X_tr
        
        for i in range(self.args.num_of_tries):
            score_list = []
            good_feature_list = []
            first_cols = len(Xts) + 1 if self.args.trail_feat_len == -1 else self.args.trail_feat_len
            if not self.args.instant_remove:
                for f in range(len(Xts[:first_cols])):
                    good_feature_list.append(f)
                    good_features = good_feature_list.copy()
                    X = sparse.hstack([Xts[j] for j in good_feature_list])
                    score = cv_search(X,y,self.model,self.args.cv_splits,self.args.seed*(i+1))
                    logger.info('current num of features:{} with auc_score:{:.5f}'.format(len(good_feature_list),score))
                    score_list.append((score,good_features))
                good_feature_list = sorted(score_list,key=lambda x:x[0])[-1][1]
                score_list.append(sorted(score_list,key=lambda x:x[0])[-1][0])
                self.save_data(X_tr,X_te,good_feature_list,i) 
            else:
                best_score = 0
                for f in range(len(Xts[:first_cols])):
                    good_feature_list.append(f)
                    X = sparse.hstack([Xts[j] for j in good_feature_list])
                    score = cv_search(X,y,self.model,self.args.cv_splits,self.args.seed*(i+1))
                    if score <= best_score:
                        good_feature_list = good_feature_list[:-1]
                    else:
                        best_score = score
                    logger.info(f'current best feature:{good_feature_list} with score:{score:.5f}')
                self.save_data(X_tr,X_te,good_feature_list,i)
                
            logger.info(f'arch:{self.arch} round:{i} best features:{good_feature_list}')     
                
        return Xts
        
    
#%%


    
