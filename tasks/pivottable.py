import os
import pickle
import numpy as np
from tasks.base_task import BaseTask
from dataclass.configs import BaseDataClass
from dataclass.choices import PIVOTTABLE_CHOICES
from dataclasses import dataclass,field
from helper.utils import logger
from typing import Optional
from tasks import register_task

@dataclass
class PivotTableConfig(BaseDataClass):
    task: str = field(default='pivottable')
    use_cache: bool = field(default=False,metadata={'help':'whether to extract data from cache'})
    use_split: Optional[PIVOTTABLE_CHOICES] = field(default='all',metadata={'help':'data split used for building pivot table'})

@register_task('build_pivottable',dataclass=PivotTableConfig)
class PivotTable(BaseTask):
    def __init__(self,
                 args,
                 train,
                 test,
                 model,
                 arch):
        self.args = args
        self.train = train
        self.test = test
        if not os.path.exists('interim_data_store'):
            os.mkdir('interim_data_store')
        
        self.COLNAMES = ["resource", "manager", "role1", "role2", "department",
                         "title", "family_desc", "family"]
    
    def create_features(self):
        if  self.args.use_split == 'all':
            X = np.vstack([self.train[:,1:-1],self.test[:,1:-1]])
            filename = 'pivottable'
        elif self.args.use_split == 'train':
            X = self.train[:,1:-1]
            filename = 'pivottable_train'
        elif self.args.use_split == 'test':
            X = self.test[:,1:-1]
            filename = 'pivottable_test'
        
        dictionary = []
        for j in range(len(self.COLNAMES)):
            dictionary.append({'total':0})
        
        path = f'interim_data_store/{filename}.pkl'
        if self.args.use_cache:
            with open(path,'rb') as f:
                dictionary = pickle.load(f)
                logger.info(f'loading data from {path}')
        else:
            logger.info('no cache data found,creating pivot data')
            for row in X:
                for j in range(len(self.COLNAMES)):
                    dictionary[j]['total'] += 1
                    if row[j] not in dictionary[j]:
                        dictionary[j][row[j]] = {'total':1}
                        for k in range(len(self.COLNAMES)):
                            key = self.COLNAMES[k]
                            dictionary[j][row[j]][key] = {row[j]:1}
                    else:
                        dictionary[j][row[j]]['total'] += 1
                        for i,key in enumerate(self.COLNAMES):
                            if row[i] not in dictionary[j][row[j]][key]:
                                dictionary[j][row[j]][key][row[i]] = 1
                            else:
                                dictionary[j][row[j]][key][row[i]] += 1
            
            with open(path,'wb') as f:
                pickle.dump(dictionary,f)
        
        return dictionary

#%%
              