import math
import pickle
import numpy as np
from helper.utils import logger
from tasks import register_task
from tasks.base_task import BaseTask
from dataclass.configs import BaseDataClass
from dataclass.choices import BODY_CHOICES
from dataclasses import dataclass,field
from typing import Optional
from helper.utils import compress_datatype,check_var

@dataclass
class BodyConfig(BaseDataClass):
    option: Optional[BODY_CHOICES] = field(default='extra',metadata={'help':'the set of data to build'})
    task: str = field(default='build_body')

def check_name(dc,key):
    if key in dc:
        value = dc[key]
    else:
        raise ValueError(f'{key} must be input')
    return value

@register_task('build_body',dataclass=BodyConfig)
class BuildBody(BaseTask):
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
        self.columns = ["resource", "manager", "role1", "role2", "department",
                        "title", "family_desc", "family"]
        
        dictionary_train = check_name(kwargs,'pivottable_train')
        dictionary_test = check_name(kwargs,'pivottable_test')
        dictionary = check_name(kwargs,'pivottable')
        
        self.dictionary_train = dictionary_train
        self.dictionary_test = dictionary_test
        self.dictionary = dictionary

    def gen_feature(self,data,feature_list):
        features = []
        for row in data:
            features.append([])
            for j in range(len(self.columns)):
                for func in feature_list:
                    features[-1].append(func(self.dictionary[j][row[j]],row,j))
        features = np.array(features)
        return features
    
    def create_features(self):
        if self.args.option == 'extra':
            
            def log_result(x,row,j):
                v = x[self.columns[0]].get(row[0],0)
                if v > 0:
                    return math.log(v)
                else:
                    return 0
                
            feature_list = [ lambda x,row,j: x[self.columns[0]].get(row[0],0) if j > 0 else 0,
                             lambda x,row,j: x[self.columns[1]].get(row[1],0) if j > 1 else 0,
                             lambda x,row,j: x[self.columns[2]].get(row[2],0) if j > 2 else 0,
                             lambda x,row,j: x[self.columns[3]].get(row[3],0) if j > 3 else 0,
                             lambda x,row,j: x[self.columns[4]].get(row[4],0) if j > 4 else 0,
                             lambda x,row,j: x[self.columns[5]].get(row[5],0) if j > 5 else 0,
                             lambda x,row,j: x[self.columns[6]].get(row[6],0) if j > 6 else 0,
                             lambda x,row,j: x[self.columns[7]].get(row[7],0) if j > 7 else 0,
                             lambda x,row,j: x[self.columns[0]].get(row[0],0)**2 if j in range(7) else 0,
                             log_result]
        elif self.args.option == 'tree':
            feature_list = [lambda x,row,j: x[self.columns[0]].get(row[0],0),
                            lambda x,row,j: x[self.columns[1]].get(row[1],0),
                            lambda x,row,j: x[self.columns[2]].get(row[2],0),
                            lambda x,row,j: x[self.columns[3]].get(row[3],0),
                            lambda x,row,j: x[self.columns[4]].get(row[4],0),
                            lambda x,row,j: x[self.columns[5]].get(row[5],0),
                            lambda x,row,j: x[self.columns[6]].get(row[6],0),
                            lambda x,row,j: x[self.columns[7]].get(row[7],0)]
        elif self.args.option == 'meta':
            feature_list = [lambda x,row,j: self.dictionary_train[j].get(row[j],{}).get('total',0)]
        else:
            feature_list = [lambda x,row,j: x[self.columns[0]].get(row[0],0) / x['total'],
                            lambda x,row,j: x[self.columns[1]].get(row[1],1) / x['total'],
                            lambda x,row,j: x[self.columns[2]].get(row[2],2) / x['total'],
                            lambda x,row,j: x[self.columns[3]].get(row[3],3) / x['total'],
                            lambda x,row,j: x[self.columns[4]].get(row[4],4) / x['total'],
                            lambda x,row,j: x[self.columns[5]].get(row[5],5) / x['total'],
                            lambda x,row,j: x[self.columns[6]].get(row[6],6) / x['total'],
                            lambda x,row,j: x[self.columns[7]].get(row[7],7) / x['total']]
        
        logger.info(f'building dataset with option:{self.args.option}')
        feature_train = self.gen_feature(self.train,feature_list)
        feature_test = self.gen_feature(self.test,feature_list)
        
        feature_train = compress_datatype(feature_train)
        feature_test = compress_datatype(feature_test)
        
        feature_train,feature_test = check_var(feature_train,feature_test)
        
        filename = self.args.option
        with open(f'interim_data_store/{filename}_data.pkl','wb') as f:
            pickle.dump((feature_train,feature_test),f)
            logger.info(f'saving {filename} data at interim_data_store')
        
        return feature_train,feature_test


#%%
