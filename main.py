import warnings
warnings.filterwarnings(action='ignore')
import tasks
import os
import pickle
import models
import numpy as np
from helper.utils import load_data,logger
from sklearn.model_selection import train_test_split
from dataclass.parser_builder import convert_namespace_to_omegaconf,get_parser,parse_args_and_arch
from argparse import Namespace

def create_args(arch):
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture',default=arch)
    args = parser.parse_args()
    return args

def main():
    parser = get_parser()
    args,_ = parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)

    if cfg.setup.use_numpy:
        train,test = load_data('data',convert_to_numpy=True)
    else:
        train,test = load_data('data')
    
    if not os.path.exists('interim_data_store'):
        os.makedirs('interim_data_store')
    
    if cfg.setup.mode == 'build_task':
        orig_dir = 'interim_data_store'
        pivot_dict = {}
        pivot_keys = ['pivottable','pivottable_test','pivottable_train']
        for file in os.listdir(orig_dir):
            if 'pivottable' in file:
                dc = pickle.load(open(os.path.join(orig_dir,file),'rb'))
                pivot_dict[file[:file.find('.pkl')]] = dc
        check_bool = [(k in pivot_keys) for k in pivot_dict]
        assert np.all(check_bool),f'pivot dictionary must contain all keys in {pivot_keys}'
    
        model = models.build_model(cfg.model)
        task = tasks.setup_task(cfg.task,train=train,test=test,model=model,arch=cfg.setup.arch,
                                **pivot_dict)
        task.create_features()
    else:
        selected_models = [
                           'lgb-lgb_baseextrbs','lgb-lgb_bsbatrlmelex','lgb-lgb_exbasic','lgb-lgb_trbsme',
                           'lgb-lgb_base_basic','rf-rf_babsme','rf-rf_base','rf-rf_trmelba',
                           'lr-lr_gr0base','lr-lr_gr1base','lr-lr_gr0tuple','lr-lr_gr1tuple_base']
        models_ds = []
        
        lr_args = Namespace(architecture='lr_normal') 
        lr_args = models.ARCH_CONFIG_REGISTRY[lr_args.architecture](lr_args)
        lr_model = models.ARCH_MODEL_REGISTRY[lr_args.architecture](lr_args)
        
        lgb_args = Namespace(architecture='lgb_normal')
        lgb_args = models.ARCH_CONFIG_REGISTRY[lgb_args.architecture](lgb_args)
        lgb_model = models.ARCH_MODEL_REGISTRY[lgb_args.architecture](lgb_args)
        
        rf_args = Namespace(architecture='rf_normal')
        rf_args = models.ARCH_CONFIG_REGISTRY[rf_args.architecture](rf_args)
        rf_model = models.ARCH_MODEL_REGISTRY[rf_args.architecture](rf_args)
        
        model_map = {'lr':lr_model,
                     'lgb':lgb_model,
                     'rf':rf_model}
        model_name_map = {'lr':'logistic_regression',
                          'lgb':'lightgbm',
                          'rf':'random_forest'}
        
        for m in selected_models:
            model,dataset = m.split('-')
            md = model_map[model]
            models_ds.append((md,dataset,model_name_map[model]))

        task = tasks.setup_task(cfg.task,models=models_ds)
        y = train[:,0]
        if cfg.setup.pattern == 'evaluate':
            for i in range(cfg.task.num_iter):
                ind_train,ind_valid = train_test_split(range(len(y)),test_size=cfg.task.valid_size,random_state=(i+1)*cfg.task.seed)
                score = task.fit_predict(y,ind_train,ind_valid)
                logger.info(f'running iter:{i} stack score:{score:.5f}')
        else:
            task.fit_predict(y)
    
if __name__ == '__main__':
    main()




#%%
