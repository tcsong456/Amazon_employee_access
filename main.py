import warnings
warnings.filterwarnings(action='ignore')
import tasks
import os
import pickle
import models
import numpy as np
from helper.utils import load_data
from dataclass.parser_builder import convert_namespace_to_omegaconf,get_parser,parse_args_and_arch

def main():
    parser = get_parser()
    args,_ = parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)

    if cfg.setup.use_numpy:
        train,test = load_data('data',convert_to_numpy=True)
    else:
        train,test = load_data('data')
    
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
    
if __name__ == '__main__':
    main()




#%%
