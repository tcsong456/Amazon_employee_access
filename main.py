import warnings
warnings.filterwarnings(action='ignore')
import tasks
import models
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
    
    model = models.build_model(cfg.model)
    task = tasks.setup_task(cfg.task,train=train,test=test,model=model,arch=cfg.setup.arch)
    
    task.create_features()
    
if __name__ == '__main__':
    main()




#%%
#import pickle
#with open('interim_data_store/pivottable.pkl','rb') as f:
#    d = pickle.load(f)
