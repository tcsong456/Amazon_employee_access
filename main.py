import warnings
warnings.filterwarnings(action='ignore')
import tasks
from helper.utils import load_data
from dataclass.parser_builder import convert_namespace_to_omegaconf,get_parser,parse_args_and_arch

def main():
    parser = get_parser()
    args,_ = parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)
    
    train,test = load_data('data')
    
    task = tasks.setup_task(cfg,train=train,test=test)
    x_tr,x_te = task.create_features()
    
    import pickle
    with open('x_tr.pkl','wb') as xtr,open('x_te.pkl','wb') as xte:
        pickle.dump(x_tr,xtr)
        pickle.dump(x_te,xte)
    
if __name__ == '__main__':
    main()




#%%
