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
    
    train,test = load_data('data')
    
    model = models.build_model(cfg.model)
    task = tasks.setup_task(cfg.task,train=train,test=test,model=model,arch=cfg.setup.arch)
    
    x = task.create_features()
    
if __name__ == '__main__':
    main()




#%%
#import pickle
#with open('x_tr.pkl','rb') as xtr,open('x_te.pkl','rb') as xte:
#    xtr = pickle.load(xtr)
#    xte = pickle.load(xte)
#x = xtr[:,[12]]
#z = set(list(x.T[0]))
#dict((k,i) for i,k in enumerate(z))
