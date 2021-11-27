from models import register_model,register_model_architecture
import lightgbm as lgb

@register_model('lightgbm')
class LGBMRegressor:
    def __init__(self,args):
        self.param  = {
                        'num_leaves':args.num_leaves,
                        'min_data_in_leaf':args.min_data_in_leaf,
                        'feature_fraction':args.feature_fraction,
                        'bagging_fraction':args.bagging_fraction,
                        'bagging_freq':args.bagging_freq,
                        'learning_rate':args.learning_rate,
                        'metric':args.metirc,
                        'objective':args.objective,
                        'num_boost_round':args.num_boost_round,
                        'verbose':args.verbose
                            }
    @staticmethod
    def add_args(parser):
        parser.add_argument('--num_leaves',)
        parser.add_argument('--min_data_in_leaf')
        parser.add_argument('--max_samples')
        parser.add_argument('--feature_fraction')
        parser.add_argument('--bagging_fraction')
        parser.add_argument('--bagging_freq')
        parser.add_argument('--learning_rate')
        parser.add_argument('--metirc')
        parser.add_argument('--objective')
    
    def fit(self,X_tr,y_tr):
        dtrain = lgb.Dataset(X_tr,label=y_tr)
        self.bst = lgb.train(self.param,dtrain)
        return self
    
    def predict(self,X_te):
        pred = self.bst.predict(X_te,self.bst.best_iteration or self.param['num_boost_round'])
        return pred

@register_model_architecture('lightgbm','lgb_normal')
def lgb_run(args):
    args.num_leaves = getattr(args,'num_leaves',100)
    args.min_data_in_leaf = getattr(args,'min_data_in_leaf',150)
    args.feature_fraction = getattr(args,'feature_fraction',0.1)
    args.bagging_fraction = getattr(args,'bagging_fraction',0.8)
    args.bagging_freq = getattr(args,'bagging_freq',1)
    args.learning_rate = getattr(args,'learning_rate',0.02)
    args.metirc = getattr(args,'metirc','l2_root')
    args.objective = getattr(args,'objective','regression')
    args.num_boost_round = getattr(args,'num_boost_round',1000)
    args.verbose = getattr(args,'verbose',-1)
    return args


#%%