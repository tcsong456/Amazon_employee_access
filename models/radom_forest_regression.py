from models import register_model,register_model_architecture
from sklearn.ensemble import RandomForestRegressor

@register_model('random_forest')
class RFRegressor:
    def __init__(self,args):
        model = RandomForestRegressor(max_depth=args.max_depth,
                                      min_samples_leaf=args.min_samples_leaf,
                                      max_samples=args.max_samples,
                                      max_features=args.max_features,
                                      n_jobs=args.n_jobs,
                                      bootstrap=args.bootstrap)
        self.model = model
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--max_depth',
                            help='The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until \
                            all leaves contain less than min_samples_split samples')
        parser.add_argument('--min_samples_leaf',
                            help='The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered \
                            if it leaves at least min_samples_leaf training samples in each of the left and right branches')
        parser.add_argument('--max_samples',
                            help='If bootstrap is True, the number of samples to draw from X to train each base estimator')
        parser.add_argument('--max_features',
                            help="The number of features to consider when looking for the best split")
        parser.add_argument('--n_jobs',
                            help='The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees.\
                            None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details')
        parser.add_argument('--bootstrap',
                            help='Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree')
    
    @classmethod
    def build_model(cls,args):
        return cls(args)
    
    def fit(self,X,y):
        self.model.fit(X,y)
        return self
    
    def predict(self,X):
        return self.model.predict(X)
    
    def get_params(self):
        return self.model.get_params()
    
@register_model_architecture('random_forest','rf_normal')
def rf_run(args):
    args.max_depth = getattr(args,'max_depth',35)
    args.min_samples_leaf = getattr(args,'min_samples_leaf',5)
    args.max_samples = getattr(args,'max_samples',0.8)
    args.max_features = getattr(args,'max_features',0.1)
    args.n_jobs = getattr(args,'n_jobs',6)
    args.bootstrap = getattr(args,'bootstrap',False)
    return args