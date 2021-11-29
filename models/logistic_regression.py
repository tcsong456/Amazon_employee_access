from models import register_model,register_model_architecture
from sklearn.linear_model import LogisticRegression

@register_model('logistic_regression')
class LR:
    def __init__(self,args):
        model = LogisticRegression(penalty=args.penalty,
                                   C=args.c,
                                   max_iter=args.max_iter,
                                   multi_class=args.multi_class)
        self.model = model
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--penalty',
                            help='Specify the norm of the penalty')
        parser.add_argument('--c',
                            type=float,
                            help='Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values \
                            specify stronger regularization')
        parser.add_argument('--max_iter',
                            help='Maximum number of iterations taken for the solvers to converge')
        parser.add_argument('--multi_class',
                            help="If the option chosen is ‘ovr’, then a binary problem is fit for each label. For ‘multinomial’ the loss minimised \
                            is the multinomial loss fit across the entire probability distribution, even when the data is binary. ‘multinomial’ is \
                            unavailable when solver=’liblinear’. ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise \
                            selects ‘multinomial’")
    
    @classmethod
    def build_model(cls,args):
        return cls(args)
    
    def fit(self,X,y):
        self.model.fit(X,y)
        return self
    
    def predict(self,X):
        return self.model.predict_proba(X)[:,1]
    
    def get_params(self):
        return self.model.get_params()

@register_model_architecture('logistic_regression','logistic_regression_gs')
def greedylr(args):
    args.penalty = args.penalty if args.penalty is not None else 'l2'
    args.c = args.c if args.c is not None else 4.0
    args.max_iter = args.max_iter if args.max_iter is not None else 100
    args.multi_class = args.multi_class if args.multi_class is not None else 'auto'
    return args
    
@register_model_architecture('logistic_regression','lr_normal')
def lr_run(args):
    args.penalty = args.penalty if hasattr(args,'penalty') else 'l2'
    args.c = args.c if hasattr(args,'c') else 4.0
    args.max_iter = args.max_iter if hasattr(args,'max_iter') else 100
    args.multi_class = args.multi_class if hasattr(args,'multi_class') else 'auto'      
    return args

#%%
