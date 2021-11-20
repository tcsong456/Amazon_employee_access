from models import register_model,register_model_architecture
from sklearn.linear_model import LogisticRegression

@register_model('logistic_regression')
class LR:
    def __init__(self,args):
        model = LogisticRegression(penalty=args.penalty,
                                   C=args.c,
                                   max_iter=args.iter)
        self.model = model
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--penalty',
                            help='Specify the norm of the penalty')
        parser.add_argument('--c',
                            help='Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values \
                            specify stronger regularization')
        parser.add_argument('--max_iter',
                            help='Maximum number of iterations taken for the solvers to converge')
        parser.add_argument('--multi_class',
                            help="If the option chosen is ‘ovr’, then a binary problem is fit for each label. For ‘multinomial’ the loss minimised \
                            is the multinomial loss fit across the entire probability distribution, even when the data is binary. ‘multinomial’ is \
                            unavailable when solver=’liblinear’. ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise \
                            selects ‘multinomial’")
    
    def fit(self,X,y):
        self.model.fit(X,y)
        return self
    
    def predict(self,X):
        return self.model.predict_proba(X)[:,1]

@register_model_architecture('logistic_regression','logistic_regression')
def greedylr(args):
    args.penalty = getattr(args,'penalty','l1')
    args.c = getattr(args,'c',1.0)
    args.max_iter = getattr(args,'max_iter',100)
    args.multi_class = getattr(args,'multi_class','auto')
        

#%%
