from models import build_model
from dataclass.utils_dataclass import gen_parser_from_dataclass

class BaseTask:
    @classmethod
    def setup_task(cls,cfg,**kwargs):
        return cls(cfg,**kwargs)
    
    @classmethod
    def add_args(cls,parser):
        dc = getattr(cls,'__dataclass')
        if dc is not None:
            gen_parser_from_dataclass(parser,dc())
    
    def build_model(self,cfg):
        model = build_model(cfg)
        return model




#%%