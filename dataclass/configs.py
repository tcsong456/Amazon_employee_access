from typing import Any,Optional
from dataclasses import dataclass,field,_MISSING_TYPE
from dataclass.choices import SETUP_MODE_CHOICES,RNNING_PATTERN

@dataclass
class BaseDataClass:
    def _get_all_attrs(self):
        return [k for k in self.__dataclass_fields__.keys()]
    
    def _get_meta(self,attr_name,meta='help',default=None):
        return self.__dataclass_fields__[attr_name].metadata.get(meta,default)
    
    def _get_name(self,attr_name):
        return self.__dataclass_fields__[attr_name].name
    
    def _get_default(self,attr_name):
        f = self.__dataclass_fields__[attr_name]
        if not isinstance(f.default_factory,_MISSING_TYPE):
            return f.default_factory
        return f.default
    
    def _get_type(self,attr_name):
        return self.__dataclass_fields__[attr_name].type
    
    def _get_help(self,attr_name):
        return self._get_meta(attr_name,'help')
    
    def _get_argparse_const(self, attr_name):
        return self._get_meta(attr_name, "argparse_const")

    def _get_argparse_alias(self, attr_name):
        return self._get_meta(attr_name, "argparse_alias")

    def _get_choices(self, attr_name):
        return self._get_meta(attr_name, "choices")
    
    @classmethod
    def from_namespace(cls,args):
        if isinstance(args,cls):
            return args
        else:
            config = cls()
            for key in config.__dataclass_fields__.keys():
                if key.startswith('_'):
                    continue
                if hasattr(args,key):
                    setattr(config,key,getattr(args,key))
            
            return config

@dataclass
class Setup(BaseDataClass):
    arch: str = field(default='logistic_regression_gs',metadata={'help':'archtechure used to implement the task'})
    use_numpy: bool = field(default=False,metadata={'help':'wether to use numpy data or not'})
    mode: Optional[SETUP_MODE_CHOICES]  = field(default='build_task',metadata={'help':'mode to run in main function'})
    pattern: Optional[RNNING_PATTERN] = field(default='evaluate',metadata={'help':'evalutate data or submit final results'})

@dataclass
class Config(BaseDataClass):
    setup: Setup = Setup()
    model: Any = None
    task: Any = None




#%%
