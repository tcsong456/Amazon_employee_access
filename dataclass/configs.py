from dataclasses import dataclass,field,_MISSING_TYPE

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

@dataclass
class Setup(BaseDataClass):
    stacking: bool = field(default=False,metadata={'help':'if use stacking to produce output'})
    stacking_splits: int = field(default=5,metadata={'help':'num of splits used for stacking'})

@dataclass
class Config(BaseDataClass):
    setup = Setup()




#%%
