import re
import ast
from enum import Enum
from typing import Any,List,Tuple,Optional
from dataclasses import MISSING
from argparse import ArgumentError

def interpret_dc_type(field_type):
    if isinstance(field_type, str):
        raise RuntimeError("field should be a type")

    if field_type == Any:
        return str

    typestring = str(field_type)
    if re.match(
        r"(typing.|^)Union\[(.*), NoneType\]$", typestring
    ) or typestring.startswith("typing.Optional"):
        return field_type.__args__[0]
    return field_type

def eval_str_list(x,x_type=float):
    if x is None:
        return x
    
    if isinstance(x,str):
        if len(x) == 0:
            return []
        x = ast.literal_eval(x)
    
    try:
        return list(map(x_type,x))
    except TypeError:
        return [x_type(x)]

def gen_parser_from_dataclass(
        parser,
        dataclass_instance):
    
    def argparse_name(name):
        full_name = '--' + name.replace('_','-')
        return full_name
    
    def get_kwargs_from_dc(dataclass,k):
        kwargs = {}
        
        field_type = dataclass._get_type(k)
        inner_type = interpret_dc_type(field_type)
        field_default = dataclass._get_default(k)
        
        if isinstance(inner_type,type) and isinstance(inner_type,Enum):
            field_choices = [t.value for t in list(inner_type)]
        else:
            field_choices = None
        
        field_help = dataclass._get_help(k)
        field_const = dataclass._get_argparse_const(k)
        
        if isinstance(field_default,str):
            kwargs['default'] = field_default
        else:
            if field_default is MISSING:
                kwargs['required'] = True
            if field_choices is not None:
                kwargs['choices'] = field_choices
            if 'List' in str(inner_type) or 'Tuple' in str(inner_type):
                if 'int' in str(inner_type):
                    kwargs['type'] = lambda x: eval_str_list(x,int)
                elif 'float' in str(inner_type):
                    kwargs['type'] = lambda x: eval_str_list(x,float)
                elif 'str' in str(inner_type):
                    kwargs['type'] = lambda x: eval_str_list(x,str)
            elif (isinstance(inner_type,type) and issubclass(inner_type,Enum)) or \
                ('Enum' in str(inner_type)):
                kwargs['type'] = str
                if field_default is not MISSING:
                    if issubclass(inner_type,Enum):
                        kwargs['default'] = field_default.value
                    else:
                        kwargs['default'] = field_default
            elif inner_type is bool:
                kwargs['action'] = 'store_false' if inner_type is True else 'store_true'
                kwargs['default'] = field_default
            else:
                kwargs['type'] = inner_type
                if field_default is not MISSING:
                    kwargs['default'] = field_default
                    
        kwargs['help'] = field_help
        if field_const is not None:
            kwargs['const'] = field_const
            kwargs['nargs'] = '?'
        return kwargs
            
    for k in dataclass_instance._get_all_attrs():
        field_name = argparse_name(k)
        if field_name is None:
            continue
        
        kwargs = get_kwargs_from_dc(dataclass_instance,k)
        field_args = [field_name]
        alias = dataclass_instance._get_argparse_alias(k)
        if alias is not None:
            field_args.append(alias)
        
        try:
            parser.add_argument(*field_args,**kwargs)
        except ArgumentError:
            pass



#%%

        