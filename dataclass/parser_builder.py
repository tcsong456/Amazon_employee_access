import sys
sys.path.append('.')

import ast
import argparse
import inspect
import os
from enum import Enum
from typing import List,Optional
from omegaconf import OmegaConf,_utils
from helper.utils import logger
from dataclass.configs import Setup,Config
from dataclasses import _MISSING_TYPE
from hydra.core.global_hydra import GlobalHydra
from hydra import compose,initialize
from argparse import Namespace
from dataclass.utils_dataclass import gen_parser_from_dataclass,interpret_dc_type

def get_parser():
    parser = argparse.ArgumentParser()
    gen_parser_from_dataclass(parser,Setup())
    
    from tasks import TASK_REGISTRY
    parser.add_argument('--task',choices=TASK_REGISTRY.keys(),help='task to implement')
    return parser

def parse_args_and_arch(parser,
                        parse_known=False,
                        input_args=None):
    from models import ARCH_MODEL_REGISTRY 
    
    args,_ = parser.parse_known_args(input_args)
    if hasattr(args,'arch'):
        if args.arch in ARCH_MODEL_REGISTRY:
            ARCH_MODEL_REGISTRY[args.arch].add_args(parser)
        else:
            raise RuntimeError()
        
    if hasattr(args,'task'):
        from tasks import TASK_REGISTRY
        TASK_REGISTRY[args.task].add_args(parser)
    
    if parse_known:
        args,extra = parser.parse_known_args(input_args)
    else:
        args = parser.parse_args(input_args)
        extra = None

    return args,extra

def _override_attr(sub_node,
                   data_class,
                   args):
    overrides = []
    if not inspect.isclass(data_class):
        return overrides
    
    def get_default(f):
        if not isinstance(f.default_factory,_MISSING_TYPE):
            return f.default_factory
        return f.default
        
    for k,v in data_class.__dataclass_fields__.items():
        if k.startswith('_'):
            continue

        val = get_default(v) if not hasattr(args,k) else getattr(args,k)
        
        field_type = interpret_dc_type(v.type)
        if (isinstance(val,str) and \
           field_type != str and \
           (not inspect.isclass(field_type) or not issubclass(field_type,Enum))):
            val = ast.literal_eval(val)
        
        if isinstance(val,tuple):
            val = list(val)
        
        v_type = getattr(v.type,'__origin__',None)
        if ((v_type is List or v_type is list or v_type is Optional) and \
            not isinstance(val,str)):
            if hasattr(v_type,'__args__'):
                t_args = v_type.__args__
                if len(t_args) == 1 and ('int' in t_args or 'float' in t_args):
                    val = list(map(t_args[0],val))
        elif val is not None and (field_type is bool or field_type is int or field_type is float):
                try:
                    val = field_type(val)
                except:
                    pass
        
        if val is None:
            overrides.append('{}.{}=null'.format(sub_node,k))
        elif val == '':
            overrides.append("{}.{}=''".format(sub_node,k))
        elif isinstance(val,str):
            overrides.append("{}.{}='{}'".format(sub_node,k,val))
        else:
            overrides.append('{}.{}={}'.format(sub_node,k,val))
    
    return overrides

def migrate_registry(name,value,registry,overrides,deletes,args):
    if value in registry:
        overrides.append('{}={}'.format(name,value))
        overrides.extend(_override_attr(name,registry[value],args))
    else:
        deletes.append(name)

def override_module_args(args):
    overrides,deletes = [],[]
    
    for k,v in Config.__dataclass_fields__.items():
        overrides.extend(_override_attr(k,v.type,args))
    
    no_dc = True
    if hasattr(args,'arch'):
        from models import ARCH_MODEL_REGISTRY
        m_cls = ARCH_MODEL_REGISTRY[args.arch]
        dc = getattr(m_cls,'__dataclass',None)
        if dc is not None:
            overrides.append('model={}'.format(args.arch))
            overrides.extend(_override_attr('model',dc,args))
            no_dc = False
    
    if hasattr(args,'task'):
        from tasks import TASK_DATACLASS_REGISTRY
        migrate_registry('task',args.task,TASK_DATACLASS_REGISTRY,overrides,deletes,args)
    else:
        deletes.append('task')
    
    if no_dc:
        deletes.append('model')
    
    return overrides,deletes

def _set_legacy_defaults(args,cls):
    if not hasattr(cls,'add_args'):
        return
    
    parser = argparse.ArgumentParser()
    cls.add_args(parser)
    
    for action in parser._actions:
        if action.dest is not argparse.SUPPRESS:
            if action.default is not argparse.SUPPRESS:
                if not hasattr(args,action.dest):
                    setattr(args,action.dest,action.default)

class omegaconf_no_object_check:
    def __init__(self):
        self.old_premitive = _utils.is_primitive_type
    
    def __enter__(self):
        _utils.is_primitive_type = lambda _:True
    
    def __exit__(self,*args,**kwargs):
        _utils.is_primitive_type = self.old_premitive

def convert_namespace_to_omegaconf(args):
    overrides,deletes = override_module_args(args)
    config_path = os.path.join('../','config')
    GlobalHydra.instance().clear()
    
    with initialize(config_path=config_path):
        try:
            composed_cfg = compose('config',overrides=overrides,strict=False)
        except:
            logger.error("Error when composing. Overrides: " + str(overrides))
            raise
        
        for k in deletes:
            composed_cfg[k] = None
    
    cfg = OmegaConf.create(OmegaConf.to_container(composed_cfg,resolve=True,enum_to_str=True))

    with omegaconf_no_object_check():
        if cfg.model is None and getattr(args,'arch',None):
            cfg.model = Namespace(**vars(args)) 
            from models import ARCH_MODEL_REGISTRY
            _set_legacy_defaults(cfg.model,ARCH_MODEL_REGISTRY[args.arch])

        if cfg.task is None and getattr(args,'task',None):
            cfg.task = Namespace(*vars(args))
            from tasks import TASK_REGISTRY
            _set_legacy_defaults(cfg.task,TASK_REGISTRY[args.task])
    
    OmegaConf.set_struct(cfg, True)
    
    return cfg


#%%
