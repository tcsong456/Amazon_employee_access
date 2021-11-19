import os
import importlib
import argparse

MODEL_REGISTRY = {}
MODEL_DATACLASS_REGISTRY = {}
ARCH_MODEL_INV_REGISTRY = {}
ARCH_MODEL_REGISTRY = {}
ARCH_CONFIG_REGISTRY = {}

def build_model(cfg):
    model = None
    model_type = getattr(cfg,'arch')
    
    if model_type in ARCH_MODEL_REGISTRY:
        model = ARCH_MODEL_REGISTRY[model_type]
    elif model_type in MODEL_DATACLASS_REGISTRY:
        model = MODEL_REGISTRY[model_type]
    
    if model_type in MODEL_DATACLASS_REGISTRY:
        dc = MODEL_DATACLASS_REGISTRY[model_type]
        
        dc.from_namspace(cfg)

def register_model(name,dataclass=None):
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        
        MODEL_REGISTRY[name] = cls
        
        cls.__dataclass = dataclass
        if dataclass is not None:
            MODEL_DATACLASS_REGISTRY[name] = dataclass
        
        return cls
    
    return register_model_cls

def register_model_architecture(model_name,arch_name):
    def register_model_arch_fn(fn):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                "Cannot register model architecture for unknown model type ({})".format(
                    model_name
                )
            )
        
        if arch_name in ARCH_MODEL_REGISTRY:
            raise ValueError(
                "Cannot register duplicate model architecture ({})".format(arch_name)
            )
        if not callable(fn):
            raise ValueError(
                "Model architecture must be callable ({})".format(arch_name)
            )
        
        ARCH_MODEL_REGISTRY[arch_name] = ARCH_MODEL_REGISTRY[model_name]
        ARCH_MODEL_INV_REGISTRY.setdefault(model_name,[]).append(arch_name)
        ARCH_CONFIG_REGISTRY[arch_name] = fn
        
        return fn
    
    return register_model_arch_fn

def import_modules(namespace):
    cur_dir = os.path.join(os.getcwd(),'models')
    for file in cur_dir:
        if (not file.startswith('__') and
            not file.startswith('_')
            and file.endswith('.py')):
            module_name = file[:file.find('.py')]
            importlib.import_module(namespace + '.' + module_name)
            
            if module_name in MODEL_REGISTRY:
                parser = argparse.ArgumentParser(add_help=False)
                group_archs = parser.add_argument_group('Named architectures')
                group_archs.add_argument('--arch',choices=ARCH_MODEL_INV_REGISTRY[module_name])
                
                args = parser.add_argument_group('command-line arguments')
                MODEL_REGISTRY[module_name].add_args(args)
                
import_modules('models')

#%%
