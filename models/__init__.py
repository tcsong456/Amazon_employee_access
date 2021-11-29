import os
import importlib
from contextlib import ExitStack
from omegaconf import open_dict,OmegaConf
from hydra.core.config_store import ConfigStore

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
    else:
        if model_type in ARCH_CONFIG_REGISTRY:
            with open_dict(cfg) if OmegaConf.is_config(cfg) else ExitStack():
                ARCH_CONFIG_REGISTRY[model_type](cfg)

    assert model is not None, (
        f"Could not infer model type from {cfg}. "
        "Available models: {}".format(
            MODEL_DATACLASS_REGISTRY.keys()
        )
        + f" Requested model type: {model_type}"
    )
        
    return model(cfg)
                
def register_model(name,dataclass=None):
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))

        MODEL_REGISTRY[name] = cls     
        cls.__dataclass = dataclass
        if dataclass is not None:
            MODEL_DATACLASS_REGISTRY[name] = dataclass
            
            cs = ConfigStore()
            dataclass.__name = name
            node = dataclass()
            cs.store(name=name,node=node,group='model',provider='tcsong')
            
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
        
        ARCH_MODEL_REGISTRY[arch_name] = MODEL_REGISTRY[model_name]
        ARCH_MODEL_INV_REGISTRY.setdefault(model_name,[]).append(arch_name)
        ARCH_CONFIG_REGISTRY[arch_name] = fn
        
        return fn
    
    return register_model_arch_fn


def import_modules(namespace):
    cur_dir = os.path.join(os.getcwd(),namespace)
    for file in os.listdir(cur_dir):
        if (not file.startswith('__') and
            not file.startswith('_')
            and file.endswith('.py')):
            model_name = file[:file.find('.py')]
            importlib.import_module(namespace + '.' + model_name)
                
import_modules('models')

#%%
