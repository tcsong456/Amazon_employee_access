import os
import importlib
from hydra.core.config_store import ConfigStore

TASK_REGISTRY = {}
TASK_CLASS_NAMES = {}
TASK_DATACLASS_REGISTRY = {}

def setup_task(cfg,**kwargs):
    task_name = getattr(cfg,'task')
    task_name in TASK_REGISTRY,'{} is not registered'.format(task_name)
    
    task = TASK_REGISTRY[task_name]
    if task_name in TASK_CLASS_NAMES:
        dc = TASK_CLASS_NAMES[task_name]
        cfg = dc.from_namespace(cfg)
    
    assert (
        task is not None
    ), f"Could not infer task type from {cfg}. Available argparse tasks: {TASK_REGISTRY.keys()}. Available hydra tasks: {TASK_DATACLASS_REGISTRY.keys()}"
    
    return task.setup_task(cfg,**kwargs)

def register_task(name,dataclass=None):
    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError("Cannot register duplicate task ({})".format(name))
        
        if cls.__name__ in TASK_CLASS_NAMES:
            raise ValueError(
                "Cannot register task with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES[name] = cls.__name__
        
        cls.__dataclass = dataclass
        if dataclass is not None:
            TASK_DATACLASS_REGISTRY[name] = dataclass
            
            cs = ConfigStore()
            node = dataclass()
            node._name = name
            cs.store(name=name,node=node,provider='tcsong',group='task')
        
        return cls
    
    return register_task_cls

def import_tasks(namespace):
    cur_dir = os.path.join(os.getcwd(),'tasks')
    for file in os.listdir(cur_dir):
        if (not file.startswith('__') and \
            not file.startswith('_') and \
            file.endswith('.py')):
            task_name = file[:file.find('.py')] if file.endswith('.py') else file
            
            importlib.import_module(namespace + '.' + task_name)

import_tasks('tasks')
    
    #%%
