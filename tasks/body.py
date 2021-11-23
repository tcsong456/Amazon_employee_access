from helper.utils import logger
from tasks import register_task
from base_task import BaseTask
from dataclass.config import BaseDataClass
from dataclass.choices import BODY_CHOICES
from dataclasses import dataclass,field
from typing import Optional

@dataclass
class BodyConfig(BaseDataClass):
    option: Optional[BODY_CHOICES] = field(default='meta',metadata={'help':'the set of data to build'})

@register_task('build_body',dataclass=BodyConfig)
class BuildBody(BaseTask):
    def __init__(self,
                 args,
                 train,
                 test,
                 model,
                 arch):
        


#%%