import argparse
from dataclass.configs import Setup
from dataclass.utils_dataclass import gen_parser_from_dataclass

def get_parser():
    parser = argparse.ArgumentParser()
    gen_parser_from_dataclass(parser,Setup())
    return parser
    
parser = get_parser()

#%%
#s = Setup()
#t = s.__dataclass_fields__['stacking_splits'].type
#issubclass(t,Enum)
