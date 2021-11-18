import logzero
import logging

def custome_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logger = logzero.setup_logger(formatter=formatter,
                                  level=logging.INFO,
                                  name=name)
    return logger

logger = custome_logger('amazon')



#%%
