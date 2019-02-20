import pdb 
import time, numpy as np
import h5py, dask.array as da
import dask

import sys
#read in config files
config1 = sys.modules['config'].config1
config2 = sys.modules['config'].config2
config3 = sys.modules['config'].config3

# from config import *
__all__ = ["np","da","dask","h5py","pdb","time","config1","config2","config3"]

#for key in sys.modules:
#    if key in ['np','pdb','time','da','dask']:
#        print(key)