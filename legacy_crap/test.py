
import os, glob, copy
import ctypes
import numpy as np
import datetime
import dateutil.parser
import scipy.interpolate
import scipy.optimize

compiledIRBEMdir = \
os.path.abspath(os.path.join(os.path.dirname( __file__ ), \
'..', 'source'))
print(compiledIRBEMdir)
compiledIRBEMdir = '/home/wyatt/Downloads/irbem-code-r620-trunk/source'
fullPaths = glob.glob(os.path.join(compiledIRBEMdir,'*.so'))
print(fullPaths)

compiledIRBEMname = os.path.basename(fullPaths[0])
print(compiledIRBEMname)
