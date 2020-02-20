import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from scipy import stats
from numpy.random import seed
import matplotlib.pyplot as plt
import time
from importlib import reload



def get_time_mili():
    import time
    return int(round(time.time() * 1000))


def get_time_stamp():
    from datetime import datetime
    now = datetime.utcnow()
    return str(now).split(".")[0].replace(":", "_").replace("-", "_").replace(" ", "__")


def start_file(ifile, message="Train start: ", time_stamp=True):
    from datetime import datetime
    now = datetime.utcnow()
    now = str(now).split(".")[0].replace(":", "_").replace("-", "_").replace(" ", "__")
    with open(ifile, 'w') as f:
        if time_stamp == True:
            f.write(message + " {}\n".format(now))
        else:
            f.write(message + "\n")


def end_file(ifile, message="Train end: ", time_stamp=True):
    from datetime import datetime
    now = datetime.utcnow()
    now = str(now).split(".")[0].replace(":", "_").replace("-", "_").replace(" ", "__")
    with open(ifile, 'a') as f:
        if time_stamp == True:
            f.write(message + " {}\n".format(now))
        else:
            f.write(message + "\n")


def append_file(ifile, message="Train end: "):
    with open(ifile, 'a') as f:
        f.write(message + "\n")

def str_to_tuple(istr):
  """
  Changes format "(680,420,60,60)" to (680, 420, 60, 60)
  """
  if (type(istr)==str) or type(istr) == np.str_:
    return tuple( [int(x) for x in list(filter( lambda x: len(x)>0, istr.replace("(","").replace(")","").split(",")))])
  else:
    return istr

def val_to_type(istr,target_type):
    if target_type == 'tuple':
        return str_to_tuple(istr)
    if target_type == 'int':
        return int(istr)
    if target_type == 'float':
        return float(istr)
    if target_type == 'str':
        return str(istr)
    if target_type == 'bool':
        return bool(istr)

def save_obj(obj, filename):
    import pickle
    import os
    odir="./"
    if "/" in filename:
        odir = "/".join(filename.split("/")[:-1])
    os.makedirs(odir, exist_ok=True)
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        print(filename + ".pkl saved")


def load_obj(name):
    import pickle
    if ".pkl" in name:
        name = "".join(name.split('.pkl')[:-1])
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_vector_from_list(ilist):
    cols = len(ilist)
    output = {}
    for i in range(cols):
        base = np.zeros((1,cols))
        base[0,i]=1
        output[ilist[i]]=base.copy()
    return output.copy()
