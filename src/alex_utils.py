# Function definitions for useful things
import numpy as np
import pickle
import sys

def print_nice(x):
   for i in range(0,len(x)):
        print(x[i])

def print_dir(object):
    x = dir(object)
    print_nice(x)

class DummyFile(object):
    def write(self,x): pass

# just suppresses output
#silent_version_of_foo = silence_func(foo)
def silence_func(func):
    def wrapper(*args, **kwargs):
        save_stdout = sys.stdout
        save_stderr = sys.stderr
        sys.stdout = DummyFile()
        sys.stderr = DummyFile()
        func(*args, **kwargs)
        sys.stdout = save_stdout
        sys.stderr = save_stderr
    return wrapper


def argmax_n(x,n):
    return np.argpartition(x,-n)[-n:]

def max_n(x,n):
    return x[argmax_n(x,n)]

def whos(variable):
    try:
        print(str(np.shape(variable)) + "\t\t" + str(type(variable)))
    except:
        print(str(len(variable)) + "\t\t" + str(type(variable)))

def load(filepath):
    filetemp = open(filepath,'rb')
    data    = pickle.load(filetemp)
    filetemp.close()
    return data

def save(filepath, variables):
    file_temp = open(filepath,'wb')
    pickle.dump(variables, file_temp)
    file_temp.close()



 

