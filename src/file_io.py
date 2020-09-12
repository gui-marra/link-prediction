import pandas as pd
import numpy as np
import os


def SaveData (path, X_train, X_test, y_train, y_test=[]):
    
    parent_folder = '/'.join([x for x in path.split('/')[:-2]])
    
    X_train.to_csv(os.path.join(path, 'X_train.csv'), sep=',', index=False)
    X_test.to_csv(os.path.join(path, 'X_test.csv'), sep=',', index=False)
    
    pd.DataFrame(y_train).to_csv(os.path.join(parent_folder,'y_train.csv'), sep=',', index=False, header=False)
    
    if len(y_test) > 0:
        pd.DataFrame(y_test).to_csv(os.path.join(parent_folder,'y_test.csv'), sep=',', index=False, header=False)
        
        
        
def LoadData (path):
    
    parent_folder = '/'.join([x for x in path.split('/')[:-2]])
    
    X_train = pd.read_csv(os.path.join(path, 'X_train.csv'), sep=',')
    X_test = pd.read_csv(os.path.join(path, 'X_test.csv'), sep=',')
    y_train = np.genfromtxt(os.path.join(parent_folder,'y_train.csv'), delimiter=',', skip_header=0)
    
    try:
        y_test = np.genfromtxt(os.path.join(parent_folder,'y_test.csv'), delimiter=',', skip_header=0)
        return X_train, X_test, y_train, y_test
    
    except Exception:
        pass
        
    return X_train, X_test, y_train