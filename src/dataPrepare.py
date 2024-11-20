import numpy as np
import h5py
from utils import *
from config import config

import os
import logging

def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        # Read the number of vectors and the dimension of each vector
        nvecs, dim = np.fromfile(f, count=2, dtype=np.uint32)

        # Adjust the number of vectors to read based on start_idx and chunk_size
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size

        # Read the vector data, which are of type float32
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32, offset=start_idx * 4 * dim)
    
    return arr.reshape(nvecs, dim)


def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        # Read the number of vectors and the dimension of each vector
        nvecs, dim = np.fromfile(f, count=2, dtype=np.uint32)

        # Adjust the number of vectors to read based on start_idx and chunk_size
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size

        # Read the vector data, which are of type int32
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32, offset=start_idx * 4 * dim)
    
    return arr.reshape(nvecs, dim)

# add SIFT and other data as well

# load groundtruth data & train data into relevant files
def getTraindata(dataname):
    metric = config.DATASET[dataname]['metric']
    datapath = '../../data/{}/'.format(dataname)
    trainpath = datapath + 'train.npy'
    gtpath = datapath + 'groundTruth.npy'

    if os.path.exists(trainpath) and os.path.exists(gtpath):     #check file size as well   
        # print ("GT already there")
        print("Groundtruth already there")
    else:
        #load the full data and get fraction
        fulldata = getFulldata(dataname, datapath)
        N = fulldata.shape[0]
        if N > 10**6:
            # pick = np.random.choice(N, np.clip(N//100, 10**4, 10**6), replace=False) # fix seed
            np.random.seed(0)
            pick = np.random.choice(N, 10**6, replace=False) # fix seed
            data_train = fulldata[pick,:]
        else:
            data_train = fulldata
        
        gt = getTrueNNS(data_train, metric, 100) # get top true 100 neighbors
        np.save(gtpath, gt)
        np.save(trainpath, data_train)
        del fulldata

def getFulldata(dataname, datapath):
    if dataname == 'glove':
        if os.path.exists(datapath+'fulldata.dat'):
            dt = config.DATASET[dataname]['dt'] 
            N = config.DATASET[dataname]['N']
            d = config.DATASET[dataname]['d']
            return np.array(np.memmap(datapath+'fulldata.dat', dtype=dt, mode='c', shape=(N,d)))
        else:
            data = np.array(h5py.File('../../data/glove/glove-100-angular.hdf5', 'r').get('train'))
            norms = np.linalg.norm(data,axis=1)
            savememmap(datapath+'fulldata.dat', data)
            np.save(datapath+'norms.npy', norms)
            return data
    if dataname == 'sift':
        if os.path.exists(datapath+'fulldata.dat'):
            dt = config.DATASET[dataname]['dt'] 
            N = config.DATASET[dataname]['N']
            d = config.DATASET[dataname]['d']
            return np.array(np.memmap(datapath+'fulldata.dat', dtype=dt, mode='c', shape=(N,d)))
        else:
            data = np.array(h5py.File('../../data/sift/sift-128-euclidean.hdf5', 'r').get('train'))
            norms = np.linalg.norm(data,axis=1)
            savememmap(datapath+'fulldata.dat', data)
            np.save(datapath+'norms.npy', norms)
            return data
    if dataname == 'deep-1b':
        # 读取 Yandex Deep1B 数据集
        if os.path.exists(datapath + 'fulldata.dat'):
            dt = config.DATASET[dataname]['dt']
            N = config.DATASET[dataname]['N']
            d = config.DATASET[dataname]['d']
            return np.array(np.memmap(datapath + 'fulldata.dat', dtype=dt, mode='c', shape=(N, d)))
        else:
            # 读取 .fbin 文件，假设数据是 float32 类型
            data_fbin = read_fbin('../../data/deep-1b/base.1B.fbin')
            norms = np.linalg.norm(data_fbin, axis=1) # axis存疑
            # 保存为 memmap
            savememmap(datapath + 'fulldata.dat', data_fbin)
            np.save(datapath + 'norms.npy', norms)
            return data_fbin
    else:
        raise ValueError(f"Unknown dataset: {dataname}")

def getQueries(dataname):
    datapath = '../../data/{}/'.format(dataname)
    if dataname == 'glove':
        if os.path.exists(datapath+'queries.npy') and os.path.exists(datapath+ 'neighbors100.npy'): 
            queries = np.load(datapath+'queries.npy')
            neighbors100 = np.load(datapath+ 'neighbors100.npy')
        else:
            queries = np.array(h5py.File('../../data/glove/glove-100-angular.hdf5', 'r').get('test'))
            neighbors100 = np.array(h5py.File('../../data/glove/glove-100-angular.hdf5', 'r').get('neighbors'))
            np.save(datapath+'queries.npy', queries)
            np.save(datapath+ 'neighbors100.npy', neighbors100)
        return [queries, neighbors100]

    if dataname == 'sift':
        if os.path.exists(datapath+'queries.npy') and os.path.exists(datapath+ 'neighbors100.npy'): 
            queries = np.load(datapath+'queries.npy')
            neighbors100 = np.load(datapath+ 'neighbors100.npy')
        else:
            queries = np.array(h5py.File('../../data/sift/sift-128-euclidean.hdf5', 'r').get('test'))
            neighbors100 = np.array(h5py.File('../../data/sift/sift-128-euclidean.hdf5', 'r').get('neighbors'))
            np.save(datapath+'queries.npy', queries)
            np.save(datapath+ 'neighbors100.npy', neighbors100)
        return [queries, neighbors100]






def load_and_print_data(dataname, datapath, num_samples=5):
    """
    Load a sample of the data for the given dataset and print the values.
    
    :param dataname: The name of the dataset ('glove', 'sift', 'yandex')
    :param datapath: The path to the dataset files
    :param num_samples: The number of samples to display (default is 5)
    """
    try:
        if dataname == 'glove':
            if os.path.exists(datapath + 'fulldata.dat'):
                # Load data from the .dat file
                dt = config.DATASET[dataname]['dt']
                N = config.DATASET[dataname]['N']
                d = config.DATASET[dataname]['d']
                data = np.array(np.memmap(datapath + 'fulldata.dat', dtype=dt, mode='r', shape=(N, d)))
                print(f"Loaded {dataname} data. First {num_samples} samples:")
                print(data[:num_samples])
            else:
                logging.warning(f"Data file {datapath + 'fulldata.dat'} not found for {dataname}.")
        
        elif dataname == 'sift':
            if os.path.exists(datapath + 'fulldata.dat'):
                # Load data from the .dat file
                dt = config.DATASET[dataname]['dt']
                N = config.DATASET[dataname]['N']
                d = config.DATASET[dataname]['d']
                data = np.array(np.memmap(datapath + 'fulldata.dat', dtype=dt, mode='r', shape=(N, d)))
                print(f"Loaded {dataname} data. First {num_samples} samples:")
                print(data[:num_samples])
            else:
                logging.warning(f"Data file {datapath + 'fulldata.dat'} not found for {dataname}.")
        
        elif dataname == 'yandex':
            # Check if the data has already been memmapped
            if os.path.exists(datapath + 'fulldata.dat'):
                dt = config.DATASET[dataname]['dt']
                N = config.DATASET[dataname]['N']
                d = config.DATASET[dataname]['d']
                data = np.array(np.memmap(datapath + 'fulldata.dat', dtype=dt, mode='r', shape=(N, d)))
                print(f"Loaded {dataname} data. First {num_samples} samples:")
                print(data[:num_samples])
            else:
                logging.warning(f"Data file {datapath + 'fulldata.dat'} not found for {dataname}.")
        
        else:
            logging.error(f"Dataset {dataname} not recognized.")
    
    except Exception as e:
        logging.error(f"An error occurred while loading data for {dataname}: {e}")


def load_and_print_norms(dataname, datapath, num_samples=5):
    """
    Load and print a sample of the norms for the given dataset.
    
    :param dataname: The name of the dataset ('glove', 'sift', 'yandex')
    :param datapath: The path to the dataset files
    :param num_samples: The number of norms to display (default is 5)
    """
    try:
        if dataname == 'glove' or dataname == 'sift':
            if os.path.exists(datapath + 'norms.npy'):
                norms = np.load(datapath + 'norms.npy')
                print(f"Loaded {dataname} norms. First {num_samples} norms:")
                print(norms[:num_samples])
            else:
                logging.warning(f"Norms file {datapath + 'norms.npy'} not found for {dataname}.")
        
        elif dataname == 'yandex':
            if os.path.exists(datapath + 'norms.npy'):
                norms = np.load(datapath + 'norms.npy')
                print(f"Loaded {dataname} norms. First {num_samples} norms:")
                print(norms[:num_samples])
            else:
                logging.warning(f"Norms file {datapath + 'norms.npy'} not found for {dataname}.")
        
        else:
            logging.error(f"Dataset {dataname} not recognized.")
    
    except Exception as e:
        logging.error(f"An error occurred while loading norms for {dataname}: {e}")


