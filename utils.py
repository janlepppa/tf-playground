import numpy as np

def batch_indices(n,batch_size, shuffle = True, seed = None):
    '''
    Split n examples in batches of size 'batch_size'. The last batch maybe smaller.
    Returns a list containing the batch indices. Before creating the batches, indices are shuffled on default. 
    '''
    assert batch_size < n
    
    inx = np.array(range(n))
     
    if shuffle == True:
        
        if seed is None:
            rng = np.random.RandomState()
        else:
            rng = np.random.RandomState(seed)
            
        rng.shuffle(inx)
    
    batch_indices = []
    
    n_batches = int(np.ceil(n/batch_size))
    
  
    for ii in range(n_batches):
        
        if ii == n_batches - 1:
            batch = inx[batch_size*ii:]
        else:
            batch = inx[batch_size*ii:batch_size*(ii + 1)]
            
        batch_indices.append(batch)
    
    return batch_indices

def normalize_labels(label_vec):
    '''
    Convert labels in 'label_vec' to integers starting from 0.
    Returns converted labels, an encoder mapping from original 
    to new labels and a decoder mapping from new to original labels.
    '''
    n = len(label_vec)
    unique_labels = np.unique(label_vec)
    
    encoder = dict(zip(list(unique_labels),range(len(unique_labels))))
    decoder = dict(zip(range(len(unique_labels)), list(unique_labels)))

    new_labels = np.empty_like(label_vec,dtype = np.int32)
    
    for ii in range(n):        
        new_labels[ii] = encoder[label_vec[ii]]
        
    return new_labels, encoder,decoder 


def fmnist_train_test_split():
    ''' 
    Returns training and test sets with labels for fashion mnist data.
    '''
    X_train, y_train = load_fmnist("fmnist/data/fashion", kind= "train")
    X_test, y_test = load_fmnist("fmnist/data/fashion", kind= "t10k")
    
    return X_train,y_train,X_test,y_test


def bc_train_test_split(seed = None, test_prop = 0.2):
    '''
    Loads training and test sets with labels for Breast Cancer data (binary classification)
    '''
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    data = load_breast_cancer()
    X = data["data"].astype(np.float32)
    y = data["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop, random_state= seed)
    
    return X_train, y_train, X_test, y_test
    
    
def load_fmnist(path, kind='train'):
    '''
    Loads fashion-mnist from local folder specified by path.
    kind    'train' or 't10k'    
    This function was taken from
    https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    '''
    import os
    import gzip
    
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels