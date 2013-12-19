import numpy as np

def load_dataset(name, delimiter='\t'):
    '''
    data, labels = load_dataset(dataset_name)
    
    Load a dataset of given name.
    
    Returns
    -------
    data : numpy ndarray of features of data
    labels : list of str, class labels
    '''
    
    data = []
    labels = []
    with open(name) as dfile:
        for line in dfile:
            tokens = line.strip().split(delimiter)
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
    
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

if __name__ == '__main__':
    import os
    file_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(file_dir, "data", "seeds.tsv")
    data, labels = load_dataset(data_dir)
    labels = labels.reshape(labels.shape[0], 1)
    
    print np.hstack((data[:6], labels[:6]))
