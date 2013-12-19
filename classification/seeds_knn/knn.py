import numpy as np

def find_plurality(labels):
    '''
    prediction = find_plurality(labels)
    
    Return the label of most votes.
    '''

    from collections import defaultdict
    counts = defaultdict(int)
    for label in labels:
        counts[label] += 1
   
    maxv = max(counts.values())
    for k,v in counts.items():
        if v == maxv:
            return k

def apply_knn(new_examples, data, labels, k = 8):
    '''
    predictions = apply_knn(new_examples, data, labels, k=8)

    Return the predicted label of new examples,based on data, labels.
    '''
    results = []
    for d in new_examples:
        dists = []
        for d2,label in zip(data, labels):
            dists.append( (np.linalg.norm(d2-d), label) )
        dists.sort(key = lambda di: di[0]) # di for norm(d2-d) above
        # print dists
        dists = dists[:k]
        results.append(find_plurality([label for _,label in dists]))

    # print len(results)
    return np.array(results).reshape(new_examples.shape[0])

def accuracy(labels, predictions):
    return np.sum(labels == predictions) / float(labels.shape[0])


def test():
    '''
    Debug
    '''
    import load
    import matplotlib.pyplot as plt
    import os

    feature_names, data, target_names, targets = load.load_seeds()
    target_names = np.array(target_names)
    labels = target_names[targets]

    pred = apply_knn(data, data, targets)
    pred = target_names[pred]

    #label_names = np.unique(labels)
    
    # plot original data
    plt.subplot(1, 2, 1)
    for k,marker,c in zip(xrange(3), "<ox", "rgb"):
        plt.scatter( data[labels == target_names[k], 0],
                     data[labels == target_names[k], 2],
                     marker = marker,
                     c = c );
    plt.title("origin")

    # plot knn predicted data
    plt.subplot(1, 2, 2)
    for k,marker,c in zip(xrange(3), "<ox", "rgb"):
        plt.scatter( data[pred == target_names[k], 0],
                     data[pred == target_names[k], 2],
                     marker = marker,
                     c = c)
                     
    # re-plot misclassified data
    wrong = plt.scatter( data[pred != labels, 0],
                         data[pred != labels, 2],
                         marker = 's',
                         c = 'y' );
    plt.legend([wrong], ["Wrong"], loc='upper right')

    file_dir = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(os.path.join(file_dir, "test", "knn_test.png"))
    
    print "Accuracy: %f" % accuracy(labels, pred)


if __name__ == '__main__':
    test()
