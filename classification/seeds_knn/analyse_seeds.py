import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from load import load_seeds
from knn import apply_knn

feature_names, data, target_names, targets = load_seeds()
#feature_names, data, target, labels = load_seeds()
#label_names = np.unique(labels).tolist()

# Regularization
mean, std = data.mean(0), data.std(0)
data -= mean
data /= std

x0, x1 = data[:,0].min()*0.9, data[:,0].max()*1.1
y0, y1 = data[:,2].min()*0.9, data[:,2].max()*1.1
x = np.linspace(x0, x1, 100)
y = np.linspace(y0, y1, 100)
x,y = np.meshgrid(x, y)

new_examples = np.vstack([x.ravel(), y.ravel()]).T
print new_examples.shape
print data[:, (0,2)].shape

pred = apply_knn(new_examples,
                 data[:, (0,2)],
                 targets).reshape(x.shape)


# Plot predction field
cmap = ListedColormap([(1.,.6,.6),(.6,1.,.6),(.6,.6,1.)])
plt.clf()
plt.xlim(x0, x1)
plt.ylim(y0, y1)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[2])
plt.pcolormesh(x, y, pred, cmap=cmap)

cmap = ListedColormap([(1.,.0,.0),(.0,1.,.0),(.0,.0,1.)])
plt.scatter(data[:,0], data[:,2], c=targets, cmap=cmap)

file_dir = os.path.dirname(os.path.realpath(__file__))
plt.savefig(os.path.join(file_dir, "result", "knn.png"))
