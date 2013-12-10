import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from find_linear_bound import find_linear_bound

# Initialize the directories
file_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data");
result_dir = os.path.join(file_dir, "result")

# Load the dataset from sklearn
data = load_iris()
features = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names
labels = target_names[target]

# Plot all scatter diagrams with combinations of all features
plt.clf()
feature_pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
for i, (f0,f1) in enumerate(feature_pairs):
    plt.subplot(2, 3, i+1)
    for j,marker,c in zip(xrange(3), ">ox", "rgb"):
        plt.scatter(features[target == j, f0],
                    features[target == j, f1],
                    marker = marker,
                    c = c)
    plt.xlabel(feature_names[f0])
    plt.ylabel(feature_names[f1])
    plt.xticks([])
    plt.yticks([])
    
plt.savefig(os.path.join(result_dir, "data_scatter.png"))


# Plot the first classification linear boundary
# We choose the 'petal width' of the feature to decide the boundary
plt.clf()

bound_fea_0 = 3 # petal width
bound_fea_1 = 2 # petal length
area1c = (0.36,0.67,1.)
#area2c = (0.98,0.52,0.52)
area2c = (0.9,0.9,0.9)

setosa = (labels == 'setosa')
not_setosa = ~setosa
bound_up = features[~setosa, bound_fea_0].min()  # we have know ~senosa will have 
bound_down = features[setosa, bound_fea_0].max() # large value
bound = (bound_up + bound_down)/2

area_xl = features[:,bound_fea_0].min()*0.9
area_xr = features[:,bound_fea_0].max()*1.1
area_yd = features[:,bound_fea_1].min()*0.9
area_yu = features[:,bound_fea_1].max()*1.1

plt.fill_between([area_xl, bound], area_yd, area_yu, color=area1c)
plt.fill_between([bound, area_xr], area_yd, area_yu, color=area2c)
plt.scatter(features[setosa, bound_fea_0],
            features[setosa, bound_fea_1],
            c='g', marker='o')
plt.scatter(features[~setosa, bound_fea_0],
            features[~setosa, bound_fea_1],
            c='r', marker='x')         
plt.xlim(area_xl, area_xr)
plt.ylim(area_yd, area_yu)
plt.xlabel(feature_names[bound_fea_0])
plt.ylabel(feature_names[bound_fea_1])
plt.legend(["setosa","not"], loc="upper left")
plt.savefig(os.path.join(result_dir, "first_bound.png"))


# Plot the second boundary, now to classify out the 
# versicolor and virginica. We will let it try the all
# features and find the best one automatically
plt.clf()
bound_fea, bound, is_larger_part, acc = find_linear_bound(
    features[~setosa], labels[~setosa], "virginica")
bound_fea_o = 0
area_xl = features[:,bound_fea].min()*0.9
area_xr = features[:,bound_fea].max()*1.1
area_yd = features[:,bound_fea_o].min()*0.9
area_yu = features[:,bound_fea_o].max()*1.1
if is_larger_part:
    area_lc = area2c
    area_rc = area1c
else:
    area_lc = area1c
    area_rc = area2c
plt.fill_between([area_xl, bound], area_yd, area_yu, color=area_lc)
plt.fill_between([bound, area_xr], area_yd, area_yu, color=area_rc)
plt.scatter(features[labels == "virginica", bound_fea],
            features[labels == "virginica", bound_fea_o],
            c='g', marker='o')
plt.scatter(features[labels == "versicolor", bound_fea],
            features[labels == "versicolor", bound_fea_o],
            c='r', marker='x')         
plt.xlim(area_xl, area_xr)
plt.ylim(area_yd, area_yu)
plt.xlabel(feature_names[bound_fea])
plt.ylabel(feature_names[bound_fea_o])
plt.legend(["virginica","versicolor"], loc="upper left")
plt.savefig(os.path.join(result_dir, "second_bound.png"))
