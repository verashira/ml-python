import numpy as np

def find_linear_bound(features, labels, target):
    best_acc = -1
    is_target = (labels == target)
    
    for i in xrange(features.shape[1]):
        feas = features[:, i]
        for t in feas:
            pred = (feas > t)
            pred_lt = (is_target == pred).mean()
            pred_st = (is_target == ~pred).mean()
            pred_acc = max(pred_lt,pred_st)
            
            if pred_acc > best_acc:
                is_larger_part = True if pred_lt > pred_st else False
                best_acc = pred_acc
                best_feature = i
                best_bound = t
    
    return best_feature, best_bound, is_larger_part, best_acc
    
#if __name__ == "__main__":
#    from sklearn.datasets import load_iris
#    data = load_iris()
#    features = data.data
#    feature_names = data.feature_names
#    target = data.target
#    target_names = data.target_names
#    labels = target_names[target]
#    features = features[labels != "setosa"]
#    labels = labels[labels != "setosa"]
#    fea,b,lg,acc =  find_linear_bound(features, labels, "virginica")
#    print fea,b,lg,acc