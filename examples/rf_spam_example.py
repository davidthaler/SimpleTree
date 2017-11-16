'''
This example compares sklearn.ensemble.RandomForestClassifier 
to simple_tree.simple_forest.RFClassifier for accuracy and
running time. The accuracy is close. The running time is not.
The data is the spam data from simple_tree.datasets.
'''
from time import time
from sklearn.ensemble import RandomForestClassifier
from simple_tree.simple_forest import RFClassifier
from simple_tree.datasets import load_spam
from sklearn.metrics import confusion_matrix as cm

print(__doc__)

# Set-up
xtr, ytr, xte, yte = load_spam()
rf = RandomForestClassifier(n_estimators=30, min_samples_leaf=30, max_depth=5)
myrf = RFClassifier(n_trees=30, min_samples_leaf=30, max_depth=5)

# Running Time
print('Running time on fit')
print('-------------------')
print('Fitting on sklearn.RandomForestRegressor')
start = time()
rf.fit(xtr, ytr)
end = time()
et = end - start
print('Fitting complete.')
print('Elapsed time %.6f seconds\n' % et)

print('Fitting on simple_tree.simple_forest.RFClassifier')
start = time()
myrf.fit(xtr, ytr)
end = time()
et = end - start
print('Fitting complete.')
print('Elapsed time %.6f seconds\n' % et)

# Accuracy
print('Accuracy - Confusion Matrices')
print('-----------------------------')
print('For sklearn.RandomForestRegressor:')
print(cm(yte, rf.predict(xte)))
print()
print('For RFClassifier:')
print(cm(yte, myrf.predict(xte)))
print()