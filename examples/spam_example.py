'''
This example compares accuracy and runtime of simple_tree.ClassificationTree
and sklearn.DecisionTreeClassifier on the spam dataset.
'''
from time import time
from sklearn.tree import DecisionTreeClassifier
from simple_tree.datasets import load_spam
from simple_tree import ClassificationTree
from sklearn.metrics import confusion_matrix as cm

print(__doc__)

print('Running time on fit')
print('-------------------')
print()
dt = DecisionTreeClassifier()
xtr, ytr, xte, yte = load_spam()
print('Fitting sklearn.DecisionTreeClassifer')
start = time()
dt.fit(xtr, ytr)
end = time()
et = end - start
print('Fitting complete.')
print('Elapsed time %.6f seconds' % et)
print()

ct = ClassificationTree()
print('Fitting simple_tree.ClassificationTree')
start = time()
ct.fit(xtr, ytr)
end = time()
et = end - start
print('Fitting complete.')
print('Elapsed time %.6f seconds' % et)
print()

sk_pred = dt.predict(xte)
st_pred = ct.predict(xte)
print('Accuracy - Confusion Matrices')
print('-----------------------------')
print()
print('sklearn:')
print(cm(yte, sk_pred))
print()
print('simple_tree.ClassificationTree:')
print(cm(yte, st_pred))
