'''
This example compares accuracy and runtime of SimpleTree 
and sklearn.DecisionTreeClassifier on the spam dataset.
'''
from time import time
from sklearn.tree import DecisionTreeClassifier
from simple_tree.datasets import load_spam
from simple_tree import SimpleTree
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

st = SimpleTree()
print('Fitting SimpleTree')
start = time()
st.fit(xtr, ytr)
end = time()
et = end - start
print('Fitting complete.')
print('Elapsed time %.6f seconds' % et)
print()

sk_pred = dt.predict(xte)
st_pred = st.predict(xte)
print('Accuracy - Confusion Matrices')
print('-----------------------------')
print()
print('sklearn:')
print(cm(yte, sk_pred))
print()
print('SimpleTree:')
print(cm(yte, st_pred))
