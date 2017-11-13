'''
This example compares runtime and mean squared error of the Lou Gehrig's
disease data (load_als from simple_tree.datasets).
'''
from time import time
from sklearn.tree import DecisionTreeRegressor
from simple_tree.datasets import load_als
from simple_tree import RegressionTree
from sklearn.metrics import mean_squared_error as mse

print(__doc__)

# Set-up
xtr, ytr, xte, yte = load_als()
dt = DecisionTreeRegressor(min_samples_leaf=30)
mytree = RegressionTree(min_samples_leaf=30)

# Runtime
print('Running time on fit')
print('-------------------')
print('Fitting sklearn.DecisionTreeRegressor')
start = time()
dt.fit(xtr, ytr)
end = time()
et = end - start
print('Fitting complete.')
print('Elapsed time %.6f seconds\n' % et)

print('Fitting simple_tree.RegressionTree')
start = time()
mytree.fit(xtr, ytr)
end = time()
et = end - start
print('Fitting complete.')
print('Elapsed time %.6f seconds\n' % et)

# Accuracy
print('Accuracy: Mean Squared Error')
print('----------------------------')
print('sklearn.DecisionTreeRegressor:')
sk_pred = dt.predict(xte)
sk_mse = mse(yte, sk_pred)
print('mse: %.6f\n' % sk_mse)

print('simple_tree.RegressionTree:')
my_pred = mytree.predict(xte)
my_mse = mse(yte, my_pred)
print('mse: %.6f\n' % my_mse)
