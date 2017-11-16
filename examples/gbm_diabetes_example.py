'''
This example compares an sklearn.ensemble.GradientBoostingRegressor to a
simple_tree.simple_gbm.GBRegressor. The GBRegressor is as accurate, but a
lot slower. The data is the diabetes data from simple_tree.datasets.
'''
from time import time
from sklearn.metrics import mean_squared_error as mse
from simple_tree.simple_gbm import GBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from simple_tree.datasets import load_diabetes

print(__doc__)

# Set-up
xtr, ytr, xte, yte = load_diabetes()
gbm = GradientBoostingRegressor()
mygbm = GBRegressor()

# Running Time
print('Running time on fit')
print('-------------------')
print('Fitting on %s.' % gbm.__class__.__name__)
start = time()
gbm.fit(xtr, ytr)
end = time()
et = end - start
print('Fitting complete.')
print('Elapsed time %.6f seconds\n' % et)

print('Fitting on %s.' % mygbm.__class__.__name__)
start = time()
mygbm.fit(xtr, ytr)
end = time()
et = end - start
print('Fitting complete.')
print('Elapsed time %.6f seconds\n' % et)

# Accuracy
print('Accuracy - mse')
print('--------------')
pred = gbm.predict(xte)
print('MSE on %s: %.3f' % (gbm.__class__.__name__, mse(yte, pred)))
mypred = mygbm.predict(xte)
print('MSE on %s: %.3f\n' % (mygbm.__class__.__name__, mse(yte, mypred)))
