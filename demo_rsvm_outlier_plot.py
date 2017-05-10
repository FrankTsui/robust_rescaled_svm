import numpy as np 
import matplotlib.pyplot as plt 
from rescaled_robust_svm import rsvm 
from config import config
import common

# generate samples
np.random.seed(0)
posi_num = 10
nega_num = 10
posi_fea = np.random.normal(loc = [1.0, 0.0], scale = 0.3, size = (posi_num, 2))
posi_gnd = np.ones(shape = (posi_num,), dtype = np.int32)
nega_fea = np.random.normal(loc = [-1.0, 0.0], scale = 0.3, size = (nega_num, 2))
nega_gnd = -1 * np.ones(shape = (nega_num,), dtype = np.int32)
fea = np.concatenate([posi_fea, nega_fea], axis = 0)
gnd = np.concatenate([posi_gnd, nega_gnd], axis = 0)

# parameter settings
config['kernel'] = 'linear'
config['rsvm_v0'] = np.ones(shape = (posi_num + nega_num,))
config['rsvm_eta'] = 1.0
config['rsvm_iter_num'] = 7

# without outliers
rsvm_obj1 = rsvm(config)
rsvm_obj1.fit(fea, gnd)
print '# without outliers, sv ratio vector: ', rsvm_obj1.sv_ratio_vec
# with outliers
gnd[1] = -1
gnd[5] = -1
gnd[posi_num + 6] = 1
rsvm_obj2 = rsvm(config)
rsvm_obj2.fit(fea, gnd)
print '# with outliers, sv ratio vector: ', rsvm_obj2.sv_ratio_vec

# plot figures
plt.figure(1)
common.plot_decision_function(rsvm_obj1, fea, gnd, title = 'without outliers')
plt.figure(2)
common.plot_decision_function(rsvm_obj2, fea, gnd, title = 'with outliers')
plt.show()