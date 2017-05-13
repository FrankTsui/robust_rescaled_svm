'''
to show the performance of robust rescaled svm on the German credit dataset
'''

from sklearn.datasets import load_svmlight_file
import numpy as np 
import matplotlib.pyplot as plt 
from robust_rescaled_svm import rsvm 
from config import config
import common

# load German credit dataset
# the dataset can be downloaded from (german.number): https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
fea, gnd = load_svmlight_file('misc/german.numer_scale')
np.random.seed(400)
smp_num = fea.shape[0]
permu_seq = np.random.permutation(smp_num)
train_percent = 0.8
train_num = np.int32(smp_num * train_percent)
train_fea = fea[permu_seq[:train_num]].copy()
train_gnd = gnd[permu_seq[:train_num]].copy()
test_fea = fea[permu_seq[train_num:]].copy()
test_gnd = gnd[permu_seq[train_num:]].copy()
del fea, gnd 

# add label noise
train_index = np.random.permutation(train_num)
noise_percent = 0.1
noise_num = np.int32(train_num * noise_percent)
for i in range(noise_num):
	if train_gnd[train_index[i]] == 1:
		train_gnd[train_index[i]] = -1
	else:
		train_gnd[train_index[i]] = 1

# robust rescaled svm
config['rsvm_v0'] = 'linear'
config['rsvm_v0'] = np.ones(shape = [train_num])
config['rsvm_eta'] = 1.0
config['rsvm_iter_num'] = 3
rsvm_obj = rsvm(config)
rsvm_obj.fit(train_fea, train_gnd)

# prediction
train_accu_vec = rsvm_obj.score(train_fea, train_gnd, last_model_flag = False)

test_pred = rsvm_obj.predict(test_fea, last_model_flag = False)
test_accu_vec1 = np.zeros(shape = [config['rsvm_iter_num']])
for i in range(config['rsvm_iter_num']):
	test_accu_vec1[i] = (test_pred[:, i] == test_gnd).astype(np.float64).mean() * 100

test_accu_vec2 = rsvm_obj.score(test_fea, test_gnd, last_model_flag = False)

# print the results
repeat_num = 70
print '-' * repeat_num
print '# sv ratio: {}'.format(rsvm_obj.sv_ratio_vec)
print '# training accuracy: {}'.format(train_accu_vec)
print '# testing accuracy (predict): {}'.format(test_accu_vec1)
print '# testing accuracy (score): {}'.format(test_accu_vec2)
print '-' * repeat_num

# plot the figure
line_width = 2.0
marker_size = 10.0
plt.figure(num = 0, figsize = (14, 6))
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, config['rsvm_iter_num'] + 1), rsvm_obj.sv_ratio_vec, \
	linestyle = '-', linewidth = line_width, marker = 'o', markersize = marker_size, color = 'b')
plt.xlabel('iter_num')
plt.ylabel('sv ratio')
plt.title('support vector ratio')
plt.subplot(1, 2, 2)
plt.plot(np.arange(1, config['rsvm_iter_num'] + 1), train_accu_vec, \
	linestyle = '-', linewidth = line_width, marker = 'o', markersize = marker_size, color = 'b', label = 'training accu')
plt.plot(np.arange(1, config['rsvm_iter_num'] + 1), test_accu_vec2, \
	linestyle = '-', linewidth = line_width, marker = 'o', markersize = marker_size, color = 'r', label = 'testing accu')
plt.xlabel('iter_num')
plt.ylabel('accuracy')
plt.title('accuracy')
plt.legend(loc = 'best')
plt.show()