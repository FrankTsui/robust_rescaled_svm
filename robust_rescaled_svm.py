'''
this algorithm is presented in:
Guibiao Xu, Zheng Cao, Bao-Gang Hu and Jose Principe, Robust support vector machines based on the 
	rescaled hinge loss, Pattern Recognition, 2017.
'''
import numpy as np 
from sklearn.svm import SVC
from collections import OrderedDict
from config import config

class rsvm:
	def __init__(self, config):
		'''
			config: parameter settings
		'''
		self.config = config

	def _create_svm_object(self):
		'''
			create an svm object according to kernel type
		'''
		if self.config['kernel'] == 'linear':
			return SVC(C = self.config['C'], kernel = self.config['kernel'], \
				shrinking = self.config['shrinking'], probability = self.config['probability'], \
				tol = self.config['tol'], cache_size = self.config['cache_size'], \
				class_weight = self.config['class_weight'], verbose = self.config['verbose'], \
				max_iter = self.config['max_iter'], decision_function_shape = self.config['decision_function_shape'], \
				random_state = self.config['random_state'])
		elif self.config['kernel'] == 'poly':
			return SVC(C = self.config['C'], kernel = self.config['kernel'], \
				degree = self.config['degree'], gamma = self.config['gamma'], coef0 = self.config['coef0'], \
				shrinking = self.config['shrinking'], probability = self.config['probability'], \
				tol = self.config['tol'], cache_size = self.config['cache_size'], \
				class_weight = self.config['class_weight'], verbose = self.config['verbose'], \
				max_iter = self.config['max_iter'], decision_function_shape = self.config['decision_function_shape'], \
				random_state = self.config['random_state'])
		elif self.config['kernel'] == 'rbf':
			return SVC(C = self.config['C'], kernel = self.config['kernel'], gamma = self.config['gamma'], \
				shrinking = self.config['shrinking'], probability = self.config['probability'], \
				tol = self.config['tol'], cache_size = self.config['cache_size'], \
				class_weight = self.config['class_weight'], verbose = self.config['verbose'], \
				max_iter = self.config['max_iter'], decision_function_shape = self.config['decision_function_shape'], \
				random_state = self.config['random_state'])
		elif self.config['kernel'] == 'sigmoid':
			return SVC(C = self.config['C'], kernel = self.config['kernel'], \
				gamma = self.config['gamma'], coef0 = self.config['coef0'], \
				shrinking = self.config['shrinking'], probability = self.config['probability'], \
				tol = self.config['tol'], cache_size = self.config['cache_size'], \
				class_weight = self.config['class_weight'], verbose = self.config['verbose'], \
				max_iter = self.config['max_iter'], decision_function_shape = self.config['decision_function_shape'], \
				random_state = self.config['random_state'])

	def fit(self, train_fea, train_gnd):
		'''
			training method
			train_fea: array like, shape = (smp_num, fea_num)
			train_gnd: array like, shape = (smp_num,), -1 and +1
		'''
		# check elements in train_gnd, the element should be -1 or +1
		assert set(train_gnd) == set([-1, 1])

		train_num = train_fea.shape[0]
		# save sample weights across iterations
		self.smp_weights_mat = np.zeros(shape = (self.config['rsvm_iter_num'], train_num))
		# save svm models across iterations
		self.svmmodel_dict = OrderedDict()
		# save support vector ratios across iterations
		self.sv_ratio_vec = np.zeros(shape = (self.config['rsvm_iter_num'],))

		self.smp_weights_mat[0] = self.config['rsvm_v0']
		for iter_i in range(self.config['rsvm_iter_num']):
			self.svmmodel_dict[iter_i] = self._create_svm_object()
			self.svmmodel_dict[iter_i].fit(train_fea, train_gnd, sample_weight = self.smp_weights_mat[iter_i])
			self.sv_ratio_vec[iter_i] = np.float64(self.svmmodel_dict[iter_i].n_support_.sum()) / train_num * 100
			# update weights of samples
			if iter_i == (self.config['rsvm_iter_num'] - 1):
				break
			else:
				tmp_outputs = self.svmmodel_dict[iter_i].decision_function(train_fea)
				tmp_hinge_loss = np.maximum(0.0, 1.0 - tmp_outputs * train_gnd)
				# weights update function
				self.smp_weights_mat[iter_i + 1] = np.exp(-self.config['rsvm_eta'] * tmp_hinge_loss)
		self.smp_weights_mat = self.smp_weights_mat.transpose()

	def predict(self, test_fea, last_model_flag = True):
		'''
			prediction function
			test_fea: array like, shape = (smp_num, fea_num)
			last_model_flag: whether only use the last svm model or not

			return
			pred: array like, shape = (smp_num, iter_num)
		'''
		if last_model_flag:
			return self.svmmodel_dict[self.config['rsvm_iter_num'] - 1].predict(test_fea)
		else:
			test_num = test_fea.shape[0]
			pred = np.zeros(shape = (test_num, self.config['rsvm_iter_num']), dtype = np.int32)
			for iter_i in range(self.config['rsvm_iter_num']):
				pred[:, iter_i] = self.svmmodel_dict[iter_i].predict(test_fea)
			return pred 

	def score(self, test_fea, test_gnd, last_model_flag = True):
		'''
			return accuracy on the given test_fea and test_gnd
			test_fea: array like, shape = (smp_num, fea_num)
			test_gnd: array like, shape = (smp_num,), -1 and +1
			last_model_flag: whether only use the last svm model or not

			return
			accu_vec: a vector
		'''
		if last_model_flag:
			return self.svmmodel_dict[self.config['rsvm_iter_num'] - 1].score(test_fea, test_gnd) * 100
		else:
			accu_vec = np.zeros(shape = (self.config['rsvm_iter_num'],))
			for iter_i in range(self.config['rsvm_iter_num']):
				accu_vec[iter_i] = self.svmmodel_dict[iter_i].score(test_fea, test_gnd) * 100
			return accu_vec

	def decision_function(self, test_fea, last_model_flag = True):
		'''
			svm outputs
			test_fea: array like, shape = (smp_num, fea_num)
			last_model_flag: whether only use the last svm model or not

			return
			distance: array like, shape = (smp_num, iter_num)
		'''
		if last_model_flag:
			return self.svmmodel_dict[self.config['rsvm_iter_num'] - 1].decision_function(test_fea)
		else:
			test_num = test_fea.shape[0]
			distance = np.zeros(shape = (test_num, self.config['rsvm_iter_num']), dtype = np.float64)
			for iter_i in range(self.config['rsvm_iter_num']):
				distance[:, iter_i] = self.svmmodel_dict[iter_i].decision_function(test_fea)
			return distance

if __name__ == '__main__':
	np.random.seed(0)
	X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
	y = [-1] * 10 + [1] * 10
	train_num =  20
	config['rsvm_v0'] = np.ones(shape = (20, ), dtype = np.float64)
	config['rsvm_eta'] = 0.5
	rsvm_obj = rsvm(config)
	rsvm_obj.fit(X, y)
	print '#### sv ratio vector ####'
	print rsvm_obj.sv_ratio_vec
	print '#### smp_weights_mat ####'
	print rsvm_obj.smp_weights_mat
