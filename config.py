'''
option settings for robust svm
'''
config = {
	# options for svm.SVC
	'C': 1.0,
	'kernel': 'rbf',
	'degree': 3.0,
	'gamma': 'auto',
	'coef0': 0.0,
	'shrinking': True,
	'probability': False,
	'tol': 1.0e-3,
	'cache_size': 200,
	'class_weight': None,
	'verbose': False,
	'max_iter': -1,
	'decision_function_shape': None,
	'random_state': None,
	# options for robust svm
	'rsvm_v0': None,
	'rsvm_eta': 0.5,
	'rsvm_iter_num': 5
}