# robust_rescaled_svm

The codes are for the algorithm presented in:
  
 Guibiao Xu, Zheng Cao, Bao-Gang Hu and Jose Principe, [Robust support vector machines based on the rescaled hinge loss function][link_paper], Pattern Recognition, Vol. 63, pp. 139-148, March 2017.

Usuage:
1. demo_rsvm_outlier_plot.py
2. demo_rsvm_on_german_credit.py

Parameter settings:
1. parameters for svm
2. rsvm_v0: the intial sample weights; they can be all one's or set using some other weight functions presented in the paper
3. rsvm_eta: the scaling constant; it determines the upper bound of the rescaled hinge loss function; usually set it around 0.5
4. rsvm_iter_num: the iteration number of half-quadratic optimization algorithm; rsvm_iter_num = 3 usually gives good results

Prerequisites:
1. numpy
2. sklearn

[link_paper]: http://www.sciencedirect.com/science/article/pii/S0031320316303065