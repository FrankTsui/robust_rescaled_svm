import numpy as np 
import matplotlib.pyplot as plt 

def plot_decision_function(classifier, fea, gnd, title):
    '''
        plot the decision function in 2-d plane
        classifiers: the svm models
        fea: array like, shape = (smp_num, fea_num)
        gnd: array like, shape = (smp_num,)
        title: title of plot
    ''' 
    fea_min = fea.min(axis = 0)
    fea_max = fea.max(axis = 0)
    mesh_num = 100
    # meshgrid
    xx, yy = np.meshgrid(np.linspace(fea_min[0], fea_max[0], mesh_num), \
        np.linspace(fea_min[1], fea_max[1], mesh_num))

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()], last_model_flag = False)
    Z_first = Z[:, 0].copy()
    Z_last = Z[:, -1].copy()
    Z_first = Z_first.reshape(xx.shape)
    Z_last = Z_last.reshape(xx.shape)
    del Z

    # plot the line, the points
    leg_svm = plt.contour(xx, yy, Z_first, levels = [0.0], colors = 'k')
    leg_rsvm = plt.contour(xx, yy, Z_last, levels = [0.0], colors = 'r')
    posi_index = gnd == 1
    nega_index = gnd == -1
    marker_size = 70
    plt.scatter(fea[:, 0], fea[:, 1], marker = 'o', \
        s = classifier.smp_weights_mat[:, -1] * marker_size * 4, c = 'w', alpha = 1.0, edgecolors = 'm', label = 'weights')
    plt.scatter(fea[posi_index, 0], fea[posi_index, 1], marker = '^', s = marker_size, c = 'g', alpha = 0.8, label = 'posi')
    plt.scatter(fea[nega_index, 0], fea[nega_index, 1], marker = 'x', s = marker_size, c = 'b', label = 'nega')
    leg_svm.collections[0].set_label('svm')
    leg_rsvm.collections[0].set_label('rsvm')
    plt.legend(loc = 'upper left')
    plt.axis('on')
    plt.title(title)