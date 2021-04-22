'''
Created on Oct 27, 2017

@author: fmoya
'''

import numpy as np
from sig_test import PermutationTest


def read_txt(path):
    
    #predictions_warehouse__DO_person_1
    print (path)
    f = open( path, "r" )
    predictions = []
    for line in f:
        predictions.append(int(line))
    
    predictions = np.array(predictions)    
    
        
    return predictions


def significance_acc():

    number_samples = 11364   #Number of samples in test dataset
    test_tuples = [(0.8947, 0.8780)]    # Two classification rates percentage of correct samples to compare
    result_tuples = []
    
    tester = PermutationTest()
    
    for t_i in test_tuples:
    
        p_a = t_i[0]
        p_b = t_i[1]
    
        if p_a > 1 or p_b > 1:
            p_a /= 100.0
            p_b /= 100.0
    
        n_a = int(p_a * number_samples)
        n_b = int(p_b* number_samples)
    
        A = np.zeros(number_samples)
        B = np.zeros(number_samples)
    
        A[:n_a] = 1
        B[:n_b] = 1
    
        #np.savetxt('prueba_saving.txt', A==B, fmt='%d')
        pval, _, _ = tester.permutation_test(A, B)
    
    return pval


if __name__ == '__main__':
    
    pathtest_LSTM = '/home/fmoya/Documents/Doktorado/OrderPicking/CNN_softmax/experiments_test/predictions_labels.txt'
    pathtest = '../experiments_final_cosine/predictions_labels.txt'
    
    tester = PermutationTest()
    
    test_LSTM = read_txt(pathtest_LSTM)
    test = read_txt(pathtest)
    
    pval, _, _ = tester.permutation_test(test_LSTM, test[0:test_LSTM.shape[0]])
    
    #pval = significance_acc()
    

    print ("p value {}".format(pval))
    
    
    
    print ("Done")