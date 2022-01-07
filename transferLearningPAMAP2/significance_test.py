'''
Created on Jun 27, 2013

@author: lrothack
'''
import math
import scipy.stats
import scipy.misc
import numpy as np
import itertools
#import tqdm

#from patrec.evaluation._mt_permutation_test import permutation_test as mt_perm_test




class SignificanceBase(object):
    @staticmethod
    def significance_marker(p):
        '''
        R-style markings.
        '''
        if p <= 0.001:
            return '***'
        elif p <= 0.01:
            return '** '
        elif p <= 0.05:
            return '*  '
        elif p <= 0.1:
            return '.  '
        else:
            return '   '

    @staticmethod
    def significance_text(p):
        if p < 0.05:
            return 'p<=%.3f' % (max(1e-3, np.ceil(p * 1000) / 1000.0))
        else:
            return 'p=%.2f' % (p)


class DifferenceProportions(SignificanceBase):
    '''
    Implements the difference of two proportions test after
    T.G.Dietterich, Approximate Statistical Tests for Comparing Supervised
    Classification Learning Algorithms, 1997

    @requires: Classifiers A and B have been estimated on exactly the same training set.
    @requires: Classifiers A and B have been tested on exactly the same test set.
    '''


    def __init__(self, num_test_samples):
        '''
        Constructor
        @param num_test_samples: The number of samples in the test set.
        '''
        self.__num_test_samples = num_test_samples

    def significance_niveau(self, error_rate_a, error_rate_b, verbose=False):
        error_rate_mean = (error_rate_a + error_rate_b) / 2.0

        z_numerator = error_rate_a - error_rate_b
        z_denominator = math.sqrt((2 * error_rate_mean * (1 - error_rate_mean)) / self.__num_test_samples)
        z = math.fabs(z_numerator / z_denominator)  # IGNORE:C0103
        # signif_diff = z > 1.96 # Z_0.975 = 1.96 (two sided test: 0.025 + 0.025 = 0.05)
        p = scipy.stats.norm.sf(z) * 2
        if verbose:
            print ('abs-diff: %.4f' % math.fabs(z_numerator))
            print ('std-err: %.4f' % z_denominator)
            print ('z-value: %.4f' % z)
            print ('p-value: %.4f' % p)
        return p

    @staticmethod
    def string_for_p(p):
        s = SignificanceBase.significance_marker(p)
        s += ' (' + SignificanceBase.significance_text(p) + ')'
        return s

    def significance_test(self, error_rate_a, error_rate_b):
        '''
        Checks if the specified error rates are significantly different
        --> two sided test at significance level 0.95 (abs(z) > Z_0.975)

        @param error_rate_a: Classifier A's error rate on test set T in [0,1]
        @param error_rate_b: Classifier B's error rate on test set T in [0,1]
        '''


        #
        # Null-hypothesis: Classifiers A and B have the same error rate
        # Check if the null-hypothesis will be rejected with probability 0.05
        #
        p = self.significance_niveau(error_rate_a, error_rate_b)

        if p < 0.01:
            print ('Error rate difference is highly significant!')
        elif p < 0.05:
            print ('Error rate  difference is significant!')
        else:
            print ('Error rate difference is *not* significant!')

        print (DifferenceProportions.string_for_p(p))
        return p < 0.05,p



if __name__ == '__main__':
    '''
    LaraMM : 39973
    LaraIMU: 51863
    PAMAP2: 3785
    Opportunity: 9894
    
    for pamap2 cnn above 88-17 is significant
    '''
    d = DifferenceProportions(9894)
    bool,p = d.significance_test(0.2453,0.2333)
    print(p)
    
    d = DifferenceProportions(34180)
    #bool,p = d.significance_test(0.333,0.322)
    

