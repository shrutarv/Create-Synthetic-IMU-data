'''
Created on Jun 27, 2013

@author: lrothack
'''
import math
import scipy.stats
import scipy.misc
import numpy as np
import itertools
import tqdm

#from patrec.evaluation._mt_permutation_test import permutation_test as mt_perm_test

class ConfidenceIntervals(object):
    '''
    Computes the confidence intervals for a given classification result after
    E. Paulus and M. Lehning, Die Evaluierung von Spracherkennungssystemen in
    Deutschland, BMBF Tech. Report, TU Braunschweig, 1995
    '''


    def __init__(self, p=0.05):
        '''
        @param p: significance level.
        FIXME: The parameter has changed its meaning. Update your code!

        Defines the confidence number c for the
        intervals (based on the significance level), e.g.
        Sign. lvl   c
        0.95        1.960
        0.99        2.567
        0.999       3.291
        '''
        self.__c = scipy.stats.norm.isf(p * 0.5)

    def compute(self, no_samples, classification_rate):
        '''Computes the confidence intervall for
        @param no_samples: The given number of samples
        @param classification_rate: The classification rate;
        how many % (0<p<1) were classified correctly

        @return: upper and lower border of the confidence interval'''
        n = float(no_samples)
        k = float(int(no_samples * classification_rate))
        c = self.__c
        # Compute upper and lower confidence interval border
        p1 = 2 * k + (c ** 2)
        p1 = p1 / (2 * (n + (c ** 2)))

        p2 = p1 ** 2
        p2 = p2 - ((k ** 2) / (n * (n + (c ** 2))))
        p2 = math.sqrt(p2)
        # lower
        p_low = p1 - p2
        # upper
        p_up = p1 + p2
        return p_low, p_up


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
            print ('abs-diff: %.4f') % math.fabs(z_numerator)
            print ('std-err: %.4f') % z_denominator
            print ('z-value: %.4f') % z
            print ('p-value: %.4f') % p
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
        return p < 0.05


class PermutationTest(SignificanceBase):
    '''
    full permutation test based on difference of means.

    Literature:
    Smucker, Allen, Carterette (2007)
    A Comparison of Statistical Significance Tests for Information Retrieval Evaluation

    Good, Phillip (2000)
    Permutation Tests -- A Practical Guide to Resampling Methods for Testing Hypotheses
    Springer Series in Statistics
    '''

    def __init__(self, limit='auto'):
        '''
        @param limit: * None    - the exact test probability is computed (may compute forever)
                      * 'auto'  - the number of test will be computed using the upper bound
                                  for p = 0.01 +- 0.001 (default), i.e., 250,000 iterations.
                      * any int - the number of permutations used.
        '''
        self.__limit = limit

    @staticmethod
    def iteration_bound(p_std_err):
        '''
        Upper bound for the number of monte carlo runs for a test to level p +- dp
        with an arbitrary p-value

        @param p_std_err: the desired standard error of the p-value
        '''
        return int(1.0 / (4 * p_std_err ** 2))

    @staticmethod
    def iteration_estimate(p_value, p_std_err=None):
        '''
        Estimate of the number of monte carlo runs for an test to level p +- dp.

        @param p: significance level
        @param p_std_err: standard error of p, if none is given it is set to 0.1*p by default
        '''
        if p_std_err is None:
            p_std_err = p_value * 0.1
        return int((p_value * (1.0 - p_value)) / (p_value ** 2))

    @staticmethod
    def exact_test(samples_a, samples_b):
        '''
        @param a: ndarray of the first sample set
        @param b: ndarray of the second sample set
        @return: [p, num]
                 p - the approximate probability that the two sequences are same
                 num - the number of permutations tested
        '''
        samples_a = samples_a.flatten()
        samples_b = samples_b.flatten()
        if samples_a.shape[0] < 1 or samples_b.shape[0] < 1:
            return 1, 0
        samples_all = np.concatenate((samples_a, samples_b))
        dist_obs = np.abs(np.mean(samples_a) - np.mean(samples_b))
        lena = float(samples_a.shape[0])
        lenb = float(samples_b.shape[0])
        sum_all = np.sum(samples_all)
        over = 0
        runs = 0
        for perm in itertools.combinations(samples_all, samples_a.shape[0]):
            sum_perm = sum(perm)
            mean_perm = sum_perm / lena
            mean_other = (sum_all - sum_perm) / lenb
            # two sided error, use abs
            if np.abs(mean_perm - mean_other) >= dist_obs:
                over += 1
            runs += 1
        p_approx = over / float(runs)
        return  p_approx, runs

    def permutation_test(self, samples_a, samples_b, p_std_err=0.001, engine='python'):
        '''
        Compute the significance level for the null hypothesis.
        The value will be approximated by only performing 'limit' number of permutations to test
        except when the total number of possible permutations is lower. In that case, the exact
        test will be used instead and num == total returned.

        @param a: ndarray of the first sample set
        @param b: ndarray of the second sample set
        @param p_std_err: the desired standard error for the p-value
        @param engine: The implementation to be used. Can be 'cpp' or 'python'

        @return: [p, num, total]
                 p - the approximate probability that the two sequences are not different or same
                 num - the number of permutations tested
                 total - the number of permutations possible
        '''
        # flatten arrays
        samples_a = samples_a.flatten()
        samples_b = samples_b.flatten()

        if samples_a.shape[0] < 1 or samples_b.shape[0] < 1:
            return 1, 0, 0
        if samples_a.shape[0] > samples_b.shape[0]:
            samples_a, samples_b = samples_b, samples_a
        ntotal = scipy.special.comb(samples_a.shape[0] + samples_b.shape[0], samples_a.shape[0])
        if not math.isinf(ntotal):
            ntotal = int(ntotal)
        limit = self.__limit
        if limit == 'auto':
            limit = PermutationTest.iteration_bound(p_std_err)

        # run permutation test
        if self.__limit is None or limit > ntotal:
            # if we can compute all permutations, do so
            p_val, n_permutations = PermutationTest.exact_test(samples_a, samples_b)
            return p_val, n_permutations, n_permutations
        elif engine == 'cpp':
            # ... else run permutation test with C++ engine ...
            #test_res = mt_perm_test(samples_a, samples_b, p_std_err, True)
            #return test_res + ()
            return
        elif engine == 'python':
            # ... or with the standard python engine
            samples_all = np.concatenate((samples_a, samples_b))
            dist_obs = np.abs(np.mean(samples_a) - np.mean(samples_b))
            lena = float(samples_a.shape[0])
            lenb = float(samples_b.shape[0])
            sum_all = np.sum(samples_all)
            over = 0
            runs = 0
            for _ in tqdm.tqdm(range(int(limit))):
                # note: this is slightly incorrect
                # since the same permutation
                # may be chosen twice
                perm = np.random.permutation(samples_all)[:samples_a.shape[0]]
                sum_perm = np.sum(perm)
                mean_perm = sum_perm / lena
                mean_other = (sum_all - sum_perm) / lenb
                # two sided error, uses abs
                if np.abs(mean_perm - mean_other) >= dist_obs:
                    over += 1
                runs += 1
            p_approx = over / float(runs)
            return p_approx, runs, ntotal
        else:
            raise ValueError('Unknown engine %s' % engine)

    @staticmethod
    def string_for_p(p, n, t=None):
        '''
        @param p: probability niveau
        @param n: number of permutations tested
        @param t: number of permutations total
        '''
        s = SignificanceBase.significance_marker(p)
        s += ' (' + SignificanceBase.significance_text(p)
        if n < 1e3:
            s += ', n=%d' % n
        else:
            s += ', n=%.1e' % n
        if t is not None:
            if t < 1e3:
                s += '/%d' % t
            else:
                s += '/%.1e' % t
        s += ')'
        return s

    def find_significant_pairs(self, lists, maxp=0.05):
        '''
        get all significant pairs out of a list of results.
        @returns a list of (index_s,index_g,p_value,p_string) where index_s is the index of the list with the smaller mean.
        '''
        n_sets = len(lists)
        res = []
        for i, j in itertools.combinations(range(n_sets), 2):
            p_approx, n_runs, poss_perms = self.permutation_test(lists[i], lists[j])
            if p_approx > maxp:
                continue
            signif_str = self.string_for_p(p_approx, n_runs, poss_perms)
            if np.mean(lists[i]) < np.mean(lists[j]):
                res.append((i, j, p_approx, signif_str))
            else:
                res.append((j, i, p_approx , signif_str))
        return res


class ParameterAnalyzer(object):

    @staticmethod
    def find_varying_by_one(modes, nparam):
        '''
        find entries in modes that vary by just one entry
        @param modes: tuples of parameters (p1,p2,..,pn)
        @param nparam: number of parameters to test
        @return corresponding sets of indices
        '''
        res = []
        for param in range(nparam):
            op = list(set(range(nparam)) - set([param]))
            pools = []
            for i in range(len(modes)):
                ps = modes[i][:nparam]
                pool = [i]
                for j in range(len(modes)):
                    qs = modes[j][:nparam]
                    if qs[param] == ps[param] or ps == qs:
                        continue
                    ok = True
                    for k in op:
                        if qs[k] != ps[k]:
                            ok = False
                    if not ok:
                        continue
                    pool.append(j)
                pool = set(pool)
                if not pool in pools and len(pool) > 1:
                    pools.append(pool)
            res.append(pools)
        return res

# -- examples --

def significance_test():
    diff_prop_test = DifferenceProportions(num_test_samples=6033)
    diff_prop_test.significance_test(error_rate_a=0.081, error_rate_b=0.071)
    print ('-----')
    diff_prop_test.significance_test(error_rate_a=0.081, error_rate_b=0.073)

if __name__ == '__main__':
    significance_test()

