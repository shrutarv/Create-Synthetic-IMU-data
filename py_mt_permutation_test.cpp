#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <vector>
#include <omp.h>

#include <boost/python.hpp>
#include <numpy/arrayobject.h>

#include <progress_bar.h>

using namespace boost;
using namespace python;

// module definitions
void* extract_pyarray(PyObject* x)
{
	return x;
}

// exception classes
class DataMustBeFloatException : public std::exception {
public:
    const char* what() const throw() {
        return "The data type of the input array must be either np.float32 or np.float64";
    }
};
class DataTypesMustBeEqualException : public std::exception {
public:
    const char* what() const throw() {
        return "The data types of the input arrays must match";
    }
};
void translate(const std::exception& e) {
    // Use the Python 'C' API to set up an exception object
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

/**
 * Compute the number of runs to obtain the desired standard deviation of
 * the p value.
 *
 * The usual formula is cv = ((1 - p)/(p * n_runs))^0.5
 * where cv is the coefficient of variation. The p-value's standard error
 * can be calculated by: p * cv = p_std
 * When substituting into the formula above and solving for number of runs
 * we get:
 *     n_runs = p(1 - p) / p_std^2
 *
 * For calculating the upper bound of MC runs (permutations), we have to derive
 * the formula for p and then set it to 0. This yields a maximum number of iterations
 * of p = 0.5. Substituting this value in for p, the upper bound of MC runs
 * for any p-value is:
 *     n_runs = 1 / (4*p_std^2)
 *
 * @brief compute_number_mc_runs
 * @param p_std: the desired p-value standard deviation
 */
unsigned long upper_bound_mc_runs(float p_std) {
    return std::lround(1.0/(4*p_std*p_std));
}

/**
 * @brief compute the factorial
 * @param n
 */
double factorial(long n) {
    double res = 1.0;
    for(long i = 2; i < n+1; ++i) {
        res *= i;
    }
    return res;
}

template <typename T>
PyObject* mt_permutation_test_template(PyArrayObject* samples_a, PyArrayObject* samples_b, float p_std, bool verbose) {
    // sanity check
    if (PyArray_DTYPE(samples_a)->type_num != PyArray_DTYPE(samples_b)->type_num) {
        throw DataTypesMustBeEqualException();
    }
    // copy array contents into std::vectors
    long size_a = PyArray_SIZE(samples_a);
    long size_b = PyArray_SIZE(samples_b);
    std::vector<T> vec_samples_a(static_cast<T*>(PyArray_DATA(samples_a)),
                                 static_cast<T*>(PyArray_DATA(samples_a)) + size_a);
    std::vector<T> vec_samples_b(static_cast<T*>(PyArray_DATA(samples_b)),
                                 static_cast<T*>(PyArray_DATA(samples_b)) + size_b);
    std::vector<T> vec_samples_all = vec_samples_a;
    vec_samples_all.insert(vec_samples_all.end(), vec_samples_b.begin(), vec_samples_b.end());
    // compute means and distance
    double mean_a = std::accumulate(vec_samples_a.begin(), vec_samples_a.end(), 0.0) / vec_samples_a.size();
    double mean_b = std::accumulate(vec_samples_b.begin(), vec_samples_b.end(), 0.0) / vec_samples_b.size();
    double dist_obs = std::abs(mean_a - mean_b);

    // set up counter variable for the test
    unsigned long perm_dist_bigger = 0;
    // get the upper bound of MC runs for the given standard error of p
    unsigned long n_runs = upper_bound_mc_runs(p_std);

    // run permutation test
    ProgressBar* pg;
    if (verbose) {
        std::cout << "Running permutation test:" << std::endl;
        std::cout << "    desired std err for p: " << p_std << std::endl;
        std::cout << "    number MC runs:        " << n_runs << std::endl;
        std::cout << "    number of threads:     " << omp_get_max_threads() << std::endl;
        pg = new ProgressBar();
        pg->start(n_runs);
        pg->update(0);
    }
#pragma omp parallel for reduction(+:perm_dist_bigger)
    for (unsigned long run_idx = 0; run_idx < n_runs; ++run_idx) {
        std::vector<T> perm_vec(vec_samples_all);
        std::random_shuffle(perm_vec.begin(), perm_vec.end());
        double mean_perm_a = std::accumulate(perm_vec.begin(), perm_vec.begin() + size_a, 0.0) / size_a;
        double mean_perm_b = std::accumulate(perm_vec.begin() + size_a, perm_vec.end(), 0.0) / size_b;
        double dist_perm = std::abs(mean_perm_a - mean_perm_b);
        if (dist_perm >= dist_obs) {
            perm_dist_bigger += 1;
        }
        if (verbose) {
            pg->update(run_idx+1);
        }
    }
    double p_value = perm_dist_bigger / static_cast<double>(n_runs);
    if (verbose) {
        std::cout << "\nTest finished:" << std::endl;
        std::cout << "    p-value: " << p_value << std::endl;
        std::cout << "    std. err. p: " << p_std << std::endl;
        delete pg;
    }

    // create PyTuple for outputs
    PyObject* tuple = PyTuple_New(2);
    // put PyArray objects in PyTuple
    PyTuple_SetItem(tuple, 0, PyFloat_FromDouble(p_value));
    PyTuple_SetItem(tuple, 1, PyLong_FromLong(n_runs));
    return tuple;
}

PyObject* mt_permutation_test(PyArrayObject& samples_a, PyArrayObject& samples_b, float p_std, bool verbose) {
    switch (PyArray_DTYPE(&samples_a)->type_num) {
    case NPY_FLOAT32:
        return mt_permutation_test_template<float>(&samples_a, &samples_b, p_std, verbose);
    case NPY_FLOAT64:
        return mt_permutation_test_template<double>(&samples_a, &samples_b, p_std, verbose);
    default:
        throw DataMustBeFloatException();
    }
}

BOOST_PYTHON_MODULE(_mt_permutation_test)
{
    register_exception_translator<DataMustBeFloatException>(&translate);
    register_exception_translator<DataTypesMustBeEqualException>(&translate);
    def("permutation_test", &mt_permutation_test);

    converter::registry::insert(&extract_pyarray, type_id<PyArrayObject>());
	import_array();
}
