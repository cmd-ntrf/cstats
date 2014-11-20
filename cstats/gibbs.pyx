import numpy
cimport numpy
cimport cython

from scipy.special import ndtr, ndtri

ctypedef numpy.float_t DTYPE_t

cdf = ndtr
ppf = ndtri
dot = numpy.dot
delete = numpy.delete
asarray = numpy.asarray

@cython.boundscheck(False)
def gibbs_sampling(int n, numpy.ndarray[DTYPE_t, ndim=1] mean, numpy.ndarray[DTYPE_t, ndim=2] cov, numpy.ndarray[DTYPE_t, ndim=2] bounds, int burning=0, int thinning=1):
    """Jayesh H. Kotecha and Petar M. Djuric (1999) :
    GIBBS SAMPLING APPROACH FOR GENERATION OF TRUNCATED MULTIVARIATE
    GAUSSIAN RANDOM VARIABLES
    """

    cdef int dim = mean.shape[0]
    cdef numpy.ndarray samples = numpy.empty((n, dim), dtype=numpy.float)
    cdef numpy.ndarray U = numpy.random.uniform(size=(burning+n*thinning, dim))

    cdef numpy.ndarray sd = numpy.empty(dim, dtype=numpy.float)
    cdef numpy.ndarray P = numpy.empty((dim, dim-1), dtype=numpy.float)
    cdef numpy.ndarray sigma = numpy.empty((dim-1, dim-1), dtype=numpy.float)
    cdef numpy.ndarray sigma_i = numpy.empty(dim-1, dtype=numpy.float)
    cdef int i, j    
    for i in xrange(dim):
        sigma = delete(delete(cov, i, axis=0), i, axis=1)
        sigma_i = delete(cov[i, :], i, 0)
        P[i] = dot(sigma_i, numpy.linalg.inv(sigma))
        sd[i] = numpy.sqrt(cov[i, i] - dot(P[i], sigma_i))

    cdef numpy.ndarray x = mean.copy()
    cdef float mu_j, f_a, f_b
    for i in xrange(-burning, n*thinning):
        for j in xrange(dim):
            mu_j = mean[j] + dot(P[j][:j], x[:j] - mean[:j]) + dot(P[j][j:], x[j+1:] - mean[j+1:])
            f_a, f_b = cdf((bounds[j] - mu_j) / sd[j])
            x[j] = mu_j + sd[j] * ppf(U[i, j] * (f_b - f_a) + f_a)
        if i >= 0:
            if thinning == 1:
                samples[i] = x
            elif i % thinning == 0:
                samples[i // thinning] = x

    return samples