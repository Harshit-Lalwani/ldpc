#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++

import numpy as np
import scipy.sparse
from typing import Union
from libcpp.vector cimport vector

import ldpc.helpers.scipy_helpers

cimport numpy as cnp

ctypedef cnp.uint8_t uint8_t


cdef vector[vector[int]] _pcm_to_cpp(pcm):
    cdef int i
    cdef int j
    cdef int m
    cdef int n
    cdef vector[vector[int]] matrix
    cdef vector[int] row

    if not (isinstance(pcm, np.ndarray) or isinstance(pcm, scipy.sparse.spmatrix)):
        raise TypeError(
            f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}"
        )

    pcm = ldpc.helpers.scipy_helpers.convert_to_binary_sparse(pcm)
    m, n = pcm.shape[0], pcm.shape[1]

    matrix.resize(m)

    if isinstance(pcm, np.ndarray):
        for i in range(m):
            row.clear()
            for j in range(n):
                row.push_back(<int>pcm[i, j])
            matrix[i] = row
    else:
        dense_pcm = pcm.toarray()
        for i in range(m):
            row.clear()
            for j in range(n):
                row.push_back(<int>dense_pcm[i, j])
            matrix[i] = row

    return matrix


cdef class RBLBPDecoder:
    def __cinit__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix], max_iter: int = 50, alpha: float = 0.5):
        cdef vector[vector[int]] matrix

        self._memory_allocated = False

        if max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if alpha <= 0.0:
            raise ValueError("alpha must be a positive float.")

        matrix = _pcm_to_cpp(pcm)
        self._m, self._n = pcm.shape[0], pcm.shape[1]
        self._max_iter = max_iter
        self._alpha = alpha
        self._decoder = new RBLBPDecoderCpp(self._m, self._n, matrix)
        self._memory_allocated = True

    def __del__(self):
        if self._memory_allocated:
            del self._decoder

    def decode(self, received_llr: np.ndarray, max_iter: int = -1, alpha: float = -1.0) -> np.ndarray:
        cdef int i
        cdef int iter_count
        cdef double alpha_value
        cdef vector[double] llr_cpp
        cdef vector[int] decoded_cpp
        cdef cnp.ndarray[uint8_t, ndim=1] out

        if received_llr.ndim != 1:
            raise ValueError("received_llr must be a 1D numpy array.")
        if received_llr.shape[0] != self._n:
            raise ValueError(f"received_llr must have length {self._n}. Not {received_llr.shape[0]}.")

        iter_count = self._max_iter if max_iter <= 0 else max_iter
        alpha_value = self._alpha if alpha <= 0.0 else alpha
        if iter_count <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if alpha_value <= 0.0:
            raise ValueError("alpha must be a positive float.")

        llr_cpp.resize(self._n)
        for i in range(self._n):
            llr_cpp[i] = <double>received_llr[i]

        decoded_cpp = self._decoder.decode(llr_cpp, iter_count, alpha_value)

        out = np.zeros(self._n, dtype=np.uint8)
        for i in range(self._n):
            out[i] = <uint8_t>decoded_cpp[i]

        return out

    @property
    def m(self) -> int:
        return self._m

    @property
    def n(self) -> int:
        return self._n

    @property
    def max_iter(self) -> int:
        return self._max_iter

    @property
    def alpha(self) -> float:
        return self._alpha
