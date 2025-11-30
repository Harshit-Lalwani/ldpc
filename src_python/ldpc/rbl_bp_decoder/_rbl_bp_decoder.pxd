#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++

from libcpp.vector cimport vector

cdef extern from "rbl_bp.hpp" namespace "ldpc::rbl_bp":

    cdef cppclass RBLBPDecoderCpp "ldpc::rbl_bp::RBLBP_Decoder":
        RBLBPDecoderCpp(int rows, int cols, const vector[vector[int]]& parity_matrix) except +
        vector[int] decode(const vector[double]& received_LLR, int max_iter, double alpha)


cdef class RBLBPDecoder:
    cdef RBLBPDecoderCpp* _decoder
    cdef int _m
    cdef int _n
    cdef int _max_iter
    cdef double _alpha
    cdef bint _memory_allocated
