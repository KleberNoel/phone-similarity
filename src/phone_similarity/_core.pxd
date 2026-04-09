# Public typed declarations for _core Cython module.
#
# These allow other .pyx modules to cimport the fast primitives
# directly without going through the Python layer.

cpdef int hamming_distance(a, b) except -1
cpdef double hamming_similarity(a, b)
cpdef double feature_edit_distance(
    object seq_a,
    object seq_b,
    dict phoneme_features,
    double insert_cost=*,
    double delete_cost=*,
)
cpdef list batch_pairwise_hamming(list arrays)
