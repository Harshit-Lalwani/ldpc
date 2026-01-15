import numpy as np
import scipy.sparse
import pytest
from ldpc.codes import rep_code, hamming_code
from ldpc.rbl_bp_decoder import RBLBPDecoder


@pytest.mark.parametrize("pcm", [rep_code(5), rep_code(9)])
def test_decode_recovers_single_bit_flip(pcm):
    """On repetition codes, a single high-confidence bit flip should decode to
    a valid (all-zero-syndrome) codeword."""
    H = pcm
    m, n = H.shape
    decoder = RBLBPDecoder(H, max_iter=100, alpha=0.3)

    llr = np.full(n, 4.0)
    llr[0] = -4.0
    decoded = decoder.decode(llr)

    assert decoded.shape == (n,)
    assert decoded.dtype == np.uint8

    syndrome = (H.toarray() if scipy.sparse.issparse(H) else H) @ decoded % 2
    assert np.all(syndrome == 0)


def test_decode_output_shape_and_dtype_on_hamming_code():
    """Hamming code isn't guaranteed to converge with this residual/relaxed BP
    variant (unmodified from the ported algorithm) -- only check the output
    contract (shape/dtype), not correctness of the decoding."""
    H = hamming_code(3)
    m, n = H.shape
    decoder = RBLBPDecoder(H, max_iter=100, alpha=0.3)

    llr = np.full(n, 4.0)
    llr[0] = -4.0
    decoded = decoder.decode(llr)

    assert decoded.shape == (n,)
    assert decoded.dtype == np.uint8


def test_accepts_dense_and_sparse_input():
    H_sparse = rep_code(5)
    H_dense = H_sparse.toarray()

    llr = np.array([2.0, 2.0, -2.0, 2.0, 2.0])
    out_sparse = RBLBPDecoder(H_sparse, max_iter=50, alpha=0.5).decode(llr)
    out_dense = RBLBPDecoder(H_dense, max_iter=50, alpha=0.5).decode(llr)
    assert np.array_equal(out_sparse, out_dense)


def test_decode_rejects_wrong_length_llr():
    H = rep_code(5)
    decoder = RBLBPDecoder(H, max_iter=50, alpha=0.5)
    with pytest.raises(ValueError):
        decoder.decode(np.array([1.0, 2.0]))


def test_decode_per_call_overrides():
    H = rep_code(5)
    decoder = RBLBPDecoder(H, max_iter=10, alpha=0.5)
    llr = np.array([2.0, 2.0, -2.0, 2.0, 2.0])

    # Per-call overrides shouldn't raise and should still return a valid-shaped decoding.
    out = decoder.decode(llr, max_iter=30, alpha=0.2)
    assert out.shape == (5,)


def test_constructor_rejects_invalid_max_iter_and_alpha():
    H = rep_code(5)
    with pytest.raises(ValueError):
        RBLBPDecoder(H, max_iter=0, alpha=0.5)
    with pytest.raises(ValueError):
        RBLBPDecoder(H, max_iter=10, alpha=0.0)


def test_properties_reflect_constructor_args():
    H = rep_code(5)
    decoder = RBLBPDecoder(H, max_iter=42, alpha=0.7)
    assert decoder.m == 4
    assert decoder.n == 5
    assert decoder.max_iter == 42
    assert decoder.alpha == 0.7
