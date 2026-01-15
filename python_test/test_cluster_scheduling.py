import numpy as np
import pytest
from ldpc.codes import rep_code, hamming_code
from ldpc.bp_decoder import BpDecoder


def test_reset_and_initialise_log_domain_bp():
    H = rep_code(5)
    bpd = BpDecoder(H, error_rate=0.1, max_iter=1)

    llr = np.array([2.0, 2.0, -2.0, 2.0, 2.0])
    bpd.initialise_log_domain_bp(llr)
    assert np.allclose(bpd.log_prob_ratios, llr)

    bpd.log_prob_ratios = llr * 3
    assert np.allclose(bpd.log_prob_ratios, llr * 3)

    bpd.reset()
    assert np.allclose(bpd.log_prob_ratios, np.zeros(bpd.bit_count))
    assert bpd.converge == False


def test_initialise_log_domain_bp_rejects_wrong_length():
    H = rep_code(5)
    bpd = BpDecoder(H, error_rate=0.1, max_iter=1)
    with pytest.raises(ValueError):
        bpd.initialise_log_domain_bp(np.array([1.0, 2.0]))


@pytest.mark.parametrize("pcm", [rep_code(5), rep_code(9), hamming_code(3)])
def test_decode_cluster_converges_to_valid_codeword(pcm):
    H = pcm
    m, n = H.shape
    bpd = BpDecoder(H, error_rate=0.1, max_iter=1)

    rng = np.random.default_rng(42)
    llr = rng.choice([-3.0, 3.0], size=n)

    bpd.reset()
    bpd.initialise_log_domain_bp(llr)

    # Sweep every check cluster (one check per cluster) several times -- enough
    # rounds for a small code to converge to a valid codeword.
    for _ in range(20):
        for check_idx in range(m):
            bpd.decode_cluster([check_idx])

    decoding = (bpd.log_prob_ratios < 0).astype(np.uint8)
    syndrome = H @ decoding % 2
    assert np.all(syndrome == 0)


def test_decode_cluster_with_multiple_checks_per_call():
    H = rep_code(6)
    m, n = H.shape
    bpd = BpDecoder(H, error_rate=0.1, max_iter=1)

    llr = np.array([3.0, 3.0, -3.0, 3.0, 3.0, 3.0])
    bpd.reset()
    bpd.initialise_log_domain_bp(llr)

    updated = bpd.decode_cluster(list(range(m)))
    assert updated.shape == (n,)
    assert np.array_equal(updated, bpd.log_prob_ratios)


def test_decode_cluster_rejects_out_of_range_index():
    H = rep_code(5)
    m, _ = H.shape
    bpd = BpDecoder(H, error_rate=0.1, max_iter=1)
    bpd.reset()
    bpd.initialise_log_domain_bp(np.ones(bpd.bit_count))
    with pytest.raises(ValueError):
        bpd.decode_cluster([m])


def test_get_residuals_shape_and_nonnegativity():
    H = rep_code(7)
    m, n = H.shape
    bpd = BpDecoder(H, error_rate=0.1, max_iter=1)

    rng = np.random.default_rng(0)
    llr = rng.choice([-2.0, 2.0], size=n)
    bpd.reset()
    bpd.initialise_log_domain_bp(llr)

    residuals = bpd.get_residuals()
    assert residuals.shape == (m,)
    assert np.all(residuals >= 0)

    # After updating a cluster, its own residual should drop towards zero
    # (it was just brought up to date).
    bpd.decode_cluster([0])
    residuals_after = bpd.get_residuals()
    assert residuals_after[0] <= residuals[0] + 1e-9


def test_original_decode_and_schedule_options_unaffected():
    """The pre-existing decode()/schedule API must keep working exactly as before."""
    H = rep_code(5)
    m, n = H.shape
    bpd = BpDecoder(H, error_rate=0.1, max_iter=10, schedule="parallel")

    syndrome = np.zeros(m, dtype=np.uint8)
    out = bpd.decode(syndrome)
    assert out.shape == (n,)
    assert bpd.converge == True

    for value in ["parallel", "serial", "serial_relative", "cluster"]:
        bpd.schedule = value
        assert bpd.schedule == value


def test_m2i2_scheduler_returns_valid_check_indices():
    H = rep_code(5)
    bpd = BpDecoder(H, error_rate=0.1, max_iter=1)

    P = [[0, 0, -1], [-1, 0, 0]]
    schedule = bpd.m2i2_scheduler(P, code_rate=0.5, EbN0=2.0, max_iterations=6)

    assert len(schedule) <= 6
    for idx in schedule:
        assert 0 <= idx < len(P)


def test_m2i2_scheduler_rejects_empty_matrix():
    H = rep_code(5)
    bpd = BpDecoder(H, error_rate=0.1, max_iter=1)
    with pytest.raises(ValueError):
        bpd.m2i2_scheduler([], code_rate=0.5, EbN0=2.0, max_iterations=4)
