"""Regression tests for the two finite-batch normalizations.

The broad mathematical identities live in ``msflow.checks``.  These tests
focus on conventions that are easy to change accidentally while refactoring:
permutation equivariance, the Deng self mask and the Proxy's separate blocks.
"""

import numpy as np

from msflow import (
    Emp,
    deng_official_batch,
    deng_official_batch_details,
    get_kernel,
    make_drift,
    epsilon_of_sigma,
    sinkhorn_drift,
    sinkhorn_proxy_details,
)


def _clouds():
    generated = np.array([
        [-1.2, 0.3], [0.1, -0.8], [1.5, 1.1], [2.2, -0.2],
    ])
    positive = np.array([
        [-2.0, 1.4], [-0.4, 0.9], [0.8, -1.7], [2.7, 0.6], [1.2, 2.1],
    ])
    return generated, positive



def test_deng_official_matches_the_paper_pseudocode_literally():
    """Independent transcription of Appendix A.1, including pos-first order."""
    x, y_pos = _clouds()
    temperature = 0.73
    got = deng_official_batch_details(x, y_pos, temperature=temperature)

    dist_pos = np.linalg.norm(x[:, None, :] - y_pos[None, :, :], axis=-1)
    dist_neg = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    dist_neg = dist_neg + np.eye(len(x)) * 1e6
    logits = np.concatenate((-dist_pos / temperature,
                             -dist_neg / temperature), axis=1)
    logits_row = logits - logits.max(axis=1, keepdims=True)
    logits_col = logits - logits.max(axis=0, keepdims=True)
    A_row = np.exp(logits_row) / np.exp(logits_row).sum(axis=1, keepdims=True)
    A_col = np.exp(logits_col) / np.exp(logits_col).sum(axis=0, keepdims=True)
    A = np.sqrt(A_row * A_col)
    A_pos, A_neg = A[:, :len(y_pos)], A[:, len(y_pos):]
    W_pos = A_pos * A_neg.sum(axis=1, keepdims=True)
    W_neg = A_neg * A_pos.sum(axis=1, keepdims=True)
    expected = W_pos @ y_pos - W_neg @ x

    np.testing.assert_allclose(got.row_attention, A_row, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(got.column_attention, A_col, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(got.affinity, A, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(got.positive_weights, W_pos, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(got.negative_weights, W_neg, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(got.field, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        deng_official_batch(x, y_pos, temperature=temperature), expected,
        rtol=2e-14, atol=2e-14,
    )


def test_deng_official_has_only_algorithm2_normalizations():
    field = make_drift("deng_official", "Laplacian", 0.73)
    assert field.name == "deng_official"
    assert field.meta["temperature"] == 0.73
    assert field.meta["self_mask_distance"] == 1e6
    assert field.meta["affinity_product_floor"] == 0.0
    assert field.meta["feature_normalization"] is False
    assert field.meta["drift_normalization"] is False
    assert field.meta["output"] == "raw_algorithm2_field"


def test_deng_official_is_permutation_and_rigid_motion_equivariant():
    x, y_pos = _clouds()
    temperature = 0.73
    expected = deng_official_batch(x, y_pos, temperature=temperature)

    pos_order = np.array([3, 0, 4, 1, 2])
    np.testing.assert_allclose(
        deng_official_batch(x, y_pos[pos_order], temperature=temperature),
        expected, rtol=5e-13, atol=5e-13,
    )
    query_order = np.array([2, 0, 3, 1])
    np.testing.assert_allclose(
        deng_official_batch(x[query_order], y_pos, temperature=temperature),
        expected[query_order], rtol=5e-13, atol=5e-13,
    )

    shift = np.array([3.4, -2.1])
    np.testing.assert_allclose(
        deng_official_batch(x + shift, y_pos + shift, temperature=temperature),
        expected, rtol=5e-13, atol=5e-13,
    )
    angle = 0.41
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    np.testing.assert_allclose(
        deng_official_batch(x @ rotation.T, y_pos @ rotation.T,
                            temperature=temperature),
        expected @ rotation.T, rtol=6e-13, atol=6e-13,
    )


def test_deng_official_is_not_the_population_row_field():
    generated, positive = _clouds()
    p, q = Emp(positive), Emp(generated)
    official = make_drift("deng_official", "Laplacian", 0.73)(p, q)
    population = make_drift("mean_shift", "Laplacian", 0.73)(p, q)
    assert np.linalg.norm(official - population) > 1e-3


def test_deng_official_self_mask_is_the_paper_ablation():
    generated, _ = _clouds()
    unmasked = deng_official_batch(
        generated, generated, temperature=0.73, self_mask=False,
    )
    masked = deng_official_batch(
        generated, generated, temperature=0.73, self_mask=True,
    )
    np.testing.assert_allclose(unmasked, 0.0, atol=2e-14)
    assert np.linalg.norm(masked) > 1e-3

    # The paper conditions the mask on y_neg being x, not merely on a square
    # distance matrix.  A distinct same-size negative batch must stay unmasked.
    distinct_negative = generated + np.array([0.4, -0.7])
    explicit = deng_official_batch_details(
        generated, generated, distinct_negative,
        temperature=0.73, self_mask=True,
    )
    explicit_unmasked = deng_official_batch_details(
        generated, generated, distinct_negative,
        temperature=0.73, self_mask=False,
    )
    np.testing.assert_allclose(explicit.field, explicit_unmasked.field)


def test_canonical_sinkhorn_proxy_vanishes_on_identical_clouds():
    generated, _ = _clouds()
    cloud = Emp(generated)
    result = sinkhorn_proxy_details(cloud, cloud, get_kernel("Gaussian", 0.9))

    np.testing.assert_allclose(result.positive_affinity, result.negative_affinity)
    np.testing.assert_allclose(result.field, 0.0, atol=2e-14)


def test_sinkhorn_proxy_retains_cross_weight_mobility():
    generated, positive = _clouds()
    q, p = Emp(generated), Emp(positive)
    result = sinkhorn_proxy_details(p, q, get_kernel("Gaussian", 0.9))

    np.testing.assert_allclose(
        result.field,
        result.mobility[:, None] * result.core,
        rtol=2e-14,
        atol=2e-14,
    )
    assert not np.allclose(result.mobility, 1.0)


def test_proxy_floor_does_not_resurrect_an_excluded_diagonal():
    generated, positive = _clouds()
    result = sinkhorn_proxy_details(
        Emp(positive), Emp(generated), get_kernel("Gaussian", 0.9),
        exclude_self=True, product_floor=1e-4,
    )
    np.testing.assert_allclose(np.diag(result.negative_affinity), 0.0)


def test_canonical_batch_fields_reject_nonuniform_empirical_weights():
    generated, positive = _clouds()
    weighted_q = Emp(generated, np.array([0.1, 0.2, 0.3, 0.4]))
    p = Emp(positive)
    with np.testing.assert_raises_regex(ValueError, "uniform samples"):
        sinkhorn_proxy_details(p, weighted_q, get_kernel("Gaussian", 0.9))
    with np.testing.assert_raises_regex(ValueError, "uniform samples"):
        make_drift("deng_official", "Laplacian", 1.0)(p, weighted_q)


def test_factories_report_the_normalization_scope():
    official = make_drift("deng_official", "Laplacian", 1.0)
    proxy = make_drift("sinkhorn_proxy", "Gaussian", 0.7)

    assert official.meta["row_scope"] == "joint_positive_negative"
    assert official.meta["temperatures"] == (1.0,)
    assert official.meta["output"] == "raw_algorithm2_field"
    assert proxy.meta["row_scope"] == "separate"
    assert proxy.meta["tau"] == 2 * 0.7**2
    assert proxy.meta["source_exact"] is True

    assert make_drift("deng_official", "Laplacian", 99.0).meta["temperature"] == 99.0


def test_sinkhorn_correction_count_has_no_silent_truncation():
    assert sinkhorn_drift(0.8, K=-0.2).meta["K"] == -1
    assert sinkhorn_drift(0.8, K=np.inf).meta["K"] == -1
    with np.testing.assert_raises_regex(ValueError, "non-negative integer"):
        sinkhorn_drift(0.8, K=2.9)
    with np.testing.assert_raises_regex(ValueError, "NaN"):
        sinkhorn_drift(0.8, K=np.nan)


def test_measure_and_bandwidth_guards_match_probability_assumptions():
    generated, _ = _clouds()
    with np.testing.assert_raises_regex(ValueError, "sum to one"):
        Emp(generated, np.ones(len(generated)))
    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        Emp(generated, np.array([0.4, 0.4, 0.3, -0.1]))
    for bad_sigma in (0.0, -1.0, np.nan, np.inf):
        with np.testing.assert_raises_regex(ValueError, "strictly positive"):
            epsilon_of_sigma(bad_sigma)
