import numpy as np

from run_scale_mass import antithetic_stein_pseudovalues, stein_pseudovalues


def test_stein_pseudovalues_average_to_off_diagonal_estimator():
    plus = np.array([[1., 2.], [3., -1.], [2., 4.], [-2., 3.]])
    baseline = np.array([.2, -.3])
    gram = np.array([
        [2., .5, -1., .2], [.5, 3., .7, -.4],
        [-1., .7, 4., .9], [.2, -.4, .9, 5.],
    ])
    tau = .4
    pseudo = stein_pseudovalues(plus, baseline, gram, tau)
    delta = plus - baseline
    expected = np.mean([
        delta[i] * delta[j] * gram[i, j] / tau**2
        for i in range(4) for j in range(4) if i != j
    ], axis=0)
    np.testing.assert_allclose(pseudo.mean(0), expected)


def test_antithetic_stein_cancels_even_response_component():
    odd = np.array([[1., 2.], [3., -1.], [2., 4.], [-2., 3.]])
    even = np.array([[8., -2.], [1., 5.], [-4., 3.], [7., 9.]])
    baseline = np.array([.2, -.3])
    plus = baseline + odd + even
    minus = baseline - odd + even
    gram = np.array([
        [2., .5, -1., .2], [.5, 3., .7, -.4],
        [-1., .7, 4., .9], [.2, -.4, .9, 5.],
    ])
    tau = .4
    actual = antithetic_stein_pseudovalues(plus, minus, gram, tau)
    expected = stein_pseudovalues(odd, np.zeros(2), gram, tau)
    np.testing.assert_allclose(actual, expected)
