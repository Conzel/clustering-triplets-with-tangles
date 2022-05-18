from data_generation import generate_planted_hierarchy
import numpy as np


def test_sim_matrix_gen():
    sim = generate_planted_hierarchy(num_classes_exp=2, num_per_class=3,
                                     initial_class_dist=5, class_dist_sim_decrease=1, noise_variance=0)
    N = 2**2 * 3
    res = np.array([[5., 5., 5., 4., 4., 4., 3., 3., 3., 3., 3., 3.],
                    [5., 5., 5., 4., 4., 4., 3., 3., 3., 3., 3., 3.],
                    [5., 5., 5., 4., 4., 4., 3., 3., 3., 3., 3., 3.],
                    [4., 4., 4., 5., 5., 5., 3., 3., 3., 3., 3., 3.],
                    [4., 4., 4., 5., 5., 5., 3., 3., 3., 3., 3., 3.],
                    [4., 4., 4., 5., 5., 5., 3., 3., 3., 3., 3., 3.],
                    [3., 3., 3., 3., 3., 3., 5., 5., 5., 4., 4., 4.],
                    [3., 3., 3., 3., 3., 3., 5., 5., 5., 4., 4., 4.],
                    [3., 3., 3., 3., 3., 3., 5., 5., 5., 4., 4., 4.],
                    [3., 3., 3., 3., 3., 3., 4., 4., 4., 5., 5., 5.],
                    [3., 3., 3., 3., 3., 3., 4., 4., 4., 5., 5., 5.],
                    [3., 3., 3., 3., 3., 3., 4., 4., 4., 5., 5., 5.]])
    assert sim.xs.shape == (N, N)
    assert np.all(sim.xs == res)
    assert sim.ys == [0,0,0,1,1,1,2,2,2,3,3,3]
