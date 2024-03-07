import numpy as np
from scipy.spatial import ConvexHull
from acels.matrix_transformation import (is_inside_hull,
                                        matrix_transform)

# Sample data for testing
points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
values = np.array([1, 2, 3, 4])
hull = ConvexHull(points)

def test_is_inside_hull():
    point_inside = np.array([0.5, 0.5])
    assert is_inside_hull(point_inside, hull) == True

    point_outside = np.array([2, 2])
    assert is_inside_hull(point_outside, hull) == False

def test_matrix_transform():
    x, y, z = 0, 0, 5
    F_x, F_y, F_z = 0, 0, 0.9
    T_x, T_y = 0, 0

    expected_current = np.array([-3.03710567e-08, 2.17011115e-01, 4.79640079e-08, 
                                 2.17011186e-01, 7.76728581e-01, 2.17011045e-01, 
                                 -3.03710615e-08, 2.17011115e-01, 4.79640031e-08])

    current = matrix_transform(x, y, z, F_x, F_y, F_z, T_x, T_y)
    print(expected_current)
    print(current)

    # Since we're dealing with floating point numbers, it's better to use np.isclose
    # or np.allclose for comparison instead of direct equality checks
    assert np.allclose(current, expected_current, atol=1e-8)