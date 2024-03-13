import numpy as np
from scipy.spatial import ConvexHull

from acels.matrix_transformation import is_inside_hull, matrix_transform

# Sample data for testing
points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
values = np.array([1, 2, 3, 4])
hull = ConvexHull(points)


def test_is_inside_hull():
    point_inside = np.array([0.5, 0.5])
    assert is_inside_hull(point_inside, hull) == True

    point_outside = np.array([2, 2])
    assert is_inside_hull(point_outside, hull) == False


def test_matrix_transform_1():
    x, y, z = 0, 0, 5
    F_x, F_y, F_z = 0, 0, 0.9
    T_x, T_y = 0, 0

    expected_current = np.array(
        [
            -3.03710567e-08,
            2.17011115e-01,
            4.79640079e-08,
            2.17011186e-01,
            7.76728581e-01,
            2.17011045e-01,
            -3.03710615e-08,
            2.17011115e-01,
            4.79640031e-08,
        ]
    )

    current = matrix_transform(x, y, z, F_x, F_y, F_z, T_x, T_y)
    print(expected_current)
    print(current)

    # Since we're dealing with floating point numbers, it's better to use np.isclose
    # or np.allclose for comparison instead of direct equality checks
    assert np.allclose(current, expected_current, atol=1e-8)


def test_matrix_transform_2():
    x, y, z = 3, -7, 5
    F_x, F_y, F_z = 0, 0, 0.9
    T_x, T_y = 0, 0

    expected_current = np.array(
        [0.22792731, 0.05278101,
         -0.20649907, 0.10354569,
         0.55229808, 0.65896062,
         0.11894888, 0.54260611,
         0.22258858])

    current = matrix_transform(x, y, z, F_x, F_y, F_z, T_x, T_y)
    print(expected_current)
    print(current)

    assert np.allclose(current, expected_current, atol=1e-8)
    
def test_matrix_transform_3():
    x, y, z = -8, 13, 7
    F_x, F_y, F_z = 0, 0, 0.9
    T_x, T_y = 0, 0

    expected_current = np.array(
        [0.28597859, 0.69259939, 0.07923026, 0.26186678,
         1.10842669, 0.04240518, -0.04954331, -0.07628679, -0.18409502]
        )

    current = matrix_transform(x, y, z, F_x, F_y, F_z, T_x, T_y)
    print(expected_current)
    print(current)

    assert np.allclose(current, expected_current, atol=1e-8)
    
    
def test_matrix_transform_4():
    x, y, z = 9, 5, 10
    F_x, F_y, F_z = 0, 0, 0.9
    T_x, T_y = 0, 0

    expected_current = np.array(
        [0.3466952792483289,
         0.466670803751715,
         0.2546862306697588,
         0.019373306719751945,
         0.928259311433401,
         1.7881959235105478,
         0.35054386687234645,
         -0.1477594537909603,
         -0.16009212299596184]
        )

    current = matrix_transform(x, y, z, F_x, F_y, F_z, T_x, T_y)
    print(expected_current)
    print(current)

    assert np.allclose(current, expected_current, atol=1e-8)