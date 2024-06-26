import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from sklearn.linear_model import LinearRegression

data = pd.read_csv("acels/data/r_z_data_1A.csv")

points = data[["r", "z"]].values  # Input points (r, z)
values_fx = data["Fx"].values  # Values to interpolate for Fx
values_fy = data["Fy"].values
values_fz = data["Fz"].values
values_tx = data["Tx"].values
values_ty = data["Ty"].values

# Determine the convex hull of the existing data points
hull = ConvexHull(points)


# Check if a point is inside the convex hull
def is_inside_hull(point, hull):
    return all((np.dot(eq[:-1], point) + eq[-1]) <= 0 for eq in hull.equations)


# Define a custom extrapolation function with non-negative constraint
def custom_extrapolate(x, y, values, hull):
    if is_inside_hull(np.array([x, y]), hull):
        # Use interpolation if the point is inside the convex hull
        interpolated_value = griddata(points, values, (x, y), method="nearest")
    else:
        # Use extrapolation (linear regression) if outside
        model = LinearRegression()
        model.fit(points, values)
        interpolated_value = model.predict([[x, y]])[0]

    # Ensure the output is not negative
    return max(interpolated_value, 1e-8)


# Update your interpolation functions to use custom_extrapolate
def interpolate_fx(r, z):
    return custom_extrapolate(r, z, values_fx, hull)


def interpolate_fy(r, z):
    return custom_extrapolate(r, z, values_fy, hull)


def interpolate_fz(r, z):
    return custom_extrapolate(r, z, values_fz, hull)


def interpolate_tx(r, z):
    return custom_extrapolate(r, z, values_tx, hull)


def interpolate_ty(r, z):
    return custom_extrapolate(r, z, values_ty, hull)


def matrix_transform(x, y, z, F_x, F_y, F_z, T_x, T_y):
    """
    Calculates the currents needed in coils to achieve desired force and torque on a point.

    ### Parameters:
    - x, y, z (float): Coordinates of the target point.
    - F_x, F_y, F_z (float): Components of the desired force at the target point.
    - T_x, T_y (float): Components of the desired torque at the target point.

    ### Returns:
    - I (np.ndarray): Array of currents for each coil to achieve the desired force and torque.

    ### Algorithm:
    1. Compute the distance and angle from each coil to the target point.
    2. Use interpolation functions to calculate forces and torques for these distances and angles.
    3. Formulate and solve a linear system to find the currents.
    """
    # Coil center coordinates
    cxi = np.array([-30, 0, 30, -30, 0, 30, -30, 0, 30])
    cyi = np.array([30, 30, 30, 0, 0, 0, -30, -30, -30])

    # Initialize matrices
    I = np.zeros(9)
    A = np.zeros((5, 9))
    F_t = np.array([F_x, F_y, F_z, T_x, T_y])

    for i, (cx, cy) in enumerate(zip(cxi, cyi)):
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        # print(f"R: {r}")
        theta_i = np.arctan2(y - cy, x - cx)

        f_x = interpolate_fx(r, z)
        f_y = interpolate_fy(r, z)
        f_z = interpolate_fz(r, z)
        t_x = interpolate_tx(r, z)
        t_y = interpolate_ty(r, z)

        # print(f"Force X: {f_x}")
        # print(f"Force Y: {f_y}")
        # print(f"Force Z: {f_z}\n")

        A[:, i] = [
            np.cos(theta_i) * f_x,
            np.sin(theta_i) * f_y,
            f_z,
            -np.sin(theta_i) * t_x,
            np.cos(theta_i) * t_y,
        ]

    # Calculate currents
    I = np.linalg.pinv(A) @ F_t
    return I


# ----------------------------------------------------------------------------
# # Test algorithm
# ----------------------------------------------------------------------------
# if __name__ == "__main__":
#     x, y, z = 9, 5, 10
#     F_x, F_y, F_z = 0, 0, 0.9
#     T_x, T_y = 0, 0

#     current = matrix_transform(x, y, z, F_x, F_y, F_z, T_x, T_y)

#     for i in current:
#         print(f"{i},")
