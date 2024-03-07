import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from sklearn.linear_model import LinearRegression
# Assuming fx_pred, fy_pred, fz_pred, tx_pred, ty_pred are defined elsewhere
# and perform similar operations as their MATLAB counterparts

data = pd.read_csv('acels/data/r_z_data_1A.csv')

points = data[['r', 'z']].values  # Input points (r, z)
values_fx = data['Fx'].values  # Values to interpolate for Fx
values_fy = data['Fy'].values
values_fz = data['Fz'].values
values_tx = data['Tx'].values
values_ty = data['Ty'].values

# Determine the convex hull of the existing data points
hull = ConvexHull(points)

# Check if a point is inside the convex hull
def is_inside_hull(point, hull):
    return all(
        (np.dot(eq[:-1], point) + eq[-1]) <= 0
        for eq in hull.equations)

# Define a custom extrapolation function with non-negative constraint
def custom_extrapolate(x, y, values, hull):
    if is_inside_hull(np.array([x, y]), hull):
        # Use interpolation if the point is inside the convex hull
        interpolated_value = griddata(points, values, (x, y), method='nearest')
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
    # Coil center coordinates
    cxi = np.array([-30, 0, 30, -30, 0, 30, -30, 0, 30])
    cyi = np.array([30, 30, 30, 0, 0, 0, -30, -30, -30])
    
    # Initialize matrices
    I = np.zeros(9)
    A = np.zeros((5, 9))
    F_t = np.array([F_x, F_y, F_z, T_x, T_y])
    
    for i, (cx, cy) in enumerate(zip(cxi, cyi)):
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
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
        
        A[:, i] = [np.cos(theta_i) * f_x, np.sin(theta_i) * f_y, f_z,
                   -np.sin(theta_i) * t_x, np.cos(theta_i) * t_y]
    
    # Calculate currents
    I = np.linalg.pinv(A) @ F_t
    return I

# ----------------------------------------------------------------------------
# # Test algorithm
# ----------------------------------------------------------------------------
# x, y, z = 0, 0, 5
# F_x, F_y, F_z = 0, 0, 0.9
# T_x, T_y = 0, 0

# current = matrix_transform(x, y, z, F_x, F_y, F_z, T_x, T_y)
# print(current)
