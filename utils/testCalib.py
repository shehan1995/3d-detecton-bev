import numpy as np

# Define the matrices given in the problem statement
K_02 = np.array([[194.9017, 0.000000e+00, 207.81938, 0.000000e+00],
                 [0.000000e+00, 173.46446, 64.08503, 0.000000e+00],
                 [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])
R_02 = np.array([[9.9999833e-01, -2.7148584e-05, -1.8177480e-03],
                 [2.5708909e-05,  9.9999952e-01, -8.6851866e-04],
                 [1.8178272e-03,  8.6851994e-04,  9.9999785e-01]])
T_02 = np.array([-0.00039439, -0.00040556,  0.0172676 ])
# S_rect_02 = np.array([1.241000e+03, 3.760000e+02])
# R_rect_02 = np.array([[9.999191e-01, 1.228161e-02, -3.316013e-03],
#                       [-1.228209e-02, 9.999246e-01, -1.245511e-04],
#                       [3.314233e-03, 1.652686e-04, 9.999945e-01]])

# Compute P_rect_02_original
R_rect_02_t = np.hstack((R_02, np.expand_dims(-T_02, axis=1)))

temp1 = np.dot(K_02, R_rect_02_t.T)
P_rect_02_original = np.dot(temp1, np.hstack((np.eye(3), np.zeros((3,1)))))

print("P_rect_02_original:\n", P_rect_02_original)
