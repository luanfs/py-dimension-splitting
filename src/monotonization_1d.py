####################################################################################
#
# Piecewise Parabolic Method (PPM) monotonization module
# Luan da Fonseca Santos - June 2022
#
# References:
# -  Phillip Colella, Paul R Woodward, The Piecewise Parabolic Method (PPM) for gas-dynamical simulations,
# Journal of Computational Physics, Volume 54, Issue 1, 1984, Pages 174-201, ISSN 0021-9991,
# https://doi.org/10.1016/0021-9991(84)90143-8.
#
# -  Carpenter , R. L., Jr., Droegemeier, K. K., Woodward, P. R., & Hane, C. E. (1990).
# Application of the Piecewise Parabolic Method (PPM) to Meteorological Modeling, Monthly Weather Review, 118(3),
# 586-612. Retrieved Mar 31, 2022,
# from https://journals.ametsoc.org/view/journals/mwre/118/3/1520-0493_1990_118_0586_aotppm_2_0_co_2.xml
#
####################################################################################

import numpy as np

###################################################################################
# Apply the 1D PPM monotonization in x direction
####################################################################################
def monotonization_1d_x(Q, q_L, q_R, dq, q6, N, mono):
    if mono == 0:
        return
    elif mono == 1:
        # In each cell, check if Q is a local maximum
        # See First equation in formula 1.10 from Collela and Woodward 1984
        local_maximum = (q_R[2:N+2,:]-Q[2:N+2,:])*(Q[2:N+2,:]-q_L[2:N+2,:])<=0

        # In this case (local maximum), the interpolation is a constant equal to Q
        q_R[2:N+2,:][local_maximum] = Q[2:N+2,:][local_maximum]
        q_L[2:N+2,:][local_maximum] = Q[2:N+2,:][local_maximum]

        # Check overshot
        overshoot  = (abs(dq[2:N+2,:]) < abs(q6[2:N+2,:]))

        # Move left
        move_left  = (q_R[2:N+2,:]-q_L[2:N+2,:])*(Q[2:N+2,:]-0.5*(q_R[2:N+2,:]+q_L[2:N+2,:])) > ((q_R[2:N+2,:]-q_L[2:N+2,:])**2)/6.0

        # Move right
        move_right = (-((q_R[2:N+2,:]-q_L[2:N+2,:])**2)/6.0 > (q_R[2:N+2,:]-q_L[2:N+2,:])*(Q[2:N+2,:]-0.5*(q_R[2:N+2,:]+q_L[2:N+2,:])) )

        overshoot_move_left  = np.logical_and(overshoot, move_left)
        overshoot_move_right = np.logical_and(overshoot, move_right)

        q_L[2:N+2,:][overshoot_move_left]  = 3.0*Q[2:N+2,:][overshoot_move_left]  - 2.0*q_R[2:N+2,:][overshoot_move_left]
        q_R[2:N+2,:][overshoot_move_right] = 3.0*Q[2:N+2,:][overshoot_move_right] - 2.0*q_L[2:N+2,:][overshoot_move_right]

        # Update the polynomial coefs
        dq[2:N+2,:] = q_R[2:N+2,:] - q_L[2:N+2,:]
        q6[2:N+2,:] = 6*Q[2:N+2,:] - 3*(q_R[2:N+2,:] + q_L[2:N+2,:])

        # Periodic boundary conditions
        q_L[N+2:N+5,:] = q_L[2:5,:]
        q_L[0:2,:]     = q_L[N:N+2,:]

        q_R[N+2:N+5,:] = q_R[2:5,:]
        q_R[0:2,:]     = q_R[N:N+2,:]

        q6[N+2:N+5,:] = q6[2:5,:]
        q6[0:2,:]     = q6[N:N+2,:]

        dq[N+2:N+5,:] = dq[2:5,:]
        dq[0:2,:]     = dq[N:N+2,:]

####################################################################################
# Apply the 1D PPM monotonization in y direction
####################################################################################
def monotonization_1d_y(Q, q_L, q_R, dq, q6, M, mono):
    if mono == 0:
        return
    elif mono == 1:
        # In each cell, check if Q is a local maximum
        # See First equation in formula 1.10 from Collela and Woodward 1984
        local_maximum = (q_R[:,2:M+2]-Q[:,2:M+2])*(Q[:,2:M+2]-q_L[:,2:M+2])<=0

        # In this case (local maximum), the interpolation is a constant equal to Q
        q_R[:,2:M+2][local_maximum] = Q[:,2:M+2][local_maximum]
        q_L[:,2:M+2][local_maximum] = Q[:,2:M+2][local_maximum]

        # Check overshot
        overshoot  = (abs(dq[:,2:M+2]) < abs(q6[:,2:M+2]))

        # Move left
        move_left  = (q_R[:,2:M+2]-q_L[:,2:M+2])*(Q[:,2:M+2]-0.5*(q_R[:,2:M+2]+q_L[:,2:M+2])) > ((q_R[:,2:M+2]-q_L[:,2:M+2])**2)/6.0

        # Move right
        move_right = (-((q_R[:,2:M+2]-q_L[:,2:M+2])**2)/6.0 > (q_R[:,2:M+2]-q_L[:,2:M+2])*(Q[:,2:M+2]-0.5*(q_R[:,2:M+2]+q_L[:,2:M+2])) )

        overshoot_move_left  = np.logical_and(overshoot, move_left)
        overshoot_move_right = np.logical_and(overshoot, move_right)

        q_L[:,2:M+2][overshoot_move_left]  = 3.0*Q[:,2:M+2][overshoot_move_left]  - 2.0*q_R[:,2:M+2][overshoot_move_left]
        q_R[:,2:M+2][overshoot_move_right] = 3.0*Q[:,2:M+2][overshoot_move_right] - 2.0*q_L[:,2:M+2][overshoot_move_right]

        # Update the polynomial coefs
        dq[:,2:M+2] = q_R[:,2:M+2] - q_L[:,2:M+2]
        q6[:,2:M+2] = 6*Q[:,2:M+2] - 3*(q_R[:,2:M+2] + q_L[:,2:M+2])

        # Periodic boundary conditions
        q_L[:,M+2:M+5] = q_L[:,2:5]
        q_L[:,0:2]     = q_L[:,M:M+2]

        q_R[:,M+2:M+5] = q_R[:,2:5]
        q_R[:,0:2]     = q_R[:,M:M+2]

        q6[:,M+2:M+5] = q6[:,2:5]
        q6[:,0:2]     = q6[:,M:M+2]

        dq[:,M+2:M+5] = dq[:,2:5]
        dq[:,0:2]     = dq[:,M:M+2]
