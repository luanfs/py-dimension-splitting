####################################################################################
#
# Module for PPM numerical flux computation
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
# Luan da Fonseca Santos - June 2022
# (luan.santos@usp.br)
####################################################################################
import numpy as np

####################################################################################
# PPM flux in x direction
####################################################################################
def numerical_flux_x(q_R, q_L, dq, q6, u_edges, simulation):
    N = simulation.N
    M = simulation.M

    # Numerical fluxes at edges
    f_L = np.zeros((N+7, M+6))# Left
    f_R = np.zeros((N+7, M+6))# Rigth

    # Aux. variables
    F = np.zeros((N+7, M+6))# Numerical flux

    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    c = u_edges*(simulation.dt/simulation.dx)

    # Flux at left edges
    f_L[3:N+4,:] = q_R[2:N+3,:] - c[3:N+4,:]*0.5*(dq[2:N+3,:] - (1.0-(2.0/3.0)*c[3:N+4,:])*q6[2:N+3,:])

    # Flux at right edges
    c = -c
    f_R[3:N+4,:] = q_L[3:N+4,:] + c[3:N+4,:]*0.5*(dq[3:N+4,:] + (1.0-(2.0/3.0)*c[3:N+4,:])*q6[3:N+4,:])

    # F - Formula 1.13 from Collela and Woodward 1984)
    F[u_edges[:,:] >= 0] = f_L[u_edges[:,:] >= 0]
    F[u_edges[:,:] <= 0] = f_R[u_edges[:,:] <= 0]

    return F

####################################################################################
# PPM flux in y direction
####################################################################################
def numerical_flux_y(q_R, q_L, dq, q6, v_edges, simulation):
    N = simulation.N
    M = simulation.M

    # Numerical fluxes at edges
    g_L = np.zeros((N+6, M+7))# Left
    g_R = np.zeros((N+6, M+7))# Rigth

    # Aux. variables
    G = np.zeros((N+6, M+7))# Numerical flux

    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    c = v_edges*(simulation.dt/simulation.dy)

    # Flux at left edges
    g_L[:,3:M+4]= q_R[:,2:M+3] - c[:,3:M+4]*0.5*(dq[:,2:M+3] - (1.0-2.0/3.0*c[:,3:M+4])*q6[:,2:M+3])

    # Flux at right edges
    c = -c
    g_R[:,3:M+4] = q_L[:,3:M+4] + c[:,3:M+4]*0.5*(dq[:,3:M+4] + (1.0-2.0/3.0*c[:,3:M+4])*q6[:,3:M+4])

    # G - Formula 1.13 from Collela and Woodward 1984)
    G[v_edges[:,:] >= 0] = g_L[v_edges[:,:] >= 0]
    G[v_edges[:,:] <= 0] = g_R[v_edges[:,:] <= 0]

    return G

####################################################################################
# Compute the 1d flux operator from PPM using its stencil
# Inputs: Q (average values),  u_edges (zonal velocity at edges)
####################################################################################
def flux_ppm_x_stencil(Q, u_edges, simulation):
    N = simulation.N
    M = simulation.M
    dx = simulation.dx
    dt = simulation.dt

    # CFL at edges - x direction
    c = np.sign(u_edges)*u_edges*dt/dx
    c2 = c*c

    # Stencil coefficients
    a = np.zeros((6, N+7, M+6))
    upositive = u_edges>=0
    unegative = ~upositive

    a[0, upositive] =  c[upositive] - c2[upositive]
    a[0, unegative] =  0.0

    a[1, upositive] = -1.0 - 5.0*c[upositive] + 6.0*c2[upositive]
    a[1, unegative] = -1.0 + 2.0*c[unegative] - c2[unegative] 

    a[2, upositive] =  7.0 + 15.0*c[upositive] - 10.0*c2[upositive]
    a[2, unegative] =  7.0 - 13.0*c[unegative] + 6.0*c2[unegative] 

    a[3, upositive] =  7.0 - 13.0*c[upositive] + 6.0*c2[upositive]
    a[3, unegative] =  7.0 + 15.0*c[unegative] - 10.0*c2[unegative] 

    a[4, upositive] = -1.0 + 2.0*c[upositive] - c2[upositive]
    a[4, unegative] = -1.0 - 5.0*c[unegative] + 6.0*c2[unegative] 

    a[5, upositive] =  0.0
    a[5, unegative] =  c[unegative] - c2[unegative] 

    F = np.zeros((N+7, M+6))

    F[3:N+4,:] = a[0,3:N+4,:]*Q[0:N+1,:] +\
                 a[1,3:N+4,:]*Q[1:N+2,:] +\
                 a[2,3:N+4,:]*Q[2:N+3,:] +\
                 a[3,3:N+4,:]*Q[3:N+4,:] +\
                 a[4,3:N+4,:]*Q[4:N+5,:] +\
                 a[5,3:N+4,:]*Q[5:N+6,:]

    F[3:N+4,:]  = u_edges[3:N+4,:]*F[3:N+4,:]/12.0

    return F

####################################################################################
# Compute the 1d flux operator from PPM using its stencil
# Inputs: Q (average values),  v_edges (zonal velocity at edges)
####################################################################################
def flux_ppm_y_stencil(Q, v_edges, simulation):
    N = simulation.N
    M = simulation.M
    dy = simulation.dy
    dt = simulation.dt

    # CFL at edges - y direction
    #c = v_edges*dt/dy
    c = np.sign(v_edges)*v_edges*dt/dy
    c2 = c*c

    # Stencil coefficients
    a = np.zeros((6, N+6, M+7))
    vpositive = v_edges>=0
    vnegative = ~vpositive

    a[0, vpositive] =  c[vpositive] - c2[vpositive]
    a[0, vnegative] =  0.0

    a[1, vpositive] = -1.0 - 5.0*c[vpositive] + 6.0*c2[vpositive]
    a[1, vnegative] = -1.0 + 2.0*c[vnegative] - c2[vnegative] 

    a[2, vpositive] =  7.0 + 15.0*c[vpositive] - 10.0*c2[vpositive]
    a[2, vnegative] =  7.0 - 13.0*c[vnegative] + 6.0*c2[vnegative] 

    a[3, vpositive] =  7.0 - 13.0*c[vpositive] + 6.0*c2[vpositive]
    a[3, vnegative] =  7.0 + 15.0*c[vnegative] - 10.0*c2[vnegative] 

    a[4, vpositive] = -1.0 + 2.0*c[vpositive] - c2[vpositive]
    a[4, vnegative] = -1.0 - 5.0*c[vnegative] + 6.0*c2[vnegative] 

    a[5, vpositive] =  0.0
    a[5, vnegative] =  c[vnegative] - c2[vnegative] 

    G = np.zeros((N+6, M+7))

    G[:,3:M+4] = a[0,:,3:M+4]*Q[:,0:M+1] +\
                 a[1,:,3:M+4]*Q[:,1:M+2] +\
                 a[2,:,3:M+4]*Q[:,2:M+3] +\
                 a[3,:,3:M+4]*Q[:,3:M+4] +\
                 a[4,:,3:M+4]*Q[:,4:M+5] +\
                 a[5,:,3:M+4]*Q[:,5:M+6]

    G[:,3:M+4]  = v_edges[:,3:M+4]*G[:,3:M+4]/12.0

    return G
