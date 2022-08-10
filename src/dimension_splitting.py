####################################################################################
# Dimension splitting operators implementation
# Luan da Fonseca Santos - June 2022
#
# References:
# Lin, S., & Rood, R. B. (1996). Multidimensional Flux-Form Semi-Lagrangian
# Transport Schemes, Monthly Weather Review, 124(9), 2046-2070, from
# https://journals.ametsoc.org/view/journals/mwre/124/9/1520-0493_1996_124_2046_mffslt_2_0_co_2.xml
#
###################################################################################

import numpy as np
import reconstruction_1d as rec
from monotonization_1d import monotonization_1d_x, monotonization_1d_y
from flux import numerical_flux_x, numerical_flux_y

####################################################################################
# Compute the 1d flux operator from PPM
# Inputs: Q (average values),  u_edges (velocity at edges)
####################################################################################
def flux_ppm_x(Q, u_edges, simulation):
    N = simulation.N
    M = simulation.M

    # Numerical fluxes at edges
    f_L = np.zeros((N+1, M))# Left
    f_R = np.zeros((N+1, M))# Rigth

    # Aux. variables
    F = np.zeros((N+1, M))# Numerical flux

    # Reconstructs the values of Q using a piecewise parabolic polynomial
    dq, q6, q_L, q_R = rec.ppm_reconstruction_x(Q[:,2:M+2], simulation)

    # Applies monotonization on the parabolas
    monotonization_1d_x(Q[:,2:M+2], q_L, q_R, dq, q6, N, simulation.mono)

    # Compute the fluxes
    numerical_flux_x(F, f_R, f_L, q_R, q_L, dq, q6, u_edges, simulation)

    flux = u_edges[1:N+1,:]*F[1:N+1,:] - u_edges[0:N,:]*F[0:N,:]
    return flux

####################################################################################
# Compute the 1d flux operator from PPM
# Inputs: Q (average values),  u_edges (velocity at edges)
####################################################################################
def flux_ppm_y(Q, v_edges, simulation):
    N = simulation.N
    M = simulation.M

    # Numerical fluxes at edges
    g_L = np.zeros((N, M+1))# Left
    g_R = np.zeros((N, M+1))# Rigth

    # Aux. variables
    G = np.zeros((N, M+1))# Numerical flux

    # Reconstructs the values of Q using a piecewise parabolic polynomial
    dq, q6, q_L, q_R = rec.ppm_reconstruction_y(Q[2:N+2,:], simulation)

    # Applies monotonization on the parabolas
    monotonization_1d_y(Q[2:N+2,:], q_L, q_R, dq, q6, M, simulation.mono)

    # Compute the fluxes
    numerical_flux_y(G, g_R, g_L, q_R, q_L, dq, q6, v_edges, simulation)
    flux = v_edges[:,1:M+1]*G[:,1:M+1] - v_edges[:,0:M]*G[:,0:M]
    return flux

####################################################################################
# Flux operator in x direction
# Inputs: Q (average values),
# u_edges (velocity in x direction at edges)
# Formula 2.7 from Lin and Rood 1996
####################################################################################
def F_operator(Q, u_edges, simulation):
    N = simulation.N
    M = simulation.M
    F_operator = np.zeros((N+5, M+5))

    F_operator[2:N+2, 2:M+2] = -(simulation.dt/simulation.dx)*flux_ppm_x(Q, u_edges, simulation)

    # Periodic boundary conditions
    # x direction
    F_operator[N+2:N+5,:] = F_operator[2:5,:]
    F_operator[0:2,:]     = F_operator[N:N+2,:]
    # y direction
    F_operator[:,M+2:M+5] = F_operator[:,2:5]
    F_operator[:,0:2]     = F_operator[:,M:M+2]
    return F_operator

####################################################################################
# Flux operator in y direction
# Inputs: Q (average values),
# v_edges (velocity in y direction at edges)
# Formula 2.8 from Lin and Rood 1996
####################################################################################
def G_operator(Q, v_edges, simulation):
    N = simulation.N
    M = simulation.M
    G_operator = np.zeros((N+5, M+5))

    G_operator[2:N+2, 2:M+2] = -(simulation.dt/simulation.dy)*flux_ppm_y(Q, v_edges, simulation)

    # Periodic boundary conditions
    # x direction
    G_operator[N+2:N+5,:] = G_operator[2:5,:]
    G_operator[0:2,:]     = G_operator[N:N+2,:]
    # y direction
    G_operator[:,M+2:M+5] = G_operator[:,2:5]
    G_operator[:,0:2]     = G_operator[:,M:M+2]
    return G_operator

####################################################################################
# Flux operator in x direction
# Inputs: Q (average values),
# u_edges (velocity in x direction at edges)
# Formula 2.7 from Lin and Rood 1996
####################################################################################
def F_operator2(Q, u_edges, simulation):
    N = simulation.N
    M = simulation.M

    flux_x = flux_ppm_x_stencil(Q, u_edges, simulation)
    F_operator = np.zeros((N+5, M+5))
    F_operator[2:N+2,2:M+2] = -(simulation.dt/simulation.dx)*(flux_x[1:N+1,:] - flux_x[0:N,:])

    # Periodic boundary conditions
    # x direction
    F_operator[N+2:N+5,:] = F_operator[2:5,:]
    F_operator[0:2,:]     = F_operator[N:N+2,:]
    # y direction
    F_operator[:,M+2:M+5] = F_operator[:,2:5]
    F_operator[:,0:2]     = F_operator[:,M:M+2]
    return F_operator

####################################################################################
# Flux operator in y direction
# Inputs: Q (average values),
# v_edges (velocity in y direction at edges)
# Formula 2.8 from Lin and Rood 1996
####################################################################################
def G_operator2(Q, v_edges, simulation):
    N = simulation.N
    M = simulation.M

    flux_y = flux_ppm_y_stencil(Q, v_edges, simulation)
    G_operator = np.zeros((N+5, M+5))
    G_operator[2:N+2,2:M+2] = -(simulation.dt/simulation.dy)*(flux_y[:,1:M+1] - flux_y[:,0:M])

    # Periodic boundary conditions
    # x direction
    G_operator[N+2:N+5,:] = G_operator[2:5,:]
    G_operator[0:2,:]     = G_operator[N:N+2,:]
    # y direction
    G_operator[:,M+2:M+5] = G_operator[:,2:5]
    G_operator[:,0:2]     = G_operator[:,M:M+2]

    return G_operator2
    
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
    c = u_edges*dt/dx
    c2 = c*c

    # Stencil coefficients
    a = np.zeros((6, N+1, M))
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

    F = np.zeros((N+1, M))
    F[1:N+1,:] = a[0,1:N+1,:]*Q[0:N  ,2:M+2] + a[1,1:N+1,:]*Q[1:N+1,2:M+2] + a[2,1:N+1,:]*Q[2:N+2,2:M+2]\
               + a[3,1:N+1,:]*Q[3:N+3,2:M+2] + a[4,1:N+1,:]*Q[4:N+4,2:M+2] + a[5,1:N+1,:]*Q[5:N+5,2:M+2]
    F[0,:] = F[N,:] #boundary condition
    F = u_edges*F/12.0
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
    c = v_edges*dt/dy
    c2 = c*c

    # Stencil coefficients
    a = np.zeros((6, N, M+1))
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

    G = np.zeros((N, M+1))
    G[:,1:M+1] = a[0,:,1:M+1]*Q[2:N+2,0:M  ] + a[1,:,1:M+1]*Q[2:N+2,1:M+1] + a[2,:,1:M+1]*Q[2:N+2,2:M+2]\
               + a[3,:,1:M+1]*Q[2:N+2,3:M+3] + a[4,:,1:M+1]*Q[2:N+2,4:M+4] + a[5,:,1:M+1]*Q[2:N+2,5:M+5]
    G[:,0] = G[:,M] #boundary condition
    G = v_edges*G/12.0
    return G
