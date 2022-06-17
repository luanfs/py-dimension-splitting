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
