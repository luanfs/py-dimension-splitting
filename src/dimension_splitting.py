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
from flux import numerical_flux_x, numerical_flux_y, flux_ppm_x_stencil, flux_ppm_y_stencil

####################################################################################
# Compute the 1d flux operator from PPM
# Inputs: Q (average values),  u_edges (velocity at edges)
####################################################################################
def flux_ppm_x(Q, u_edges, simulation):
    N = simulation.N
    M = simulation.M

    # Reconstructs the values of Q using a piecewise parabolic polynomial
    dq, q6, q_L, q_R = rec.ppm_reconstruction_x(Q, simulation)

    # Applies monotonization on the parabolas
    monotonization_1d_x(Q, q_L, q_R, dq, q6, N, simulation.mono)

    # Compute the fluxes
    F = numerical_flux_x(q_R, q_L, dq, q6, u_edges, simulation)

    flux = u_edges[4:N+4,:]*F[4:N+4,:] - u_edges[3:N+3,:]*F[3:N+3,:]

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
    F = np.zeros((N+6, M+6))

    F[3:N+3,:] = -(simulation.dt/simulation.dx)*flux_ppm_x(Q, u_edges, simulation)

    return F

####################################################################################
# Flux operator in x direction
# Inputs: Q (average values),
# u_edges (velocity in x direction at edges)
# Formula 2.7 from Lin and Rood 1996
####################################################################################
def F_operator_stencil(Q, u_edges, simulation):
    N = simulation.N
    M = simulation.M

    flux_x = flux_ppm_x_stencil(Q, u_edges, simulation)

    F = np.zeros((N+6, M+6))
    F[:,:] = -(simulation.dt/simulation.dx)*(flux_x[1:N+7,:] - flux_x[0:N+6,:])

    return F

####################################################################################
# Compute the 1d flux operator from PPM
# Inputs: Q (average values),  u_edges (velocity at edges)
####################################################################################
def flux_ppm_y(Q, v_edges, simulation):
    N = simulation.N
    M = simulation.M

    # Reconstructs the values of Q using a piecewise parabolic polynomial
    dq, q6, q_L, q_R = rec.ppm_reconstruction_y(Q, simulation)

    # Applies monotonization on the parabolas
    monotonization_1d_y(Q, q_L, q_R, dq, q6, M, simulation.mono)

    # Compute the fluxes
    G = numerical_flux_y(q_R, q_L, dq, q6, v_edges, simulation)

    flux = v_edges[:,4:M+4]*G[:,4:M+4] - v_edges[:,3:M+3]*G[:,3:M+3]

    return flux

####################################################################################
# Flux operator in y direction
# Inputs: Q (average values),
# v_edges (velocity in y direction at edges)
# Formula 2.8 from Lin and Rood 1996
####################################################################################
def G_operator(Q, v_edges, simulation):
    N = simulation.N
    M = simulation.M
    G = np.zeros((N+6, M+6))

    G[:, 3:M+3] = -(simulation.dt/simulation.dy)*flux_ppm_y(Q, v_edges, simulation)

    return G

####################################################################################
# Flux operator in y direction
# Inputs: Q (average values),
# v_edges (velocity in y direction at edges)
# Formula 2.8 from Lin and Rood 1996
####################################################################################
def G_operator_stencil(Q, v_edges, simulation):
    N = simulation.N
    M = simulation.M

    flux_y = flux_ppm_y_stencil(Q, v_edges, simulation)

    G = np.zeros((N+6, M+6))
    G[:,:] = -(simulation.dt/simulation.dy)*(flux_y[:,1:M+7] - flux_y[:,0:M+6])

    return G
