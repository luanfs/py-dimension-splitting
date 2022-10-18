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
import reconstruction_1d as rec
from monotonization_1d import monotonization_1d_x, monotonization_1d_y

####################################################################################
# Compute the 1d flux operator
# Inputs: Q (average values),  u_edges (velocity at edges)
####################################################################################
def compute_flux_x(flux_x, Q, u_edges, ax, cx, cx2, simulation):
    N = simulation.N

    # Reconstructs the values of Q using a piecewise parabolic polynomial
    dq, q6, q_L, q_R = rec.ppm_reconstruction_x(Q, simulation)

    # Applies monotonization on the parabolas
    monotonization_1d_x(Q, q_L, q_R, dq, q6, simulation)

    # Compute the fluxes
    numerical_flux_x(Q, q_R, q_L, dq, q6, u_edges, flux_x, ax, cx, cx2, simulation)

####################################################################################
# Routine to compute the correct flux in x direction
####################################################################################
def numerical_flux_x(Q, q_R, q_L, dq, q6, u_edges, flux_x, ax, cx, cx2, simulation):
    if simulation.mono == 1: # Monotonization
        numerical_flux_ppm_x(q_R, q_L, dq, q6, u_edges, flux_x, simulation)

    elif simulation.mono == 0: # No monotonization
        if simulation.fvmethod == 'PPM':
            flux_ppm_x_stencil(Q, u_edges, flux_x, ax, cx, cx2, simulation)

####################################################################################
# PPM flux in x direction
####################################################################################
def numerical_flux_ppm_x(q_R, q_L, dq, q6, u_edges, flux_x, simulation):
    N = simulation.N
    M = simulation.M

    # Numerical fluxes at edges
    f_L = np.zeros((N+7, M+6)) # Left
    f_R = np.zeros((N+7, M+6)) # Right

    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    c = u_edges*(simulation.dt/simulation.dx)

    # Flux at left edges
    f_L[3:N+4,:] = q_R[2:N+3,:] - c[3:N+4,:]*0.5*(dq[2:N+3,:] - (1.0-(2.0/3.0)*c[3:N+4,:])*q6[2:N+3,:])

    # Flux at right edges
    c = -c
    f_R[3:N+4,:] = q_L[3:N+4,:] + c[3:N+4,:]*0.5*(dq[3:N+4,:] + (1.0-(2.0/3.0)*c[3:N+4,:])*q6[3:N+4,:])

    # F - Formula 1.13 from Collela and Woodward 1984)
    flux_x[u_edges[:,:] >= 0] = f_L[u_edges[:,:] >= 0]
    flux_x[u_edges[:,:] <= 0] = f_R[u_edges[:,:] <= 0]

####################################################################################
# Compute the 1d flux operator from PPM using its stencil
# Inputs: Q (average values),  u_edges (zonal velocity at edges)
####################################################################################
def flux_ppm_x_stencil(Q, u_edges, flux_x, ax, cx, cx2, simulation):
    N = simulation.N

    #flux_ppm_x_stencil_coefficients(u_edges, ax, cx, cx2, simulation)

    flux_x[3:N+4,:] = ax[0,3:N+4,:]*Q[0:N+1,:] +\
                      ax[1,3:N+4,:]*Q[1:N+2,:] +\
                      ax[2,3:N+4,:]*Q[2:N+3,:] +\
                      ax[3,3:N+4,:]*Q[3:N+4,:] +\
                      ax[4,3:N+4,:]*Q[4:N+5,:] +\
                      ax[5,3:N+4,:]*Q[5:N+6,:]

    flux_x[3:N+4,:]  = flux_x[3:N+4,:]/12.0

####################################################################################
# Compute the 1d flux operator
# Inputs: Q (average values),  v_edges (velocity at edges)
####################################################################################
def compute_flux_y(flux_y, Q, v_edges, ay, cy, cy2, simulation):
    M = simulation.M

    # Reconstructs the values of Q using a piecewise parabolic polynomial
    dq, q6, q_L, q_R = rec.ppm_reconstruction_y(Q, simulation)

    # Applies monotonization on the parabolas
    monotonization_1d_y(Q, q_L, q_R, dq, q6, simulation)

    # Compute the fluxes
    numerical_flux_y(Q, q_R, q_L, dq, q6, v_edges, flux_y, ay, cy, cy2, simulation)

####################################################################################
# Routine to compute the correct flux in x direction
####################################################################################
def numerical_flux_y(Q, q_R, q_L, dq, q6, v_edges, flux_y, ay, cy, cy2, simulation):
    if simulation.mono == 1: # Monotonization
        numerical_flux_ppm_y(q_R, q_L, dq, q6, v_edges, flux_y, simulation)

    elif simulation.mono == 0: # No monotonization
        if simulation.fvmethod == 'PPM':
            flux_ppm_y_stencil(Q, v_edges, flux_y, ay, cy, cy2, simulation)

####################################################################################
# PPM flux in y direction
####################################################################################
def numerical_flux_ppm_y(q_R, q_L, dq, q6, v_edges, flux_y,  simulation):
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
    flux_y[v_edges[:,:] >= 0] = g_L[v_edges[:,:] >= 0]
    flux_y[v_edges[:,:] <= 0] = g_R[v_edges[:,:] <= 0]

####################################################################################
# Compute the 1d flux operator from PPM using its stencil
# Inputs: Q (average values),  v_edges (zonal velocity at edges)
####################################################################################
def flux_ppm_y_stencil(Q, v_edges, flux_y, ay, cy, cy2, simulation):
    M = simulation.N

    #flux_ppm_y_stencil_coefficients(v_edges, ay, cy, cy2, simulation)

    flux_y[:,3:M+4] = ay[0,:,3:M+4]*Q[:,0:M+1] +\
                      ay[1,:,3:M+4]*Q[:,1:M+2] +\
                      ay[2,:,3:M+4]*Q[:,2:M+3] +\
                      ay[3,:,3:M+4]*Q[:,3:M+4] +\
                      ay[4,:,3:M+4]*Q[:,4:M+5] +\
                      ay[5,:,3:M+4]*Q[:,5:M+6]

    flux_y[:,3:M+4]  = flux_y[:,3:M+4]/12.0
