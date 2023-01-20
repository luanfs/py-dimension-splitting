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
from cfl import cfl_x, cfl_y

####################################################################################
# Compute the 1d flux operator
# Inputs: Q (average values),  u_edges (velocity at edges)
# ax (stencil coefficients), cx, cx2 (CFL, CFL^2 in x direction)
# Output: flux_x (flux in x direction)
####################################################################################
def compute_flux_x(Q, px, cx, simulation):
    N = simulation.N

    # Reconstructs the values of Q using a piecewise parabolic polynomial
    rec.ppm_reconstruction_x(Q, px, simulation)

    # Compute the fluxes
    numerical_flux_ppm_x(px, cx, simulation)

####################################################################################
# PPM flux in x direction
####################################################################################
def numerical_flux_ppm_x(px, cx, simulation):
    N = simulation.N
    M = simulation.M
    ng = simulation.ng
    i0 = simulation.i0
    iend = simulation.iend

    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    # Flux at left edges
    px.f_L[i0:iend+1,:] = px.q_R[i0-1:iend,:] - cx[i0:iend+1,:]*0.5*(px.dq[i0-1:iend,:] - (1.0-(2.0/3.0)*cx[i0:iend+1,:])*px.q6[i0-1:iend,:])

    # Flux at right edges
    px.f_R[i0:iend+1,:] = px.q_L[i0:iend+1,:] - cx[i0:iend+1,:]*0.5*(px.dq[i0:iend+1,:] + (1.0+(2.0/3.0)*cx[i0:iend+1,:])*px.q6[i0:iend+1,:])

    # F - Formula 1.13 from Collela and Woodward 1984)
    px.f_upw[cx[:,:] >= 0] = px.f_L[cx[:,:] >= 0]
    px.f_upw[cx[:,:] <= 0] = px.f_R[cx[:,:] <= 0]

####################################################################################
# Compute the 1d flux operator in y direction
# Inputs: Q (average values),  v_edges (velocity at edges)
# ay (stencil coefficients), cy, cy2 (CFL, CFL^2 in y direction)
# Output: flux_y (flux in y direction)
####################################################################################
def compute_flux_y(Q, py, cy, simulation):
    M = simulation.M

    # Reconstructs the values of Q using a piecewise parabolic polynomial
    rec.ppm_reconstruction_y(Q, py, simulation)

    # Compute the fluxes
    numerical_flux_ppm_y(py, cy, simulation)

####################################################################################
# PPM flux in y direction
####################################################################################
def numerical_flux_ppm_y(py, cy, simulation):
    N = simulation.N
    M = simulation.M
    ng = simulation.ng
    j0 = simulation.j0
    jend = simulation.jend

    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    # Flux at left edges
    py.f_L[:,j0:jend+1] = py.q_R[:,j0-1:jend] - cy[:,j0:jend+1]*0.5*(py.dq[:,j0-1:jend] - (1.0-2.0/3.0*cy[:,j0:jend+1])*py.q6[:,j0-1:jend])

    # Flux at right edges
    py.f_R[:,j0:jend+1] = py.q_L[:,j0:jend+1] - cy[:,j0:jend+1]*0.5*(py.dq[:,j0:jend+1] + (1.0+2.0/3.0*cy[:,j0:jend+1])*py.q6[:,j0:jend+1])

    # G - Formula 1.13 from Collela and Woodward 1984)
    py.f_upw[cy[:,:] >= 0] = py.f_L[cy[:,:] >= 0]
    py.f_upw[cy[:,:] <= 0] = py.f_R[cy[:,:] <= 0]

####################################################################################
# Compute the fluxes needed for dimension spliting operators
####################################################################################
def compute_fluxes(Qx, Qy, px, py, cx, cy, simulation):
    compute_flux_x(Qx, px, cx, simulation)
    compute_flux_y(Qy, py, cy, simulation)

"""
####################################################################################
# Compute the 1d flux operator from PPM using its stencil
# Inputs: Q (average values),  u_edges (zonal velocity at edges)
# ax (stencil coefficients), cx, cx2 (CFL, CFL^2 in x direction)
# Output: flux_x (flux in x direction)
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

    flux_x[3:N+4,:]  = flux_x[3:N+4,:]

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

"""
