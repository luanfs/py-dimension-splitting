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
import numexpr as ne

####################################################################################
# Compute the fluxes needed for dimension spliting operators
####################################################################################
def compute_fluxes(Qx, Qy, px, py, cx, cy, simulation):
    # Reconstructs the values of Q using a piecewise parabolic polynomial
    rec.ppm_reconstruction_x(Qx, px, simulation)
    rec.ppm_reconstruction_y(Qy, py, simulation)

    # Compute the fluxes
    numerical_flux_ppm_x(px, cx, simulation)
    numerical_flux_ppm_y(py, cy, simulation)

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
    #px.f_L[i0:iend+1,:] = px.q_R[i0-1:iend,:] - cx[i0:iend+1,:]*0.5*(px.dq[i0-1:iend,:] - (1.0-(2.0/3.0)*cx[i0:iend+1,:])*px.q6[i0-1:iend,:])
    q_R = px.q_R[i0-1:iend,:]
    c   = cx[i0:iend+1,:]
    dq  = px.dq[i0-1:iend,:]
    q6  = px.q6[i0-1:iend,:]
    px.f_L[i0:iend+1,:] = ne.evaluate("q_R - c*0.5*(dq - (1.0-(2.0/3.0)*c)*q6)")

    # Flux at right edges
    #px.f_R[i0:iend+1,:] = px.q_L[i0:iend+1,:] - cx[i0:iend+1,:]*0.5*(px.dq[i0:iend+1,:] + (1.0+(2.0/3.0)*cx[i0:iend+1,:])*px.q6[i0:iend+1,:])
    q_L = px.q_L[i0:iend+1,:]
    c   = cx[i0:iend+1,:]
    dq  = px.dq[i0:iend+1,:]
    q6  = px.q6[i0:iend+1,:]
    px.f_R[i0:iend+1,:] = ne.evaluate("q_L - c*0.5*(dq + (1.0+(2.0/3.0)*c)*q6)")

    # F - Formula 1.13 from Collela and Woodward 1984)
    mask = ne.evaluate('cx >= 0')
    px.f_upw[mask]  = px.f_L[mask]
    px.f_upw[~mask] = px.f_R[~mask]

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
    #py.f_L[:,j0:jend+1] = py.q_R[:,j0-1:jend] - cy[:,j0:jend+1]*0.5*(py.dq[:,j0-1:jend] - (1.0-2.0/3.0*cy[:,j0:jend+1])*py.q6[:,j0-1:jend])
    q_R = py.q_R[:,j0-1:jend]
    c   = cy[:,j0:jend+1]
    dq  = py.dq[:,j0-1:jend]
    q6  = py.q6[:,j0-1:jend]
    py.f_L[:,j0:jend+1] = ne.evaluate("q_R - c*0.5*(dq - (1.0-(2.0/3.0)*c)*q6)")

    # Flux at right edges
    #py.f_R[:,j0:jend+1] = py.q_L[:,j0:jend+1] - cy[:,j0:jend+1]*0.5*(py.dq[:,j0:jend+1] + (1.0+2.0/3.0*cy[:,j0:jend+1])*py.q6[:,j0:jend+1])
    q_L = py.q_L[:,j0:jend+1]
    c   = cy[:,j0:jend+1]
    dq  = py.dq[:,j0:jend+1]
    q6  = py.q6[:,j0:jend+1]
    py.f_R[:,j0:jend+1] = ne.evaluate("q_L - c*0.5*(dq + (1.0+(2.0/3.0)*c)*q6)")

    # G - Formula 1.13 from Collela and Woodward 1984)
    mask = ne.evaluate('cy >= 0')
    py.f_upw[mask]  = py.f_L[mask]
    py.f_upw[~mask] = py.f_R[~mask]


