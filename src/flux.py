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
def compute_fluxes(Qx, Qy, px, py, U_pu, U_pv, cx, cy, simulation):
    # Reconstructs the values of Q using a piecewise parabolic polynomial
    rec.ppm_reconstruction_x(Qx, px, simulation)
    rec.ppm_reconstruction_y(Qy, py, simulation)

    # Compute the fluxes
    numerical_flux_ppm_x(px, U_pu, cx, simulation)
    numerical_flux_ppm_y(py, U_pv, cy, simulation)

####################################################################################
# PPM flux in x direction
####################################################################################
def numerical_flux_ppm_x(px, U_pu, cx, simulation):
    N = simulation.N
    M = simulation.M
    ng = simulation.ng
    i0 = simulation.i0
    iend = simulation.iend

    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    # Flux at left edges
    upos = U_pu.upos
    c   = cx[i0:iend+1,:][upos]
    q_R = px.q_R[i0-1:iend,:][upos]
    dq  = px.dq[i0-1:iend,:][upos]
    q6  = px.q6[i0-1:iend,:][upos]
    px.f_L[i0:iend+1,:][upos] = ne.evaluate("q_R + c*0.5*(q6-dq) - q6*c*c/3.0")

    # Flux at right edges
    uneg = U_pu.uneg
    c   = cx[i0:iend+1,:][uneg]
    q_L = px.q_L[i0:iend+1,:][uneg]
    dq  = px.dq[i0:iend+1,:][uneg]
    q6  = px.q6[i0:iend+1,:][uneg]
    px.f_R[i0:iend+1,:][uneg] = ne.evaluate("q_L - c*0.5*(q6+dq) - q6*c*c/3.0")

    # F - Formula 1.13 from Collela and Woodward 1984)
    px.f_upw[i0:iend+1,:][upos] = px.f_L[i0:iend+1,:][upos]
    px.f_upw[i0:iend+1,:][uneg] = px.f_R[i0:iend+1,:][uneg]
    px.f_upw[i0:iend+1,:] = U_pu.u_averaged[i0:iend+1,:]*px.f_upw[i0:iend+1,:]

####################################################################################
# PPM flux in y direction
####################################################################################
def numerical_flux_ppm_y(py, U_pv, cy, simulation):
    N = simulation.N
    M = simulation.M
    ng = simulation.ng
    j0 = simulation.j0
    jend = simulation.jend

    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    # Flux at left edges
    vpos = U_pv.vpos
    c   = cy[:,j0:jend+1][vpos]
    q_R = py.q_R[:,j0-1:jend][vpos]
    dq  = py.dq[:,j0-1:jend][vpos]
    q6  = py.q6[:,j0-1:jend][vpos]
    py.f_L[:,j0:jend+1][vpos] = ne.evaluate("q_R + c*0.5*(q6-dq) - q6*c*c/3.0")

    # Flux at right edges
    vneg = U_pv.vneg
    c   = cy[:,j0:jend+1][vneg]
    q_L = py.q_L[:,j0:jend+1][vneg]
    dq  = py.dq[:,j0:jend+1][vneg]
    q6  = py.q6[:,j0:jend+1][vneg]
    py.f_R[:,j0:jend+1][vneg] = ne.evaluate("q_L - c*0.5*(q6+dq) - q6*c*c/3.0")

    # F - Formula 1.13 from Collela and Woodward 1984)
    py.f_upw[:,j0:jend+1][vpos] = py.f_L[:,j0:jend+1][vpos]
    py.f_upw[:,j0:jend+1][vneg] = py.f_R[:,j0:jend+1][vneg]
    py.f_upw[:,j0:jend+1] = U_pv.v_averaged[:,j0:jend+1]*py.f_upw[:,j0:jend+1]
