####################################################################################
# This module contains the routine that computes the discrete
# differential operator using finite volume discretization
# Luan da Fonseca Santos - 2023
####################################################################################

import numpy as np
from flux                import compute_fluxes

####################################################################################
# Given Q, compute div(UQ), where U = (u,v), and cx and cy
# are the cfl numbers (must be already computed)
# The divergence is given by px.dF + py.dF
####################################################################################
def divergence(Q, u_edges, v_edges, px, py, cx, cy, simulation):
    # Compute the fluxes
    compute_fluxes(Q, Q, px, py, cx, cy, simulation)

    # Applies F and G operators
    F_operator(px.dF, u_edges, px.f_upw, cx, simulation)
    G_operator(py.dF, v_edges, py.f_upw, cy, simulation)

    N = simulation.N
    ng = simulation.ng

    # Splitting scheme
    if simulation.opsplit==1:
        Qx = Q+0.5*px.dF
        Qy = Q+0.5*py.dF
    elif simulation.opsplit==2:
        # L04 equation 7 and 8
        px.dF = px.dF + (cx[1:,:]-cx[:N+ng,:])*Q
        py.dF = py.dF + (cy[:,1:]-cy[:,:N+ng])*Q
        Qx = Q+0.5*px.dF
        Qy = Q+0.5*py.dF
    elif simulation.opsplit==3:
        # PL07 - equation 17 and 18
        Qx = 0.5*(Q + (Q + px.dF)/(1.0-(cx[1:,:]-cx[:N+ng,:])))
        Qy = 0.5*(Q + (Q + py.dF)/(1.0-(cy[:,1:]-cy[:,:N+ng])))

    # Compute the fluxes again
    compute_fluxes(Qy, Qx, px, py, cx, cy, simulation)

    # Applies F and G operators again
    F_operator(px.dF, u_edges, px.f_upw, cx, simulation)
    G_operator(py.dF, v_edges, py.f_upw, cy, simulation)

####################################################################################
# Operator splitting implementation
# Luan da Fonseca Santos - June 2022
#
# References:
# Lin, S., & Rood, R. B. (1996). Multidimensional Flux-Form Semi-Lagrangian
# Transport Schemes, Monthly Weather Review, 124(9), 2046-2070, from
# https://journals.ametsoc.org/view/journals/mwre/124/9/1520-0493_1996_124_2046_mffslt_2_0_co_2.xml
#
###################################################################################

#from flux import compute_flux_x, compute_flux_y

####################################################################################
# Flux operator in x direction
# Inputs: Q (average values),
# u_edges (velocity in x direction at edges)
# Formula 2.7 from Lin and Rood 1996
####################################################################################
def F_operator(F, u_edges, flux_x, cx, simulation):
    N = simulation.N
    i0 = simulation.i0
    iend = simulation.iend

    F[i0:iend,:] = -(simulation.dt/simulation.dx)*(u_edges[i0+1:iend+1,:]*flux_x[i0+1:iend+1,:] - u_edges[i0:iend,:]*flux_x[i0:iend,:])

####################################################################################
# Flux operator in y direction
# Inputs: Q (average values),
# v_edges (velocity in y direction at edges)
# Formula 2.8 from Lin and Rood 1996
####################################################################################
def G_operator(G, v_edges, flux_y, cy, simulation):
    M = simulation.M
    j0 = simulation.j0
    jend = simulation.jend

    G[:,j0:jend] = -(simulation.dt/simulation.dy)*(v_edges[:,j0+1:jend+1]*flux_y[:,j0+1:jend+1] - v_edges[:,j0:jend]*flux_y[:,j0:jend])


