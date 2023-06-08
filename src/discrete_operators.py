####################################################################################
# This module contains the routine that computes the discrete
# differential operator using finite volume discretization
# Luan da Fonseca Santos - 2023
####################################################################################

import numpy as np
from flux import compute_fluxes
import numexpr as ne

####################################################################################
# Given Q, compute div(UQ), where U = (u,v), and cx and cy
# are the cfl numbers (must be already computed)
# The divergence is given by px.dF + py.dF
####################################################################################
def divergence(simulation):
    # Compute the fluxes
    compute_fluxes(simulation.Q, simulation.Q, simulation.px, simulation.py,\
    simulation.U_pu, simulation.U_pv, simulation.cx, simulation.cy, simulation)

    # Applies F and G operators
    F_operator(simulation)
    G_operator(simulation)

    N = simulation.N
    ng = simulation.ng
    Q = simulation.Q
    # Splitting scheme
    if simulation.opsplit==1:
        pxdF = simulation.px.dF
        pydF = simulation.py.dF
        Qx = ne.evaluate("Q+0.5*pxdF")
        Qy = ne.evaluate("Q+0.5*pydF")
    elif simulation.opsplit==2:
        # L04 equation 7 and 8
        #px.dF = px.dF + (cx[1:,:]-cx[:N+ng,:])*Q
        #py.dF = py.dF + (cy[:,1:]-cy[:,:N+ng])*Q
        #Qx = Q+0.5*px.dF
        #Qy = Q+0.5*py.dF
        pxdF = simulation.px.dF
        pydF = simulation.py.dF
        c1x, c2x = simulation.cx[1:,:], simulation.cx[:N+ng,:]
        c1y, c2y = simulation.cy[:,1:], simulation.cy[:,:N+ng]
        Qx = ne.evaluate('(Q + 0.5*(pxdF + (c1x-c2x)*Q))')
        Qy = ne.evaluate('(Q + 0.5*(pydF + (c1y-c2y)*Q))')
    elif simulation.opsplit==3:
        # PL07 - equation 17 and 18
        #Qx = 0.5*(Q + (Q + px.dF)/(1.0-(cx[1:,:]-cx[:N+ng,:])))
        #Qy = 0.5*(Q + (Q + py.dF)/(1.0-(cy[:,1:]-cy[:,:N+ng])))
        pxdF = simulation.px.dF
        pydF = simulation.py.dF
        c1x, c2x = simulation.cx[1:,:], simulation.cx[:N+ng,:]
        c1y, c2y = simulation.cy[:,1:], simulation.cy[:,:N+ng]
        Qx = ne.evaluate('0.5*(Q + (Q + pxdF)/(1.0-(c1x-c2x)))')
        Qy = ne.evaluate('0.5*(Q + (Q + pydF)/(1.0-(c1y-c2y)))')

    # Compute the fluxes again
    compute_fluxes(Qy, Qx, simulation.px, simulation.py,\
    simulation.U_pu, simulation.U_pv, simulation.cx, simulation.cy, simulation)

    # Applies F and G operators again
    F_operator(simulation)
    G_operator(simulation)

    # Compute the divergence
    pxdF = simulation.px.dF
    pydF = simulation.py.dF
    dt = simulation.dt
    simulation.div[:,:] = ne.evaluate("-(pxdF + pydF)/dt")

####################################################################################
# Flux operator in x direction
# Inputs: Q (average values),
# u_edges (velocity in x direction at edges)
# Formula 2.7 from Lin and Rood 1996
####################################################################################
def F_operator(simulation):
    N = simulation.N
    i0 = simulation.i0
    iend = simulation.iend

    #F[i0:iend,:] = -(cx[i0+1:iend+1,:]*flux_x[i0+1:iend+1,:] - cx[i0:iend,:]*flux_x[i0:iend,:])
    f1 = simulation.px.f_upw[i0+1:iend+1,:]
    f2 = simulation.px.f_upw[i0:iend,:]
    simulation.px.dF[i0:iend,:] = ne.evaluate("-(f1-f2)")
    simulation.px.dF[i0:iend,:] = simulation.px.dF[i0:iend,:]*simulation.dt/simulation.dx

####################################################################################
# Flux operator in y direction
# Inputs: Q (average values),
# v_edges (velocity in y direction at edges)
# Formula 2.8 from Lin and Rood 1996
####################################################################################
def G_operator(simulation):
    M = simulation.M
    j0 = simulation.j0
    jend = simulation.jend

    #G[:,j0:jend] = -(cy[:,j0+1:jend+1]*flux_y[:,j0+1:jend+1] - cy[:,j0:jend]*flux_y[:,j0:jend])
    g1 = simulation.py.f_upw[:,j0+1:jend+1]
    g2 = simulation.py.f_upw[:,j0:jend]
    simulation.py.dF[:,j0:jend] = ne.evaluate("-(g1-g2)")
    simulation.py.dF[:,j0:jend] = simulation.py.dF[:,j0:jend]*simulation.dt/simulation.dy

