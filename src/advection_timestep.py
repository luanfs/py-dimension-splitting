####################################################################################
# This module contains the routine that computes one advection timestep
# Luan da Fonseca Santos - October 2022
####################################################################################

import numpy as np
from dimension_splitting import F_operator, G_operator
from advection_ic        import velocity_adv_2d
from stencil             import flux_ppm_x_stencil_coefficients, flux_ppm_y_stencil_coefficients
from cfl                 import cfl_x, cfl_y
from flux                import compute_fluxes

def adv_timestep(Q, u_edges, v_edges, F, G, FQ, GQ, flux_x, flux_y, ax, cx, cx2, ay, cy, cy2, Xu, Yu, Xv, Yv, t, simulation):

    N  = simulation.N    # Number of cells in x direction
    M  = simulation.M    # Number of cells in y direction

    # Ghost cells
    ngl = simulation.ngl
    ngr = simulation.ngr
    ng  = simulation.ng


    # Grid interior indexes
    i0 = simulation.i0
    iend = simulation.iend
    j0 = simulation.j0
    jend = simulation.jend


    # Compute the fluxes
    compute_fluxes(Q, Q, u_edges, v_edges, flux_x, flux_y, ax, ay, cx, cy, cx2, cy2, simulation)

    # Applies F and G operators
    F_operator(FQ, u_edges, flux_x, ax, cx, cx2, simulation)
    G_operator(GQ, v_edges, flux_y, ay, cy, cy2, simulation)

    # Compute the fluxes again
    compute_fluxes(Q + 0.5*GQ, Q + 0.5*FQ, u_edges, v_edges, flux_x, flux_y, ax, ay, cx, cy, cx2, cy2, simulation)

    # Applies F and G operators again
    F_operator(F, u_edges, flux_x, ax, cx, cx2, simulation)
    G_operator(G, v_edges, flux_y, ay, cy, cy2, simulation)

    # Update
    Q[i0:iend,j0:jend] = Q[i0:iend,j0:jend] + F[i0:iend,j0:jend] + G[i0:iend,j0:jend]

    # Periodic boundary conditions
    # x direction
    Q[iend:N+ng,:] = Q[i0:i0+ngr,:]
    Q[0:i0,:]      = Q[N:N+ngl,:]

    # y direction
    Q[:,jend:N+ng] = Q[:,j0:j0+ngr]
    Q[:,0:j0]      = Q[:,M:M+ngl]

    # Velocity and  CFL numbers update - only for time dependent velocity
    if simulation.vf == 2:
        # Velocity update
        u_edges[:,:], _ = velocity_adv_2d(Xu, Yu, t, simulation)
        _, v_edges[:,:] = velocity_adv_2d(Xv, Yv, t, simulation)

        # CFL at edges - x direction
        cx, cx2 = cfl_x(u_edges, simulation)

        # CFL at edges - y direction
        cy, cy2 = cfl_y(v_edges, simulation)

        # Coefs update
        flux_ppm_x_stencil_coefficients(u_edges, ax, cx, cx2, simulation)
        flux_ppm_y_stencil_coefficients(v_edges, ay, cy, cy2, simulation)
