####################################################################################
# This module contains the routine that computes one advection timestep
# Luan da Fonseca Santos - October 2022
####################################################################################

import numpy as np
from advection_ic        import velocity_adv_2d, u_velocity_adv_2d, v_velocity_adv_2d
from cfl                 import cfl_x, cfl_y
from discrete_operators  import divergence
from averaged_velocity import time_averaged_velocity

def adv_timestep(Q, u_edges, v_edges, F, G, FQ, GQ, px, py, cx, cy, Xu, Yu, Xv, Yv, t, k, simulation):

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

    # Compute the velocity need for the departure point (only for variable velocity)
    time_averaged_velocity(u_edges, v_edges, k, simulation)

    # CFL calculation
    cx[:,:] = cfl_x(u_edges[:,:,0], simulation)
    cy[:,:] = cfl_y(v_edges[:,:,0], simulation)

    # Compute the divergence
    divergence(Q, u_edges[:,:,0], v_edges[:,:,0], px, py, cx, cy, simulation)

    # Update
    Q[i0:iend,j0:jend] = Q[i0:iend,j0:jend] + px.dF[i0:iend,j0:jend] + py.dF[i0:iend,j0:jend]

    # Periodic boundary conditions
    # x direction
    Q[iend:N+ng,:] = Q[i0:i0+ngr,:]
    Q[0:i0,:]      = Q[N:N+ngl,:]

    # y direction
    Q[:,jend:N+ng] = Q[:,j0:j0+ngr]
    Q[:,0:j0]      = Q[:,M:M+ngl]

    # Updates for next time step
    if simulation.vf >= 2:
        # Velocity update
        if simulation.dp == 1:
            u_edges[:,:,0] = u_velocity_adv_2d(Xu, Yu, t, simulation)
            v_edges[:,:,0] = v_velocity_adv_2d(Xv, Yv, t, simulation)
        elif simulation.dp == 2:
            u_edges[:,:,0] = u_edges[:,:,1]
            v_edges[:,:,0] = v_edges[:,:,1]
            u_edges[:,:,1] = u_velocity_adv_2d(Xu, Yu, t, simulation)
            v_edges[:,:,1] = v_velocity_adv_2d(Xv, Yv, t, simulation)
        elif simulation.dp == 3:
            u_edges[:,:,0] = u_edges[:,:,1]
            u_edges[:,:,1] = u_edges[:,:,2]
            u_edges[:,:,2] = u_velocity_adv_2d(Xu, Yu, t, simulation)

            v_edges[:,:,0] = v_edges[:,:,1]
            v_edges[:,:,1] = v_edges[:,:,2]
            v_edges[:,:,2] = v_velocity_adv_2d(Xv, Yv, t, simulation)

