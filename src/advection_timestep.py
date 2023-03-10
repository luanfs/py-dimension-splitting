####################################################################################
# This module contains the routine that computes one advection timestep
# Luan da Fonseca Santos - October 2022
####################################################################################

import numpy as np
from advection_ic        import velocity_adv_2d, u_velocity_adv_2d, v_velocity_adv_2d
from cfl                 import cfl_x, cfl_y
from discrete_operators  import divergence
from averaged_velocity   import time_averaged_velocity
import numexpr as ne

def adv_timestep(Q, div, U_pu, U_pv, px, py, cx, cy, Xu, Yu, Xv, Yv, t, k, simulation):
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
    time_averaged_velocity(U_pu, U_pv, Xu, Yu, Xv, Yv, k, t, simulation)

    # CFL calculation
    cx[:,:] = cfl_x(U_pu.u_averaged[:,:], simulation)
    cy[:,:] = cfl_y(U_pv.v_averaged[:,:], simulation)

    # Compute the divergence
    divergence(Q, div, px, py, cx, cy, simulation)

    # Update
    #Q[i0:iend,j0:jend] = Q[i0:iend,j0:jend] - div[i0:iend,j0:jend]
    Q0 = Q[i0:iend,j0:jend]
    d0 = -simulation.dt*div[i0:iend,j0:jend]
    Q[i0:iend,j0:jend] = ne.evaluate('Q0+d0')

    # Periodic boundary conditions
    # x direction
    Q[iend:N+ng,:] = Q[i0:i0+ngr,:]
    Q[0:i0,:]      = Q[N:N+ngl,:]

    # y direction
    Q[:,jend:N+ng] = Q[:,j0:j0+ngr]
    Q[:,0:j0]      = Q[:,M:M+ngl]

