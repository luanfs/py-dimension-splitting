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

def adv_timestep(t, k, simulation):
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
    time_averaged_velocity(simulation.U_pu, simulation.U_pv, t, simulation)

    # CFL calculation
    simulation.cx[:,:] = cfl_x(simulation.U_pu.u_averaged[:,:], simulation)
    simulation.cy[:,:] = cfl_y(simulation.U_pv.v_averaged[:,:], simulation)

    # Compute the divergence
    divergence(simulation)

    # Update
    #Q[i0:iend,j0:jend] = Q[i0:iend,j0:jend] - div[i0:iend,j0:jend]
    Q0 = simulation.Q[i0:iend,j0:jend]
    d0 = -simulation.dt*simulation.div[i0:iend,j0:jend]
    simulation.Q[i0:iend,j0:jend] = ne.evaluate('Q0+d0')

    # Periodic boundary conditions
    # x direction
    simulation.Q[iend:N+ng,:] = simulation.Q[i0:i0+ngr,:]
    simulation.Q[0:i0,:]      = simulation.Q[N:N+ngl,:]

    # y direction
    simulation.Q[:,jend:N+ng] = simulation.Q[:,j0:j0+ngr]
    simulation.Q[:,0:j0]      = simulation.Q[:,M:M+ngl]

    # Updates for next time step
    if simulation.vf >= 2:
        # Velocity update
        simulation.U_pu.u_old[:,:] = simulation.U_pu.u[:,:]
        simulation.U_pv.v_old[:,:] = simulation.U_pv.v[:,:]
        simulation.U_pu.u[:,:] = u_velocity_adv_2d(simulation.Xu, simulation.Yu, t, simulation)
        simulation.U_pv.v[:,:] = v_velocity_adv_2d(simulation.Xv, simulation.Yv, t, simulation)

