####################################################################################
# This module contains the routine that initializates the advection routine variables
# Luan da Fonseca Santos - 2023
####################################################################################

import numpy as np
from parameters_2d       import ppm_parabola, velocity
from advection_ic        import q0_adv_2d, qexact_adv_2d, u_velocity_adv_2d, v_velocity_adv_2d
from cfl                 import cfl_x, cfl_y


def adv_vars(simulation):
    N  = simulation.N    # Number of cells in x direction
    M  = simulation.M    # Number of cells in y direction
    ic = simulation.ic   # Initial condition

    x  = simulation.x    # Grid
    xc = simulation.xc
    x0 = simulation.x0
    xf = simulation.xf
    dx = simulation.dx   # Grid spacing

    y  = simulation.y    # Grid
    yc = simulation.yc
    y0 = simulation.y0
    yf = simulation.yf
    dy = simulation.dy   # Grid spacing

    dt = simulation.dt   # Time step
    Tf = simulation.Tf   # Total period definition

    tc = simulation.tc
    icname = simulation.icname
    recon = simulation.recon
    dp = simulation.dp

    # Ghost cells
    ngl = simulation.ngl
    ngr = simulation.ngr
    ng  = simulation.ng

    # Grid interior indexes
    i0 = simulation.i0
    iend = simulation.iend
    j0 = simulation.j0
    jend = simulation.jend

    # Velocity at edges
    simulation.U_pu = velocity(simulation, 'pu')
    simulation.U_pv = velocity(simulation, 'pv')

    # Grid
    simulation.Xc, simulation.Yc = np.meshgrid(xc, yc, indexing='ij')
    X , Y  = np.meshgrid(x , y , indexing='ij')

    # edges
    simulation.Xu, simulation.Yu = np.meshgrid(x, yc,indexing='ij')
    simulation.Xv, simulation.Yv = np.meshgrid(xc, y,indexing='ij')

    # Initial velocity
    simulation.U_pu.u[:,:] = u_velocity_adv_2d(simulation.Xu, simulation.Yu, 0.0, simulation)
    simulation.U_pv.v[:,:] = v_velocity_adv_2d(simulation.Xv, simulation.Yv, 0.0, simulation)
    simulation.U_pu.u_old[:,:] = simulation.U_pu.u[:,:]
    simulation.U_pv.v_old[:,:] = simulation.U_pv.v[:,:]
    simulation.U_pu.u_timecenter[:,:] = simulation.U_pu.u[:,:]
    simulation.U_pv.v_timecenter[:,:] = simulation.U_pv.v[:,:]

    # CFL at edges - x direction
    simulation.cx = cfl_x(simulation.U_pu.u[:,:], simulation)

    # CFL at edges - y direction
    simulation.cy = cfl_y(simulation.U_pv.v[:,:], simulation)

    # CFL number
    CFL_x = np.amax(abs(simulation.cx))
    CFL_y = np.amax(abs(simulation.cy))
    simulation.CFL = max(CFL_x, CFL_y)

    # Compute average values of Q (initial condition)
    simulation.Q[i0:iend,j0:jend] = q0_adv_2d(simulation.Xc[i0:iend,j0:jend], simulation.Yc[i0:iend,j0:jend], simulation)

    # Periodic boundary conditions
    # x direction
    simulation.Q[iend:N+ng,:] = simulation.Q[i0:i0+ngr,:]
    simulation.Q[0:i0,:]      = simulation.Q[N:N+ngl,:]

    # y direction
    simulation.Q[:,jend:N+ng] = simulation.Q[:,j0:j0+ngr]
    simulation.Q[:,0:j0]      = simulation.Q[:,M:M+ngl]

    # PPM parabolas
    simulation.px = ppm_parabola(simulation,'x')
    simulation.py = ppm_parabola(simulation,'y')

    return
