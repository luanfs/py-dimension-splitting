####################################################################################
# This module contains the routine that initializates the advection routine variables
# Luan da Fonseca Santos - 2023
####################################################################################

import numpy as np
from parameters_2d       import ppm_parabola, velocity_edges
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
    U_pu = velocity_edges(simulation, 'pu')
    U_pv = velocity_edges(simulation, 'pv')

    # Grid
    Xc, Yc = np.meshgrid(xc, yc, indexing='ij')
    X , Y  = np.meshgrid(x , y , indexing='ij')

    # edges
    Xu, Yu = np.meshgrid(x, yc,indexing='ij')
    Xv, Yv = np.meshgrid(xc, y,indexing='ij')

    # Initial velocity
    U_pu.u[:,:] = u_velocity_adv_2d(Xu, Yu, 0.0, simulation)
    U_pv.v[:,:] = v_velocity_adv_2d(Xv, Yv, 0.0, simulation)
    U_pu.u_old[:,:] = U_pu.u[:,:]
    U_pv.v_old[:,:] = U_pv.v[:,:]

    # CFL at edges - x direction
    cx = cfl_x(U_pu.u[:,:], simulation)

    # CFL at edges - y direction
    cy = cfl_y(U_pv.v[:,:], simulation)

    # CFL number
    CFL_x = np.amax(abs(cx))
    CFL_y = np.amax(abs(cy))
    CFL = max(CFL_x, CFL_y)

    # Compute average values of Q (initial condition)
    Q = np.zeros((N+ng, M+ng))
    div = np.zeros((N+ng, M+ng))
    Q[i0:iend,j0:jend] = q0_adv_2d(Xc[i0:iend,j0:jend], Yc[i0:iend,j0:jend], simulation)

    # Periodic boundary conditions
    # x direction
    Q[iend:N+ng,:] = Q[i0:i0+ngr,:]
    Q[0:i0,:]      = Q[N:N+ngl,:]

    # y direction
    Q[:,jend:N+ng] = Q[:,j0:j0+ngr]
    Q[:,0:j0]      = Q[:,M:M+ngl]

    # PPM parabolas
    px = ppm_parabola(simulation,'x')
    py = ppm_parabola(simulation,'y')

    return Q, div, U_pu, U_pv, px, py, cx, cy, \
           Xc, Yc, Xu, Yu, Xv, Yv, CFL
