####################################################################################
#
# Advection module
# Luan da Fonseca Santos - April 2022
# Solves the 2d advection equation  with periodic boundary conditions
# The initial condition Q(x,y,0) is given in the module parameters_2d.py
#
# References:
# Lin, S., & Rood, R. B. (1996). Multidimensional Flux-Form Semi-Lagrangian
# Transport Schemes, Monthly Weather Review, 124(9), 2046-2070, from
# https://journals.ametsoc.org/view/journals/mwre/124/9/1520-0493_1996_124_2046_mffslt_2_0_co_2.xml
#
####################################################################################

import numpy as np
from miscellaneous       import diagnostics_adv_2d, print_diagnostics_adv_2d, plot_2dfield_graphs, output_adv
from parameters_2d       import q0_adv_2d, graphdir, qexact_adv_2d, velocity_adv_2d
from dimension_splitting import F_operator, G_operator
from stencil             import flux_ppm_x_stencil_coefficients, flux_ppm_y_stencil_coefficients
from cfl                 import cfl_x, cfl_y
from errors import *

def adv_2d(simulation, plot):
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
    mono   = simulation.mono  # Monotonization scheme

    # Velocity at edges
    u_edges = np.zeros((N+7, M+6))
    v_edges = np.zeros((N+6, M+7))

    # Number of time steps
    Nsteps = int(Tf/dt)

    # Grid
    Xc, Yc = np.meshgrid(xc, yc, indexing='ij')
    X , Y  = np.meshgrid(x , y , indexing='ij')

    # edges
    Xu, Yu = np.meshgrid(x, yc,indexing='ij')
    Xv, Yv = np.meshgrid(xc, y,indexing='ij')
    u_edges[:, :], _ = velocity_adv_2d(Xu, Yu, 0.0, simulation)
    _, v_edges[:,:] = velocity_adv_2d(Xv, Yv, 0.0, simulation)

    # CFL at edges - x direction
    cx, cx2 = cfl_x(u_edges, simulation)
 
    # CFL at edges - y direction
    cy, cy2 = cfl_y(v_edges, simulation)
 
    # CFL number
    CFL_x = np.amax(cx)
    CFL_y = np.amax(cy)
    CFL = np.sqrt(CFL_x**2 + CFL_y**2)

    # Dimension splitting variables
    FQ = np.zeros((N+6, M+6))
    GQ = np.zeros((N+6, M+6))
    F  = np.zeros((N+6, M+6))    
    G  = np.zeros((N+6, M+6))

    # Flux at edges
    flux_x = np.zeros((N+7, M+6))
    flux_y = np.zeros((N+6, M+7))

    # Stencil coefficients
    ay = np.zeros((6, N+6, M+7))
    ax = np.zeros((6, N+7, M+6))

    # Compute the coefficients
    flux_ppm_x_stencil_coefficients(u_edges, ax, cx, cx2, simulation)
    flux_ppm_y_stencil_coefficients(v_edges, ay, cy, cy2, simulation)

    # Compute average values of Q (initial condition)
    Q = np.zeros((N+6, M+6))
    Q[3:N+3,3:M+3] = q0_adv_2d(Xc[3:N+3,3:M+3], Yc[3:N+3,3:M+3], simulation)

    # Periodic boundary conditions
    # x direction
    Q[N+3:N+6,:] = Q[3:6,:]
    Q[0:3,:]     = Q[N:N+3,:]
    # y direction
    Q[:,M+3:M+6] = Q[:,3:6]
    Q[:,0:3]     = Q[:,M:M+3]

    # Plotting var
    plotstep = 100

    # Compute initial mass
    total_mass0, _ = diagnostics_adv_2d(Q, simulation, 1.0)

    # Error variables
    error_linf = np.zeros(Nsteps+1)
    error_l1   = np.zeros(Nsteps+1)
    error_l2   = np.zeros(Nsteps+1)
    
    # Initial plotting
    output_adv(Xc[3:N+3,3:M+3] , Yc[3:N+3,3:M+3] , simulation, Q, error_linf, error_l1, error_l2, plot, 0, 0.0, Nsteps, plotstep, total_mass0, CFL)

    # Time looping
    for k in range(1, Nsteps+1):
        # Time
        t = k*dt

        # Applies F and G operators
        F_operator(FQ, Q, u_edges, flux_x, ax, cx, cx2, simulation)
        G_operator(GQ, Q, v_edges, flux_y, ay, cy, cy2, simulation)

        # Applies F and G operators again
        F_operator(F, Q + 0.5*GQ, u_edges, flux_x, ax, cx, cx2, simulation)
        G_operator(G, Q + 0.5*FQ, v_edges, flux_y, ay, cy, cy2, simulation)

        # Update
        #Q = Q + F + G
        Q[3:N+3,3:M+3] = Q[3:N+3,3:M+3] + F[3:N+3,3:M+3] + G[3:N+3,3:M+3]

        # Periodic boundary conditions
        # x direction
        Q[N+3:N+6,:] = Q[3:6,:]
        Q[0:3,:]     = Q[N:N+3,:]
        # y direction
        Q[:,M+3:M+6] = Q[:,3:6]
        Q[:,0:3]     = Q[:,M:M+3]

        # Velocity and  CFL numbers update - only for time dependent velocity
        if simulation.ic == 4:
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

 
        # Output and plot
        output_adv(Xc[3:N+3,3:M+3], Yc[3:N+3,3:M+3], simulation, Q, error_linf, error_l1, error_l2, plot, k, t, Nsteps, plotstep, total_mass0, CFL)
    #---------------------------------------End of time loop---------------------------------------

    if plot:
        # Plot the error evolution graph
        title = simulation.title +'- '+icname+', CFL='+str(CFL)+',\n N='+str(N)+', '+simulation.fvmethod+', mono = '+simulation.monot
        filename = graphdir+'2d_adv_tc'+str(tc)+'_ic'+str(ic)+'_N'+str(N)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'_errors.png'
        plot_time_evolution([error_linf, error_l1, error_l2], Tf, ['$L_\infty}$','$L_1$','$L_2$'], 'Error', filename, title)
    return error_linf[Nsteps], error_l1[Nsteps], error_l2[Nsteps]
