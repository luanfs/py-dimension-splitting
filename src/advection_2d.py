####################################################################################
#
# Piecewise Parabolic Method (PPM) advection module
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
from errors import *
from miscellaneous       import diagnostics_adv_2d, print_diagnostics_adv_2d, plot_2dfield_graphs
from parameters_2d       import q0_adv_2d, graphdir, qexact_adv_2d, velocity_adv_2d
from dimension_splitting import F_operator, G_operator

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
    u_edges = np.zeros((N+1, M))
    v_edges = np.zeros((N, M+1))

    # Number of time steps
    Nsteps = int(Tf/dt)

    # Grid
    Xc, Yc = np.meshgrid(xc, yc, indexing='ij')
    X , Y  = np.meshgrid(x , y , indexing='ij')

    # edges
    Xu, Yu = np.meshgrid(x, yc,indexing='ij')
    Xv, Yv = np.meshgrid(xc, y,indexing='ij')
    u_edges[0:N+1, 0:M], _ = velocity_adv_2d(Xu, Yu, 0.0, simulation)
    _, v_edges[0:N, 0:M+1] = velocity_adv_2d(Xv, Yv, 0.0, simulation)

    # CFL number
    CFL_x = np.amax(abs(u_edges))*dt/dx
    CFL_y = np.amax(abs(v_edges))*dt/dy
    CFL = np.sqrt(CFL_x**2 + CFL_y**2)

    # Compute average values of Q (initial condition)
    Q = np.zeros((N+5, M+5))
    Q[2:N+2,2:M+2] = q0_adv_2d(Xc, Yc, simulation)

    # Periodic boundary conditions
    # x direction
    Q[N+2:N+5,:] = Q[2:5,:]
    Q[0:2,:]     = Q[N:N+2,:]
    # y direction
    Q[:,M+2:M+5] = Q[:,2:5]
    Q[:,0:2]     = Q[:,M:M+2]

    # Vector field plot var
    nplot = 40
    xplot = np.linspace(x0, xf, nplot)
    yplot = np.linspace(y0, yf, nplot)
    xplot, yplot = np.meshgrid(xplot, yplot)
    Uplot = np.zeros((nplot, nplot))
    Vplot = np.zeros((nplot, nplot))
    Uplot[0:nplot,0:nplot], Vplot[0:nplot,0:nplot] = velocity_adv_2d(xplot, yplot, 0.0, simulation)
    plotstep = 1

    # Plot the initial condition graph
    if plot:
        qmin = str("{:.2e}".format(np.amin(Q[2:N+2,2:M+2])))
        qmax = str("{:.2e}".format(np.amax(Q[2:N+2,2:M+2])))
        filename = graphdir+'2d_adv_tc'+str(tc)+'_ic'+str(ic)+'_t'+str(0)+'_N'+str(N)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'.png'
        title = '2D advection - '+icname+' - time='+str(0)+',\n N='+str(N)+', '+simulation.fvmethod+', mono = '+simulation.monot+ ', Min = '+ qmin +', Max = '+qmax
        plot_2dfield_graphs([Q[2:N+2,2:M+2]], Xc, Yc, [Uplot], [Vplot], xplot, yplot, filename, title)

    # Compute initial mass
    total_mass0, _ = diagnostics_adv_2d(Q, simulation, 1.0)

    # Error variables
    error_linf = np.zeros(Nsteps+1)
    error_l1   = np.zeros(Nsteps+1)
    error_l2   = np.zeros(Nsteps+1)

    # Time looping
    for t in range(1, Nsteps+1):
        # Velocity update
        u_edges[0:N+1, 0:M], _ = velocity_adv_2d(Xu, Yu, t*dt, simulation)
        _, v_edges[0:N, 0:M+1] = velocity_adv_2d(Xv, Yv, t*dt, simulation)

        # Applies F and G operators
        FQ = F_operator(Q, u_edges, simulation)
        GQ = G_operator(Q, v_edges, simulation)

        F = F_operator(Q + 0.5*GQ, u_edges, simulation)
        G = G_operator(Q + 0.5*FQ, v_edges, simulation)

        # Update
        Q = Q + F + G
        #Q[2:N+2,2:M+2] = Q[2:N+2,2:M+2] + F[2:N+2,2:M+2] + G[2:N+2,2:M+2]
        #Q[2:N+2,2:M+2] = qexact_adv_2d(Xc, Yc, t*dt, simulation)

        # Periodic boundary conditions
        # x direction
        #Q[N+2:N+5,:] = Q[2:5,:]
        #Q[0:2,:]     = Q[N:N+2,:]
        # y direction
        #Q[:,M+2:M+5] = Q[:,2:5]
        #Q[:,0:2]     = Q[:,M:M+2]

        # Output and plot
        if plot:
            # Compute exact solution
            q_exact = qexact_adv_2d(Xc, Yc, t*dt, simulation)

            # Diagnostic computation
            total_mass, mass_change = diagnostics_adv_2d(Q, simulation, total_mass0)

            # Relative errors in different metrics
            error_linf[t], error_l1[t], error_l2[t] = compute_errors(Q[2:N+2,2:M+2], q_exact)

            if error_linf[t] > 10**(1):
                # CFL number
                CFL_x = np.amax(abs(u_edges))*dt/dx
                CFL_y = np.amax(abs(v_edges))*dt/dy
                CFL = np.sqrt(CFL_x**2 + CFL_y**2)
                print('\nStopping due to large errors.')
                print('The CFL number is', CFL)
                exit()

            if t%plotstep == 0:
                # Plot the graph and print diagnostic
                    qmin = str("{:.2e}".format(np.amin(Q)))
                    qmax = str("{:.2e}".format(np.amax(Q)))
                    Uplot[0:nplot,0:nplot], Vplot[0:nplot,0:nplot]  = velocity_adv_2d(xplot, yplot, t*dt, simulation)
                    filename = graphdir+'2d_adv_tc'+str(tc)+'_ic'+str(ic)+'_t'+str(t)+'_N'+str(N)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'.png'
                    title = '2D advection - '+icname+' - time='+str(t*dt)+', CFL='+str(CFL)+',\n N='+str(N)+', '+simulation.fvmethod+', mono = '+simulation.monot+ ', Min = '+ qmin +', Max = '+qmax
                    plot_2dfield_graphs([Q[2:N+2,2:M+2]], Xc, Yc, [Uplot], [Vplot], xplot, yplot,  filename, title)
            print_diagnostics_adv_2d(error_linf[t], error_l1[t], error_l2[t], mass_change, t, Nsteps)

    #---------------------------------------End of time loop---------------------------------------

    if plot:
        # Plot the error graph
        title = simulation.title +'- '+icname+', CFL='+str(CFL)+',\n N='+str(N)+', '+simulation.fvmethod+', mono = '+simulation.monot
        filename = graphdir+'2d_adv_tc'+str(tc)+'_ic'+str(ic)+'_N'+str(N)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'_erros.png'
        plot_time_evolution([error_linf, error_l1, error_l2], Tf, ['$L_\infty}$','$L_1$','$L_2$'], 'Error', filename, title)

        # Plot the solution graph
        qmin = str("{:.2e}".format(np.amin(Q)))
        qmax = str("{:.2e}".format(np.amax(Q)))

        filename = graphdir+'2d_adv_tc'+str(tc)+'_ic'+str(ic)+'_t'+str(t)+'_N'+str(N)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'.png'
        title = '2D advection - '+icname+' - time='+str(t*dt)+', CFL='+str(CFL)+',\n N='+str(N)+', '+simulation.fvmethod+', mono = '+simulation.monot+ ', Min = '+ qmin +', Max = '+qmax
        plot_2dfield_graphs([Q[2:N+2,2:M+2]], Xc, Yc, [Uplot], [Vplot], xplot, yplot, filename, title)
        print('\nGraphs have been ploted in '+ graphdir)
        print('Error evolution is shown in '+filename)

    else:
        # Compute exact solution
        q_exact = qexact_adv_2d(Xc, Yc, Tf, simulation)

        # Relative errors in different metrics
        error_inf, error_1, error_2 = compute_errors(Q[2:N+2,2:M+2], q_exact)

        # Plot the solution graph
        qmin = str("{:.2e}".format(np.amin(Q)))
        qmax = str("{:.2e}".format(np.amax(Q)))
        filename = graphdir+'2d_adv_tc'+str(tc)+'_ic'+str(ic)+'_t'+str(t)+'_N'+str(N)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'.png'
        title = '2D advection - '+icname+' - time='+str(t*dt)+', CFL='+str(CFL)+',\n N='+str(N)+', '+simulation.fvmethod+', mono = '+simulation.monot+ ', Min = '+ qmin +', Max = '+qmax
        plot_2dfield_graphs([Q[2:N+2,2:M+2]], Xc, Yc, [Uplot], [Vplot], xplot, yplot, filename, title)
        return error_inf, error_1, error_2
