####################################################################################
#
# Module for diagnostic computation and output routines
#
# Luan da Fonseca Santos - October 2022
# (luan.santos@usp.br)
####################################################################################

import numpy as np
from errors import compute_errors
from advection_ic  import qexact_adv_2d, velocity_adv_2d
from parameters_2d import graphdir
from plot import plot_2dfield_graphs

####################################################################################
# Diagnostic variables computation
####################################################################################
def diagnostics_adv_2d(Q_average, simulation, total_mass0):
    N  = simulation.N    # Number of cells in x direction
    M  = simulation.M    # Number of cells in y direction
    dx = simulation.dx   # Grid spacing in x direction
    dy = simulation.dy   # Grid spacing in y direction

    # Grid interior indexes
    i0 = simulation.i0
    iend = simulation.iend
    j0 = simulation.j0
    jend = simulation.jend

    total_mass =  np.sum(Q_average[i0:iend,j0:jend])*dx*dy  # Compute new mass
    if abs(total_mass0)>10**(-10):
        mass_change = abs(total_mass0-total_mass)/abs(total_mass0)
    else:
        mass_change = abs(total_mass0-total_mass)
    return total_mass, mass_change

####################################################################################
# Print the diagnostics variables on the screen
####################################################################################
def print_diagnostics_adv_2d(error_linf, error_l1, error_l2, mass_change, t, Nsteps):
    print('\nStep', t, 'from', Nsteps)
    print('Error (Linf, L1, L2) :',"{:.2e}".format(error_linf), "{:.2e}".format(error_l1), "{:.2e}".format(error_l2))
    print('Total mass variation:', "{:.2e}".format(mass_change))

####################################################################################
# Output and plot
####################################################################################
def output_adv(Xc, Yc, simulation, Q, error_linf, error_l1, error_l2, plot, k, t, Nsteps, plotstep, total_mass0, CFL):
    N  = simulation.N    # Number of cells in x direction
    M  = simulation.M    # Number of cells in y direction
    dx = simulation.dx
    dt = simulation.dt

    # Grid interior indexes
    i0 = simulation.i0
    iend = simulation.iend
    j0 = simulation.j0
    jend = simulation.jend

    if plot or k==Nsteps:
        # Compute exact solution
        q_exact = qexact_adv_2d(Xc, Yc, t, simulation)

        # Diagnostic computation
        total_mass, mass_change = diagnostics_adv_2d(Q, simulation, total_mass0)

        # Relative errors in different metrics
        error_linf[k], error_l1[k], error_l2[k] = compute_errors(Q[i0:iend,j0:jend], q_exact)

        if error_linf[k] > 10**(2):
            print('\nStopping due to large errors.')
            exit()

        # Diagnostic computation
        total_mass, mass_change = diagnostics_adv_2d(Q, simulation, total_mass0)

        if plot and k>0:
            # Print diagnostics on the screen
            print_diagnostics_adv_2d(error_linf[k], error_l1[k], error_l2[k], mass_change, k, Nsteps)

        if k%plotstep==0 or k==0 or k==Nsteps :
            # Plot the graph and print diagnostic
            # simulation parameters
            x0 = simulation.x0
            xf = simulation.xf
            y0 = simulation.y0
            yf = simulation.yf
            ic = simulation.ic
            tc = simulation.tc
            icname = simulation.icname

            # Plotting variables
            nplot = 20
            xplot = np.linspace(x0, xf, nplot)
            yplot = np.linspace(y0, yf, nplot)
            xplot, yplot = np.meshgrid(xplot, yplot)
            Uplot = np.zeros((nplot, nplot))
            Vplot = np.zeros((nplot, nplot))

            qmin = str(np.amin(Q))
            qmax = str(np.amax(Q))

            # colorbar range
            if simulation.ic == 1:
                Qmin = -0.1
                Qmax =  2.1
            elif simulation.ic == 2 or simulation.ic == 4:
                Qmin = -0.2
                Qmax =  1.0
            elif simulation.ic == 3:
                Qmin = -0.3
                Qmax =  1.3
            elif simulation.ic == 5:
                Qmin =  0.95
                Qmax =  1.05
            # Plot
            Uplot[0:nplot,0:nplot], Vplot[0:nplot,0:nplot]  = velocity_adv_2d(xplot, yplot, t, simulation)
            filename = graphdir+'2d_adv_tc'+str(tc)+'_ic'+str(ic)+'_t'+str(k)+'_N'+str(N)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'.png'
            title = '2D advection - '+icname+' - time='+str(t)+', CFL='+str(CFL)+',\n N='+str(N)+', '+simulation.fvmethod+', mono = '+simulation.monot+ ', Min = '+ qmin +', Max = '+qmax
            plot_2dfield_graphs([Q[i0:iend,j0:jend]], [Qmin], [Qmax], ['jet'], Xc, Yc, [Uplot], [Vplot], xplot, yplot, filename, title)
