####################################################################################
#
# Module for output routines.
#
# Luan da Fonseca Santos - October 2022
# (luan.santos@usp.br)
####################################################################################

import numpy as np
from errors         import compute_errors
from advection_ic   import qexact_adv_2d, velocity_adv_2d
from parameters_2d  import graphdir
from plot           import plot_2dfield_graphs
from diagnostics    import diagnostics_adv_2d

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
def output_adv(Xc, Yc, simulation, Q, div, error_linf, error_l1, error_l2, plot, k, t, Nsteps, plotstep, total_mass0, CFL, divtest_flag):
    N  = simulation.N    # Number of cells in x direction

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

        if plot and k>0 and (not divtest_flag):
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
            vf = simulation.vf
            tc = simulation.tc
            icname = simulation.icname

            # Plotting variables
            nplot = 20
            xplot = np.linspace(x0, xf, nplot)
            yplot = np.linspace(y0, yf, nplot)
            xplot, yplot = np.meshgrid(xplot, yplot)
            Uplot = np.zeros((nplot, nplot))
            Vplot = np.zeros((nplot, nplot))

            qmin = np.amin(Q)
            qmax = np.amax(Q)
            qmin = str("{:.2e}".format(qmin))
            qmax = str("{:.2e}".format(qmax))
            time = str("{:.2e}".format(t))
            CFL  = str("{:.2e}".format(CFL))

            # colorbar range
            if simulation.ic == 1:
                Qmin = -0.1
                Qmax =  2.1
            elif simulation.ic == 2 or simulation.ic == 4:
                Qmin = -0.2
                Qmax =  1.1
            elif simulation.ic == 3:
                Qmin = -0.2
                Qmax =  1.2
            elif simulation.ic == 5:
                Qmin =  0.99
                Qmax =  1.01

            Uplot[0:nplot,0:nplot], Vplot[0:nplot,0:nplot]  = velocity_adv_2d(xplot, yplot, t, simulation)

            # Plot scalar field
            if (not divtest_flag):
                filename = graphdir+'2d_adv_tc'+str(tc)+'_ic'+str(ic)+'_vf'+str(vf)+'_t'+str(k)+'_N'+str(N)+\
                '_'+simulation.opsplit_name+'_'+simulation.recon_name+'_'+simulation.dp_name
                title = icname+', velocity = '+str(vf)+' - time='+time+', CFL='+str(CFL)+'\nN='+str(N)+', '+\
                simulation.opsplit_name+', ' +simulation.recon_name+', '+simulation.dp_name+\
                ' , Min = '+ qmin +', Max = '+qmax
                plot_2dfield_graphs([Q[i0:iend,j0:jend]], [Qmin], [Qmax], ['jet'], Xc, Yc, [Uplot], [Vplot], xplot, yplot, filename, title)

                # plot final error
                if k == Nsteps:
                    filename = graphdir+'2d_adv_error_tc'+str(tc)+'_ic'+str(ic)+'_vf'+str(vf)+'_t'+str(k)+'_N'+str(N)+\
                    '_split'+simulation.opsplit_name+'_'+simulation.recon_name+'_'+simulation.dp_name
                    title = icname+', velocity = '+str(vf)+' - time='+time+', CFL='+str(CFL)+'\nN='+str(N)+', '+\
                    simulation.opsplit_name+', ' +simulation.recon_name+', '+simulation.dp_name
                    error = q_exact-Q[i0:iend,j0:jend]
                    emin = np.amin(error)
                    emax = np.amax(error)
                    plot_2dfield_graphs([error], [emin], [emax], ['seismic'], Xc, Yc, [Uplot], [Vplot], xplot, yplot, filename, title)

            # Plot  error div field
            else:
                filename = graphdir+'2d_adv_error_tc'+str(tc)+'_ic'+str(ic)+'_vf'+str(vf)+'_t'+str(k)+'_N'+str(N)+\
                '_'+simulation.opsplit_name+'_'+simulation.recon_name+'_'+simulation.dp_name
                title = 'Divergence error, velocity = '+str(vf)+', CFL='+str(CFL)+', N='+str(N)+'\n '\
                +simulation.opsplit_name+', '+simulation.recon_name+', '+simulation.dp_name
                error = div[i0:iend,j0:jend]
                emin = np.amin(error)
                emax = np.amax(error)
                plot_2dfield_graphs([error], [emin], [emax], ['seismic'], Xc, Yc, [Uplot], [Vplot], xplot, yplot, filename, title)

