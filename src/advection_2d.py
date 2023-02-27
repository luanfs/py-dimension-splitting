####################################################################################
#
# Advection module
# Luan da Fonseca Santos - April 2022
# Solves the 2d advection equation  with periodic boundary conditions
# The initial condition Q(x,y,0) is given in the module advection_ic.py
#
# References:
# Lin, S., & Rood, R. B. (1996). Multidimensional Flux-Form Semi-Lagrangian
# Transport Schemes, Monthly Weather Review, 124(9), 2046-2070, from
# https://journals.ametsoc.org/view/journals/mwre/124/9/1520-0493_1996_124_2046_mffslt_2_0_co_2.xml
#
####################################################################################

import numpy as np
from plot                import plot_2dfield_graphs
from diagnostics         import diagnostics_adv_2d
from output              import print_diagnostics_adv_2d, output_adv
from parameters_2d       import graphdir
from errors import *
from advection_timestep  import adv_timestep
from advection_vars      import adv_vars
from advection_ic        import velocity_adv_2d, u_velocity_adv_2d, v_velocity_adv_2d

def adv_2d(simulation, plot, divtest_flag):
    dt = simulation.dt   # Time step
    Tf = simulation.Tf   # Total period definition

    # Grid interior indexes
    i0 = simulation.i0
    iend = simulation.iend
    j0 = simulation.j0
    jend = simulation.jend

    # Number of time steps
    Nsteps = int(Tf/dt)

    # Plotting var
    plotstep = int(Nsteps/5)

    # Get vars
    Q, div, U_pu, U_pv, px, py, cx, cy,\
    Xc, Yc, Xu, Yu, Xv, Yv, CFL = adv_vars(simulation)

    if (divtest_flag):
        Nsteps = 1
        plotstep = 1

    # Compute initial mass
    total_mass0, _ = diagnostics_adv_2d(Q, simulation, 1.0)

    # Error variables
    error_linf = np.zeros(Nsteps+1)
    error_l1   = np.zeros(Nsteps+1)
    error_l2   = np.zeros(Nsteps+1)

    # Initial plotting
    if (not divtest_flag):
        output_adv(Xc[i0:iend,j0:jend], Yc[i0:iend,j0:jend], simulation, Q, div, error_linf, error_l1, error_l2, plot, 0, 0.0, Nsteps, plotstep, total_mass0, CFL, divtest_flag)

    # Time looping
    for k in range(1, Nsteps+1):
        # Time
        t = k*dt

        # Applies a time step
        adv_timestep(Q, div, U_pu, U_pv, px, py, cx, cy, Xu, Yu, Xv, Yv, t, k, simulation)

        # Output and plot
        output_adv(Xc[i0:iend,j0:jend], Yc[i0:iend,j0:jend], simulation, Q, div, error_linf, error_l1, error_l2, plot, k, t, Nsteps, plotstep, total_mass0, CFL, divtest_flag)

        # Updates for next time step
        if simulation.vf >= 2:
            # Velocity update
            U_pu.u[:,:] = u_velocity_adv_2d(Xu, Yu, t, simulation)
            U_pv.v[:,:] = v_velocity_adv_2d(Xv, Yv, t, simulation)
    #---------------------------------------End of time loop---------------------------------------

    if plot:
        CFL  = str("{:.2e}".format(CFL))
        # Plot the error evolution graph
        title = simulation.title +'- '+simulation.icname+', CFL='+str(CFL)+',\n N='+str(simulation.N)+', '+simulation.opsplit_name+', '+simulation.recon_name+', '+simulation.dp_name
        filename = graphdir+'2d_adv_tc'+str(simulation.tc)+'_ic'+str(simulation.ic)+'_N'+str(simulation.N)+'_'+simulation.opsplit_name+'_'+simulation.recon_name+'_'+simulation.dp_name+'_errors.png'
        plot_time_evolution([error_linf, error_l1, error_l2], Tf, ['$L_\infty}$','$L_1$','$L_2$'], 'Error', filename, title)
    return error_linf[Nsteps], error_l1[Nsteps], error_l2[Nsteps]
