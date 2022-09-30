####################################################################################
#
# Module for miscellaneous routines
#
# Luan da Fonseca Santos - April 2022
# (luan.santos@usp.br)
####################################################################################

import os
import numpy as np
from parameters_2d import graphdir, qexact_adv_2d, velocity_adv_2d
import matplotlib.pyplot as plt
from errors import *

####################################################################################
# Create a folder
# Reference: https://gist.github.com/keithweaver/562d3caa8650eefe7f84fa074e9ca949
####################################################################################
def createFolder(dir):
   try:
      if not os.path.exists(dir):
         os.makedirs(dir)
   except OSError:
      print ('Error: Creating directory. '+ dir)

####################################################################################
# Create the needed directories
####################################################################################
def createDirs():
   print("--------------------------------------------------------")
   print("Dimension splitting python implementation by Luan Santos - 2022\n")
   # Check directory graphs does not exist
   if not os.path.exists(graphdir):
      print('Creating directory ',graphdir)
      createFolder(graphdir)

   print("--------------------------------------------------------")

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
# Plot the 2d graphs given in the list fields
####################################################################################
def plot_2dfield_graphs(scalar_fields, xplot, yplot, vector_fieldsu, vector_fieldsv, xv, yv, filename, title):
    n = len(scalar_fields)
    for k in range(0, n):
        plt.contourf(xplot, yplot, scalar_fields[k], cmap='jet', levels=100)
        plt.colorbar(orientation='vertical', fraction=0.046, pad=0.04)
        rnorm = 1.0
        #rnorm = np.sqrt(vector_fieldsu[k]**2 + vector_fieldsv[k]**2)
        plt.quiver(xv, yv, vector_fieldsu[k]/rnorm, vector_fieldsv[k]/rnorm)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.savefig(filename)
        plt.close()
        
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

            # Plot
            qmin = str("{:.2e}".format(np.amin(Q)))
            qmax = str("{:.2e}".format(np.amax(Q)))
            Uplot[0:nplot,0:nplot], Vplot[0:nplot,0:nplot]  = velocity_adv_2d(xplot, yplot, t, simulation)
            filename = graphdir+'2d_adv_tc'+str(tc)+'_ic'+str(ic)+'_t'+str(k)+'_N'+str(N)+'_'+simulation.fvmethod+'_mono'+simulation.monot+'.png'
            title = '2D advection - '+icname+' - time='+str(t)+', CFL='+str(CFL)+',\n N='+str(N)+', '+simulation.fvmethod+', mono = '+simulation.monot+ ', Min = '+ qmin +', Max = '+qmax
            plot_2dfield_graphs([Q[i0:iend,j0:jend]], Xc, Yc, [Uplot], [Vplot], xplot, yplot, filename, title)
