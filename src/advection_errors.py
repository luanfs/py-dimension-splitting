###################################################################################
#
# Module to compute the error convergence in L_inf, L1 and L2 norms
# for the advection equation using the Piecewise Parabolic Method (PPM)
# Luan da Fonseca Santos - April 2022
#
####################################################################################

from advection_2d import adv_2d
import numpy as np
from errors import *
from parameters_2d import simulation_adv_par_2d, graphdir
from advection_ic  import velocity_adv_2d

def error_analysis_adv2d(simulation):
    # Initial condition
    ic = simulation.ic

    # Velocity
    vf = simulation.vf

    # Reconstruction method
    recon = simulation.recon

    # Departure point method
    dp = simulation.dp

    # Operator splitting
    opsplit = simulation.opsplit

    # Test case
    tc = simulation.tc

    # Interval
    x0 = simulation.x0
    xf = simulation.xf
    y0 = simulation.y0
    yf = simulation.yf

    # Number of tests
    Ntest = 3

    # Number of cells
    N = np.zeros(Ntest)
    N[0] = 48
    M = np.zeros(Ntest)
    M[0] = 48

    # Timesteps
    dt = np.zeros(Ntest)

    u, v = velocity_adv_2d(x0, y0, 0, simulation)

    # Period
    if simulation.vf==1: # constant velocity
        Tf = 5.0
        dt[0] = 0.08
    elif simulation.vf == 2: # variable velocity
        Tf = 5.0
        dt[0] = 0.05
    elif simulation.vf == 3: # variable velocity
        Tf = 5.0
        dt[0] = 0.025
    else:
        exit()

    # Errors array
    recons = (1,1)
    deps   = (1,2)
    split  = (3,1)
    #recons = (simulation.recon,)
    #deps = (simulation.dp,)
    #split = (simulation.opsplit,)

    K = len(deps)
    error_linf = np.zeros((Ntest, K))
    error_l1   = np.zeros((Ntest, K))
    error_l2   = np.zeros((Ntest, K))

    # Compute number of cells and time step for each simulation
    for i in range(1, Ntest):
        N[i]  = N[i-1]*2.0
        M[i]  = M[i-1]*2.0
        dt[i] = dt[i-1]*0.5

    # Let us test and compute the error
    d = 0
    for k in range(0,len(deps)):
        recon = recons[k]
        dp = deps[k]
        opsplit = split[k]
        for i in range(0, Ntest):
            # Update simulation parameters
            simulation = simulation_adv_par_2d(int(N[i]), int(M[i]), dt[i], Tf, ic, vf, tc, recon, dp, opsplit)

            # Run advection routine and get the errors
            error_linf[i,k], error_l1[i,k], error_l2[i,k] =  adv_2d(simulation, False, False)
            print('\nParameters: N = '+str(int(N[i]))+', dt = '+str(dt[i]),', recon = ', recon,', split = ',opsplit, ', dp = ', dp)

            # Output
            print_errors_simul(error_linf[:,k], error_l1[:,k], error_l2[:,k], i)

    # plot errors for different all schemes in  different norms
    error_list = [error_linf, error_l1, error_l2]
    norm_list  = ['linf','l1','l2']
    norm_title  = [r'$L_{\infty}$',r'$L_1$',r'$L_2$']
 
    e = 0
    for errors in error_list:
        emin, emax = np.amin(errors), np.amax(errors)
     
        # convergence rate min/max
        n = len(errors)
        CR = np.abs(np.log(errors[1:n])-np.log(errors[0:n-1]))/np.log(2.0)
        CRmin, CRmax = np.amin(CR), np.amax(CR)       

        title = simulation.title + ' - ' + simulation.icname+', vf='+ str(simulation.vf)+\
            ', norm='+norm_title[e]
        filename = graphdir+'1d_adv_tc'+str(tc)+'_ic'+str(ic)+'_vf'+str(vf)+\
            '_norm'+norm_list[e]+'_parabola_errors.pdf'

        names = []
        errors_list = []
        for k in range(0,len(deps)):
            recon = recons[k]
            dp = deps[k]
            ops = split[k]
            names.append('slipt'+str(ops)+'PPM'+str(recon)+' - RK'+str(dp)) 
            errors_list.append(errors[:,k])
        plot_errors_loglog(N, errors_list, names, filename, title, emin, emax)
        e = e+1
