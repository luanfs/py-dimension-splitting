####################################################################################
#
# Module for PPM numerical flux computation
#
# References:
# -  Phillip Colella, Paul R Woodward, The Piecewise Parabolic Method (PPM) for gas-dynamical simulations,
# Journal of Computational Physics, Volume 54, Issue 1, 1984, Pages 174-201, ISSN 0021-9991,
# https://doi.org/10.1016/0021-9991(84)90143-8.
#
# -  Carpenter , R. L., Jr., Droegemeier, K. K., Woodward, P. R., & Hane, C. E. (1990).
# Application of the Piecewise Parabolic Method (PPM) to Meteorological Modeling, Monthly Weather Review, 118(3),
# 586-612. Retrieved Mar 31, 2022,
# from https://journals.ametsoc.org/view/journals/mwre/118/3/1520-0493_1990_118_0586_aotppm_2_0_co_2.xml
#
# Luan da Fonseca Santos - June 2022
# (luan.santos@usp.br)
####################################################################################
import numpy as np

####################################################################################
# PPM flux in x direction
####################################################################################
def numerical_flux_x(F, f_R, f_L, q_R, q_L, dq, q6, u_edges, simulation):
    N = simulation.N
    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    x = u_edges*(simulation.dt/simulation.dx)

    # Flux at left edges
    f_L[1:N+1,:]= q_R[2:N+2,:] - x[0:N]*0.5*(dq[2:N+2,:] - (1.0-2.0/3.0*x[0:N])*q6[2:N+2])
    f_L[0,:] = f_L[N,:]

    # Flux at right edges
    x = -x
    f_R[1:N+1,:] = q_L[3:N+3,:] + x[1:N+1,:]*0.5*(dq[3:N+3,:] + (1.0-2.0/3.0*x[1:N+1,:])*q6[3:N+3,:])
    f_R[0,:] = f_R[N,:] # Periodic bc

    # F - Formula 1.13 from Collela and Woodward 1984)
    F[u_edges >= 0] = f_L[u_edges >= 0]
    F[u_edges <= 0] = f_R[u_edges <= 0]

####################################################################################
# PPM flux in y direction
####################################################################################
def numerical_flux_y(G, g_R, g_L, q_R, q_L, dq, q6, v_edges, simulation):
    M = simulation.M
    # Compute the fluxes (formula 1.12 from Collela and Woodward 1984)
    y = v_edges*(simulation.dt/simulation.dy)

    # Flux at left edges
    g_L[:,1:M+1]= q_R[:,2:M+2] - y[:,0:M]*0.5*(dq[:,2:M+2] - (1.0-2.0/3.0*y[:,0:M])*q6[:,2:M+2])
    g_L[:,0] = g_L[:,M]

    # Flux at right edges
    y = -y
    g_R[:,1:M+1] = q_L[:,3:M+3] + y[:,1:M+1]*0.5*(dq[:,3:M+3] + (1.0-2.0/3.0*y[:,1:M+1])*q6[:,3:M+3])
    g_R[:,0] = g_R[:,M] # Periodic bc

    # G - Formula 1.13 from Collela and Woodward 1984)
    G[v_edges >= 0] = g_L[v_edges >= 0]
    G[v_edges <= 0] = g_R[v_edges <= 0]
