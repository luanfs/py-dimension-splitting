####################################################################################
# Operator splitting implementation
# Luan da Fonseca Santos - June 2022
#
# References:
# Lin, S., & Rood, R. B. (1996). Multidimensional Flux-Form Semi-Lagrangian
# Transport Schemes, Monthly Weather Review, 124(9), 2046-2070, from
# https://journals.ametsoc.org/view/journals/mwre/124/9/1520-0493_1996_124_2046_mffslt_2_0_co_2.xml
#
###################################################################################

#from flux import compute_flux_x, compute_flux_y

####################################################################################
# Flux operator in x direction
# Inputs: Q (average values),
# u_edges (velocity in x direction at edges)
# Formula 2.7 from Lin and Rood 1996
####################################################################################
def F_operator(F, u_edges, flux_x, cx, simulation):
    N = simulation.N
    i0 = simulation.i0
    iend = simulation.iend
    #compute_flux_x(flux_x, Q, u_edges, ax, cx, cx2, simulation)
    F[i0:iend,:] = -(simulation.dt/simulation.dx)*(u_edges[i0+1:iend+1,:]*flux_x[i0+1:iend+1,:] - u_edges[i0:iend,:]*flux_x[i0:iend,:])

####################################################################################
# Flux operator in y direction
# Inputs: Q (average values),
# v_edges (velocity in y direction at edges)
# Formula 2.8 from Lin and Rood 1996
####################################################################################
def G_operator(G, v_edges, flux_y, cy, simulation):
    M = simulation.M
    j0 = simulation.j0
    jend = simulation.jend
    #compute_flux_y(flux_y, Q, v_edges, ay, cy, cy2, simulation)
    G[:,j0:jend] = -(simulation.dt/simulation.dy)*(v_edges[:,j0+1:jend+1]*flux_y[:,j0+1:jend+1] - v_edges[:,j0:jend]*flux_y[:,j0:jend])

