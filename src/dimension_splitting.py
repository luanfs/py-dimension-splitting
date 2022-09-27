####################################################################################
# Dimension splitting operators implementation
# Luan da Fonseca Santos - June 2022
#
# References:
# Lin, S., & Rood, R. B. (1996). Multidimensional Flux-Form Semi-Lagrangian
# Transport Schemes, Monthly Weather Review, 124(9), 2046-2070, from
# https://journals.ametsoc.org/view/journals/mwre/124/9/1520-0493_1996_124_2046_mffslt_2_0_co_2.xml
#
###################################################################################

from flux import compute_flux_x, compute_flux_y

####################################################################################
# Flux operator in x direction
# Inputs: Q (average values),
# u_edges (velocity in x direction at edges)
# Formula 2.7 from Lin and Rood 1996
####################################################################################
def F_operator(F, Q, u_edges, flux_x, ax, cx, cx2, simulation):
    N = simulation.N
    compute_flux_x(flux_x, Q, u_edges, ax, cx, cx2, simulation)
    F[3:N+3,:] = -(simulation.dt/simulation.dx)*(u_edges[4:N+4,:]*flux_x[4:N+4,:] - u_edges[3:N+3,:]*flux_x[3:N+3,:])

####################################################################################
# Flux operator in y direction
# Inputs: Q (average values),
# v_edges (velocity in y direction at edges)
# Formula 2.8 from Lin and Rood 1996
####################################################################################
def G_operator(G, Q, v_edges, flux_y, ay, cy, cy2, simulation):
    M = simulation.M
    compute_flux_y(flux_y, Q, v_edges, ay, cy, cy2, simulation)
    G[:, 3:M+3] = -(simulation.dt/simulation.dy)*(v_edges[:,4:M+4]*flux_y[:,4:M+4] - v_edges[:,3:M+3]*flux_y[:,3:M+3])

