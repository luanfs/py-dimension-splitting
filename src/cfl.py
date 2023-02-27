####################################################################################
#
# Module for CFL number computation
####################################################################################
import numpy as np
import numexpr as ne

####################################################################################
# CFL number in x direction
####################################################################################
def cfl_x(u_edges, simulation):
    dt = simulation.dt
    dx = simulation.dx
    cx = ne.evaluate('u_edges*dt/dx')
    return cx

####################################################################################
# CFL number in x direction
####################################################################################
def cfl_y(v_edges, simulation):
    dt = simulation.dt
    dy = simulation.dy
    cy = ne.evaluate('v_edges*dt/dy')
    return cy
