####################################################################################
#
# Module for diagnostic computation routines
#
# Luan da Fonseca Santos - October 2022
# (luan.santos@usp.br)
####################################################################################

import numpy as np

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


