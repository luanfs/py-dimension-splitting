####################################################################################
#
# Module for computing the time-averaged velocity needed
# for departure point calculation
# Luan Santos 2023
####################################################################################

def time_averaged_velocity(u_edges, v_edges, k, simulation):
    # Compute the velocity needed for the departure point
    if simulation.vf >= 2:
        if simulation.dp == 1:
            u_edges[:,:,0] = u_edges[:,:,0]
            v_edges[:,:,0] = v_edges[:,:,0]
        if simulation.dp == 2:
            u_edges[:,:,0] = 1.5*u_edges[:,:,1]-0.5*u_edges[:,:,0]
            v_edges[:,:,0] = 1.5*v_edges[:,:,1]-0.5*v_edges[:,:,0]
        if simulation.dp == 3:
            if k==1:
                u_edges[:,:,0] = u_edges[:,:,0]
                v_edges[:,:,0] = v_edges[:,:,0]
            elif k==2:
                u_edges[:,:,0] = 1.5*u_edges[:,:,2]-0.5*u_edges[:,:,0]
                v_edges[:,:,0] = 1.5*v_edges[:,:,2]-0.5*v_edges[:,:,0]
            elif k>2:
                u_edges[:,:,0] = (23.0*u_edges[:,:,2]-16.0*u_edges[:,:,1]+5.0*u_edges[:,:,0])/12.0
                v_edges[:,:,0] = (23.0*v_edges[:,:,2]-16.0*v_edges[:,:,1]+5.0*v_edges[:,:,0])/12.0


