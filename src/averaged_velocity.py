####################################################################################
#
# Module for computing the time-averaged velocity needed
# for departure point calculation
# Luan Santos 2023
####################################################################################

from advection_ic import u_velocity_adv_2d, v_velocity_adv_2d

def time_averaged_velocity(u_edges, v_edges,  Xu, Yu, Xv, Yv, k, t, simulation):
    # Compute the velocity needed for the departure point
    if simulation.vf >= 2:
        if simulation.dp == 2:
            dt = simulation.dt
            dto2 = dt*0.5

            K1u = u_velocity_adv_2d(Xu, Yu, t, simulation)
            K2u = u_velocity_adv_2d(Xu-dto2*K1u, Yu,t-dto2, simulation)
            K3u = u_velocity_adv_2d(Xu-dt*2.0*K2u+dt*K1u, Yu, t-dt, simulation)
            K1v = v_velocity_adv_2d(Xv, Yv, t, simulation)
            K2v = v_velocity_adv_2d(Xv, Yv-dto2*K1v,t-dto2, simulation)
            K3v = v_velocity_adv_2d(Xv, Yv-dt*2.0*K2v+dt*K1v, t-dt, simulation)
            u_edges[:,:] = (K1u + 4.0*K2u + K3u)/6.0
            v_edges[:,:] = (K1v + 4.0*K2v + K3v)/6.0

