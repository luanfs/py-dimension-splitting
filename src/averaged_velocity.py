####################################################################################
#
# Module for computing the time-averaged velocity needed
# for departure point calculation
# Luan Santos 2023
####################################################################################

from advection_ic import u_velocity_adv_2d, v_velocity_adv_2d
import numexpr as ne
def time_averaged_velocity(u_edges, v_edges,  Xu, Yu, Xv, Yv, k, t, simulation):
    # Compute the velocity needed for the departure point
    if simulation.vf >= 2:
        if simulation.dp == 2:
            dt = simulation.dt
            dto2 = simulation.dto2
            twodt = simulation.twodt

            simulation.K1u = u_velocity_adv_2d(Xu, Yu, t, simulation)
            simulation.K2u = u_velocity_adv_2d(Xu-dto2*simulation.K1u, Yu,t-dto2, simulation)
            simulation.K3u = u_velocity_adv_2d(Xu-twodt*simulation.K2u+dt*simulation.K1u, Yu, t-dt, simulation)

            simulation.K1v = v_velocity_adv_2d(Xv, Yv, t, simulation)
            simulation.K2v = v_velocity_adv_2d(Xv, Yv-dto2*simulation.K1v,t-dto2, simulation)
            simulation.K3v = v_velocity_adv_2d(Xv, Yv-twodt*simulation.K2v+dt*simulation.K1v, t-dt, simulation)
            u_edges[:,:] = ne.evaluate('(K1u + 4.0*K2u + K3u)/6.0',local_dict=vars(simulation))
            v_edges[:,:] = ne.evaluate('(K1v + 4.0*K2v + K3v)/6.0',local_dict=vars(simulation))

