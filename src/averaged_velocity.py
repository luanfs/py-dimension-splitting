####################################################################################
#
# Module for computing the time-averaged velocity needed
# for departure point calculation
# Luan Santos 2023
####################################################################################

from advection_ic import u_velocity_adv_2d, v_velocity_adv_2d
import numexpr as ne
import numpy   as np

def time_averaged_velocity(U_pu, U_pv,  Xu, Yu, Xv, Yv, k, t, simulation):
    # Interior grid indexes
    i0   = simulation.i0
    iend = simulation.iend
    j0   = simulation.j0
    jend = simulation.jend
    N = simulation.N
    ng = simulation.ng

    # Compute the velocity needed for the departure point
    if simulation.vf == 1: # constant velocity
        U_pu.u_averaged[:,:] = U_pu.u[:,:]
        U_pv.v_averaged[:,:] = U_pv.v[:,:]
    elif simulation.vf >= 2:
        if simulation.dp == 1:
            U_pu.u_averaged[:,:] = U_pu.u[:,:]
            U_pv.v_averaged[:,:] = U_pv.v[:,:]

        elif simulation.dp == 2:
            dt = simulation.dt
            dto2 = simulation.dto2
            twodt = simulation.twodt

            #----------------------------------------------------
            # First departure point estimate
            xd = Xu[:,:]-dto2*U_pu.u[:,:]

            # Velocity data at edges used for interpolation
            u_interp = 1.5*U_pu.u[:,:] - 0.5*U_pu.u_old[:,:] # extrapolation for time at n+1/2

            # Linear interpolation
            for j in range(0, N+ng):
                U_pu.u_averaged[i0:iend+1,j] = np.interp(xd[i0:iend+1,j], Xu[i0-1:iend+2,j], u_interp[i0-1:iend+2,j])

            #----------------------------------------------------
            # First departure point estimate
            yd = Yv[:,:]-dto2*U_pv.v[:,:]

            # Velocity data at edges used for interpolation
            v_interp = 1.5*U_pv.v[:,:] - 0.5*U_pv.v_old[:,:] # extrapolation for time at n+1/2

            # Linear interpolation
            for i in range(0, N+ng):
                U_pv.v_averaged[i,j0:jend+1] = np.interp(yd[i,j0:jend+1], Yv[i,j0-1:jend+2], v_interp[i,j0-1:jend+2])

        elif simulation.dp == 3:
            dt = simulation.dt
            dto2 = simulation.dto2
            twodt = simulation.twodt

            #----------------------------------------------------
            simulation.K1u = u_velocity_adv_2d(Xu, Yu, t, simulation)
            simulation.K2u = u_velocity_adv_2d(Xu-dto2*simulation.K1u, Yu,t-dto2, simulation)
            simulation.K3u = u_velocity_adv_2d(Xu-twodt*simulation.K2u+dt*simulation.K1u, Yu, t-dt, simulation)
            U_pu.u_averaged[:,:] = ne.evaluate('(K1u + 4.0*K2u + K3u)/6.0',local_dict=vars(simulation))

           #----------------------------------------------------
            simulation.K1v = v_velocity_adv_2d(Xv, Yv, t, simulation)
            simulation.K2v = v_velocity_adv_2d(Xv, Yv-dto2*simulation.K1v,t-dto2, simulation)
            simulation.K3v = v_velocity_adv_2d(Xv, Yv-twodt*simulation.K2v+dt*simulation.K1v, t-dt, simulation)
            U_pv.v_averaged[:,:] = ne.evaluate('(K1v + 4.0*K2v + K3v)/6.0',local_dict=vars(simulation))

