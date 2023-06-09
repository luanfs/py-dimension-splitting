####################################################################################
#
# Module for computing the time-averaged velocity needed
# for departure point calculation
# Luan Santos 2023
####################################################################################

from advection_ic import u_velocity_adv_2d, v_velocity_adv_2d
import numexpr as ne
import numpy   as np

def time_averaged_velocity(U_pu, U_pv, t, simulation):
    # Interior grid indexes
    i0   = simulation.i0
    iend = simulation.iend
    j0   = simulation.j0
    jend = simulation.jend
    N = simulation.N
    ng = simulation.ng

    u = U_pu.u[i0:iend+1,:]
    U_pu.upos = ne.evaluate('u>=0')
    U_pu.uneg = ~U_pu.upos

    v = U_pv.v[:,j0:jend+1]
    U_pv.vpos = ne.evaluate('v>=0')
    U_pv.vneg = ~U_pv.vpos

    # Compute the velocity needed for the departure point
    if simulation.vf == 1: # constant velocity
        simulation.U_pu.u_averaged[:,:] = simulation.U_pu.u[:,:]
        simulation.U_pv.v_averaged[:,:] = simulation.U_pv.v[:,:]
    elif simulation.vf >= 2:
        if simulation.dp == 1:
            simulation.U_pu.u_averaged[:,:] = simulation.U_pu.u[:,:]
            simulation.U_pv.v_averaged[:,:] = simulation.U_pv.v[:,:]

        elif simulation.dp == 2:
            dt = simulation.dt
            dto2 = simulation.dto2
            twodt = simulation.twodt

            #----------------------------------------------------
            # x direction
            # Velocity data at edges used for interpolation
            u_interp = ne.evaluate('1.5*u - 0.5*u_old', local_dict=vars(simulation.U_pu)) # extrapolation for time at n+1/2

            # Linear interpolation
            upos, uneg = U_pu.upos, U_pu.uneg
            a = (dto2*simulation.U_pu.u[i0:iend+1,:])/simulation.dx
            ap, an = a[upos], a[uneg]
            u1, u2 = u_interp[i0-1:iend,:][upos], u_interp[i0:iend+1,:][upos] 
            u3, u4 = u_interp[i0:iend+1:,:][uneg], u_interp[i0+1:iend+2,:][uneg]
            simulation.U_pu.u_averaged[i0:iend+1,:][upos] = ne.evaluate('(1.0-ap)*u2 + ap*u1')
            simulation.U_pu.u_averaged[i0:iend+1,:][uneg] = ne.evaluate('-an*u4 + (1.0+an)*u3')

            #----------------------------------------------------
            # y direction
            # Velocity data at edges used for interpolation
            v_interp = ne.evaluate('1.5*v - 0.5*v_old', local_dict=vars(simulation.U_pv)) # extrapolation for time at n+1/2
            # Linear interpolation
            vpos, vneg = U_pv.vpos, U_pv.vneg
            a = (dto2*simulation.U_pv.v[:,j0:jend+1])/simulation.dy
            ap, an = a[vpos], a[vneg]
            v1, v2 = v_interp[:,j0-1:jend][vpos], v_interp[:,j0:jend+1][vpos] 
            v3, v4 = v_interp[:,j0:jend+1][vneg], v_interp[:,j0+1:jend+2][vneg]
            simulation.U_pv.v_averaged[:,j0:jend+1][vpos] = ne.evaluate('(1.0-ap)*v2 + ap*v1')
            simulation.U_pv.v_averaged[:,j0:jend+1][vneg] = ne.evaluate('-an*v4 + (1.0+an)*v3')


        elif simulation.dp == 3:
            dt = simulation.dt
            dto2 = simulation.dto2
            twodt = simulation.twodt

            #----------------------------------------------------
            simulation.K1u = u_velocity_adv_2d(simulation.Xu, simulation.Yu, t, simulation)
            simulation.K2u = u_velocity_adv_2d(simulation.Xu-dto2*simulation.K1u, simulation.Yu,t-dto2, simulation)
            simulation.K3u = u_velocity_adv_2d(simulation.Xu-twodt*simulation.K2u+dt*simulation.K1u, simulation.Yu, t-dt, simulation)
            U_pu.u_averaged[:,:] = ne.evaluate('(K1u + 4.0*K2u + K3u)/6.0',local_dict=vars(simulation))

           #----------------------------------------------------
            simulation.K1v = v_velocity_adv_2d(simulation.Xv, simulation.Yv, t, simulation)
            simulation.K2v = v_velocity_adv_2d(simulation.Xv, simulation.Yv-dto2*simulation.K1v,t-dto2, simulation)
            simulation.K3v = v_velocity_adv_2d(simulation.Xv, simulation.Yv-twodt*simulation.K2v+dt*simulation.K1v, t-dt, simulation)
            U_pv.v_averaged[:,:] = ne.evaluate('(K1v + 4.0*K2v + K3v)/6.0',local_dict=vars(simulation))

