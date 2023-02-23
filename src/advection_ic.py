####################################################################################
#
# Module for advection test case set up (initial condition, exact solution and etc)
#
# Luan da Fonseca Santos - October 2022
# (luan.santos@usp.br)
#
# Deformational flow is from the paper "A class of deformational ﬂow test cases for linear
# transport problems  on the sphere", 2010, Ramachandran D. Nair and Peter H. Lauritzen
#
####################################################################################

import numpy as np
import numexpr as ne
####################################################################################
# Initial condition
####################################################################################
def q0_adv_2d(x, y, simulation):
    q = qexact_adv_2d(x, y, 0, simulation)
    return q

####################################################################################
# Exact solution to the advection problem
####################################################################################
def qexact_adv_2d(x, y, t, simulation):
    x0 = simulation.x0
    xf = simulation.xf
    y0 = simulation.y0
    yf = simulation.yf
    ic = simulation.ic

    if simulation.vf == 1: # constant speed
        u, v = velocity_adv_2d(x, y, t, simulation)
        X = x-u*t
        mask = (X != xf)
        X[mask] = (X[mask]-x0)%(xf-x0) + x0 # maps back to [x0,xf]

        Y = y-v*t
        mask = (Y != yf)
        Y[mask] = (Y[mask]-y0)%(yf-y0) + y0 # maps back to [y0,yf]
    else:
        X = x
        Y = y

    if simulation.ic == 1:
        z = np.sin(2.0*np.pi*X)*np.sin(2.0*np.pi*Y) + 1.0

    elif simulation.ic == 2:
        z = np.exp(-10.0*(np.cos(np.pi*X))**2)
        z = z*np.exp(-10.0*(np.cos(np.pi*Y))**2)

    elif simulation.ic == 3:
        maskx = np.logical_and(X>=0.4,X<=0.6)
        masky = np.logical_and(Y>=0.4,Y<=0.6)
        z = x*0
        z[np.logical_and(maskx, masky)] = 1.0

    elif simulation.ic == 4:
        z1 = np.exp(-10.0*(np.cos(np.pi*(X-0.1)))**2)
        z1 = z1*np.exp(-10.0*(np.cos(np.pi*Y))**2)
        z2 = np.exp(-10.0*(np.cos(np.pi*(X+0.1)))**2)
        z2 = z2*np.exp(-10.0*(np.cos(np.pi*Y))**2)
        z = z1+z2

    elif simulation.ic == 5:
        z = np.ones(np.shape(x))

    return z

####################################################################################
# Velocity field
####################################################################################
def velocity_adv_2d(x, y, t, simulation):
    u = u_velocity_adv_2d(x, y, t, simulation)
    v = v_velocity_adv_2d(x, y, t, simulation)
    return u, v

####################################################################################
# Velocity field - u component
####################################################################################
def u_velocity_adv_2d(x, y, t, simulation):
    pi = np.pi
    if simulation.vf == 1:
        u = 0.2
    elif simulation.vf == 2:
        T = 5
        #u = np.sin(np.pi*x)**2*np.sin(2.0*np.pi*y)*np.cos(np.pi*t/T)
        u = ne.evaluate('sin(pi*x)**2*sin(2.0*pi*y)*cos(pi*t/T)')

    elif simulation.vf == 3:
        phi_hat = 10
        T = 5

        pi = np.pi
        twopi = pi*2.0
        pio2  = pi*0.5
        Lx = twopi
        Ly = pi

        #X = -np.pi + x*2.0*np.pi
        X = ne.evaluate('-pi + x*twopi')
        #Y = -np.pi*0.5 + y*np.pi
        Y = ne.evaluate('-pio2 + y*pi')

        #arg1 = twopi*(X/Lx - t/T)
        arg1 = ne.evaluate('twopi*(X/Lx - t/T)')
        #arg2 = pi*Y/Ly
        arg2 = ne.evaluate('pi*Y/Ly')
        #arg3 = pi*t/T
        arg3 = ne.evaluate('pi*t/T')
        #c1 = (phi_hat/T)*(Lx/(2*np.pi))**2
        c1 = ne.evaluate('(phi_hat/T)*(Lx/(2*pi))**2')
        #u = c1 * (pi/Ly) * (np.sin(arg1))**2 * (2.0*np.cos(arg2)*np.sin(arg2)) * (np.cos(arg3))
        u = ne.evaluate('c1*(pi/Ly)*(sin(arg1))**2*(2.0*cos(arg2)*sin(arg2))*(cos(arg3))')
        #u = u - Lx/T
        u = ne.evaluate('u - Lx/T')
        u = -u/twopi
    return u

####################################################################################
# Velocity field - v component
####################################################################################
def v_velocity_adv_2d(x, y, t, simulation):
    pi = np.pi
    if simulation.vf == 1:
        v = -0.2
    elif simulation.vf == 2:
        T = 5.0
        #v = -np.sin(np.pi*y)**2*np.sin(2.0*np.pi*x)*np.cos(np.pi*t/T)
        v = -ne.evaluate('sin(pi*y)**2*sin(2.0*pi*x)*cos(pi*t/T)')
    elif simulation.vf == 3:
        phi_hat = 10
        T = 5

        pi = np.pi
        twopi = pi*2.0
        pio2 = pi*0.5
        Lx = twopi
        Ly = pi

        #X = -np.pi + x*2.0*np.pi
        X = ne.evaluate('-pi + x*twopi')
        #Y = -np.pi*0.5 + y*np.pi
        Y = ne.evaluate('-pio2 + y*pi')

        #arg1 = twopi*(X/Lx - t/T)
        arg1 = ne.evaluate('twopi*(X/Lx - t/T)')
        #arg2 = pi*Y/Ly
        arg2 = ne.evaluate('pi*Y/Ly')
        #arg3 = pi*t/T
        arg3 = ne.evaluate('pi*t/T')

        #c1 = (phi_hat/T)*(Lx/(2*np.pi))**2
        c1 = ne.evaluate('(phi_hat/T)*(Lx/(2*pi))**2')
        #v = c1 * (2.0*pi/Lx) * (2.0*np.sin(arg1)*np.cos(arg1)) * (np.cos(arg2))**2 * np.cos(arg3)
        v = ne.evaluate('c1*(2.0*pi/Lx)*(2.0*sin(arg1)*cos(arg1))*((cos(arg2))**2)*cos(arg3)')
        v = -v/pi
    return v
