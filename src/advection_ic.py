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
        z = np.sin(2.0*X)*np.sin(2.0*Y) + 1.0

    elif simulation.ic == 2:
        z = np.exp(-5.0*(np.sin(X*0.5))**2)
        z = z*np.exp(-5.0*(np.sin(Y))**2)

    elif simulation.ic == 3:
        maskx = np.logical_and(X>=-0.25*np.pi,X<=0.25*np.pi)
        masky = np.logical_and(Y>=-0.125*np.pi,Y<=0.125*np.pi)
        z = x*0
        z[np.logical_and(maskx, masky)] = 1.0

    elif simulation.ic == 4:
        A = 0.2
        Lx = 2*np.pi
        x0 = -1*Lx/12.0
        y0 = 0.0
        x1 =  1*Lx/12.0
        y1 = 0.0
        z0 = 0.95*np.exp(-((x-x0)**2 + (y-y0)**2)/A)
        z1 = 0.95*np.exp(-((x-x1)**2 + (y-y1)**2)/A)
        z = z0 + z1

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
    if simulation.vf == 1:
        u = 0.2
    elif simulation.vf == 2:
        phi_hat = 10
        T = 5

        pi = np.pi
        twopi = pi*2.0
        Lx = twopi
        Ly = pi

        arg1 = twopi*(x/Lx - t/T)
        arg2 = pi*y/Ly
        arg3 = pi*t/T
        c1 = (phi_hat/T)*(Lx/(2*np.pi))**2
        u = c1 * (pi/Ly) * (np.sin(arg1))**2 * (2.0*np.cos(arg2)*np.sin(arg2)) * (np.cos(arg3))
        u = u - Lx/T
        u = -u
    return u

####################################################################################
# Velocity field - v component
####################################################################################
def v_velocity_adv_2d(x, y, t, simulation):
    if simulation.vf == 1:
        v = -0.1
    elif simulation.vf == 2:
        phi_hat = 10
        T = 5

        pi = np.pi
        twopi = pi*2.0
        Lx = twopi
        Ly = pi

        arg1 = twopi*(x/Lx - t/T)
        arg2 = pi*y/Ly
        arg3 = pi*t/T
        c1 = (phi_hat/T)*(Lx/(2*np.pi))**2
        v = c1 * (2.0*pi/Lx) * (2.0*np.sin(arg1)*np.cos(arg1)) * (np.cos(arg2))**2 * np.cos(arg3)
        v = -v
    return v
