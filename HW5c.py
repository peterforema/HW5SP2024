# region imports
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# endregion

# region functions
def ode_system(t, X, *params):
    '''
    The ode system is defined in terms of state variables.
    I have as unknowns:
    x: position of the piston (This is not strictly needed unless I want to know x(t))
    xdot: velocity of the piston
    p1: pressure on right of piston
    p2: pressure on left of the piston
    For initial conditions, we see: x=x0=0, xdot=0, p1=p1_0=p_a, p2=p2_0=p_a
    :param X: The list of state variables.
    :param t: The time for this instance of the function.
    :param params: the list of physical constants for the system.
    :return: The list of derivatives of the state variables.
    '''
    #unpack the parameters
    A, Cd, ps, pa, V, beta, rho, Kvalve, m, y=params

    #state variables
    x = X[0]
    xdot = X[1]
    p1 = X[2]
    p2 = X[3]

    #calculate derivatives
    xddot = xdot
    p1dot = Kvalve * (p2 - p1)
    p2dot = (ps - p2) / (beta * rho * V) - Cd * A * np.sqrt(2 * rho * (p1 - p2) / rho) / (2 * A)

    #return the list of derivatives of the state variables
    return [xddot, p1dot, p2dot, xdot]

def main():
    # After some trial and error, I found all the action seems to happen in the first 0.02 seconds
    t=np.linspace(0,0.02,200)
    myargs=(4.909E-4, 0.6, 1.4E7,1.0E5,1.473E-4,2.0E9,850.0,2.0E-5,30, 0.002)
    pa = myargs[3]
    ic = [0, 0, pa, pa]
    sln=solve_ivp(lambda t, X: ode_system(t, X, *myargs), (0, 0.02), ic, t_eval=t)

    #unpack result into meaningful names
    xvals=sln.y[0]
    xdot=sln.y[1]
    p1=sln.y[2]
    p2=sln.y[3]

    #plot the result
    plt.subplot(2, 1, 1)
    plt.plot(t, xvals, 'r-', label='$x$')
    plt.ylabel('$x$')
    plt.legend(loc='upper left')

    ax2=plt.twinx()
    ax2.plot(t, xdot, 'b-', label='$\dot{x}$')
    plt.ylabel('$\dot{x}$')
    plt.legend(loc='lower right')

    plt.subplot(2,1,2)
    plt.plot(t, p1, 'b-', label='$P_1$')
    plt.plot(t, p2, 'r-', label='$P_2$')
    plt.legend(loc='lower right')
    plt.xlabel('Time, s')
    plt.ylabel('$P_1, P_2 (Pa)$')

    plt.show()
# endregion

# region function calls
if __name__=="__main__":
    main()
# endregion

#stem code from Prof Smay