# region imports
import HW5a as pta
import random as rnd
from matplotlib import pyplot as plt
# endregion

#region functions

def ffPoint(Re, rr):
    """
    This function takes Re and rr as parameters and outputs a friction factor according to the following:
    1.  if Re>4000 use Colebrook Equation
    2.  if Re<2000 use f=64/Re
    3.  else calculate a probabilistic friction factor where the distribution has a mean midway between the prediction
        of the f=64/Re and Colebrook Equations and a standard deviation of 20% of this mean
    :param Re:  the Reynolds number
    :param rr:  the relative roughness
    :return:  the friction factor
    """
    if Re >= 4000:
        return pta.ff(Re, rr, CBEQN=True)
    elif Re <= 2000:
        return pta.ff(Re, rr)
    else:
        # Calculate the predictions for Colebrook and Laminar equations in the transition region
        CBff = pta.ff(Re, rr, CBEQN=True)  # Prediction of Colebrook Equation in Transition region
        Lamff = pta.ff(Re, rr)  # Prediction of Laminar Equation in Transition region
        mean = (CBff + Lamff) / 2
        sig = 0.2 * mean
        # Use normalvariate to select a number randomly from a normal distribution
        return rnd.normalvariate(mean, sig)

def PlotPoint(Re,f):
    pta.plotMoody(plotPoint=True, pt=(Re,f))

def main():
    Re=float(input("Enter the Reynolds number:  "))
    rr=float(input("Enter the relative roughness:  "))
    f=ffPoint(Re, rr)
    PlotPoint(Re, f)
# endregion

# region function calls
if __name__=="__main__":
    main()
# endregion

#stem code from Prof Smay