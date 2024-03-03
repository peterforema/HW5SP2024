#region imports
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
#endregion

#region functions
def ff(Re, rr, CBEQN=False):
    """
    This function calculates the friction factor for a pipe based on the
    notion of laminar, turbulent and transitional flow.
    :param Re: the Reynolds number under question.
    :param rr: the relative pipe roughness (expect between 0 and 0.05)
    :param CBEQN:  boolean to indicate if I should use Colebrook (True) or laminar equation
    :return: the (Darcy) friction factor
    """
    if CBEQN:
        # Colebrook equation
        cb = lambda f: 1.0 / np.sqrt(f) + 2.0 * np.log10(rr / 3.7 + 2.51 / (Re * np.sqrt(f)))
        result = fsolve(cb, 0.02)  # initial guess of 0.02
        return result[0]
    else:
        # Laminar flow equation
        return 64 / Re


def plotMoody(plotPoint=False, pt=(0, 0)):
    """
    This function produces the Moody diagram for a Re range from 1 to 10^8 and
    for relative roughness from 0 to 0.05 (20 steps). The laminar region is described
    by the simple relationship of f=64/Re whereas the turbulent region is described by
    the Colebrook equation.
    """
    # Step 1: create logspace arrays for ranges of Re
    ReValsCB = np.logspace(np.log10(4000), 8, num=100)  # for use with Colebrook equation (Re from 4000 to 10^8)
    ReValsL = np.logspace(np.log10(600), np.log10(2000), num=20)  # for Laminar flow (Re from 600 to 2000)
    ReValsTrans = np.logspace(np.log10(2000), np.log10(4000), num=20)  # for Transition flow (Re from 2000 to 4000)

    # Step 2: create array for range of relative roughnesses
    rrVals = np.linspace(0, 0.05, 20)  # 20 steps from 0 to 0.05

    # Calculate the friction factor in the laminar and transition ranges
    ffLam = np.array([ff(Re, 0, False) for Re in ReValsL])
    ffTrans = np.array([ff(Re, 0, True) for Re in ReValsTrans])

    # Calculate friction factor values for each rr at each Re for turbulent range.
    ffCB = np.array([[ff(Re, rr, True) for Re in ReValsCB] for rr in rrVals])

    # Construct the plot
    plt.figure(figsize=(10, 6))
    plt.loglog(ReValsL, ffLam, 'b-', label='Laminar Flow')
    plt.loglog(ReValsTrans, ffTrans, 'r--', label='Transition Flow')
    for idx, ffCBRow in enumerate(ffCB):
        plt.loglog(ReValsCB, ffCBRow, 'k-', linewidth=0.5,
                   label=f'rr={rrVals[idx]}' if idx in [0, len(rrVals) - 1] else None)
        if idx in [0, len(rrVals) - 1]:
            plt.text(ReValsCB[-1], ffCBRow[-1], f'{rrVals[idx]}')

    plt.xlim(600, 1E8)
    plt.ylim(0.008, 0.1)
    plt.xlabel("Reynolds number (Re)", fontsize=14)
    plt.ylabel("Friction factor (f)", fontsize=14)
    plt.title("Moody Diagram", fontsize=16)
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if plotPoint:
        plt.plot(pt[0], pt[1], 'ro', markersize=5)

    plt.show()

#endregion

#region function calls
if __name__ == "__main__":
    plotMoody()
#endregion

#stem code from Prof Smay