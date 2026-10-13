import numpy as np

def logistic_sigmoid(x, a=1.0, b=1.0, c=0.0, d=0.0):
    """Implementation of the logistic sigmoid function with basic shape defined as y = 1/(1+exp(-x))

    This implementation is an extended version that allows to modify the function by applying
    4 parameters: y = a * 1/(1+exp(b*(-x + c)) + d

    with parameters:
    a: modifies the amplitude of the function
    b: modifies the steepness of the function
    c: translates the function over the x axis
    d: translates the function over the y axis
    """
    r = a * 1 / (1 + np.exp(b * (-x + c))) + d
    return r

class FOTB:
    """Implements partitioning to storage organs

    FOTBa defines the steepness of the function (default 20)
    FOTBb defines the location of the centre point (default 1.0)
    FOTBc defines the fraction allocated to organs at DVS=2 (default 1.0)
    """

    def __init__(self, FOTBa=20, FOTBb=1.2, FOTBc=1.0):
        self.FOTBa = FOTBa
        self.FOTBb = FOTBb
        self.FOTBc = FOTBc

    def __call__(self, x):
        FO = logistic_sigmoid(x, a=self.FOTBc, b=self.FOTBa, c=self.FOTBb)
        if FO >= 0.99:
            return 1.0
        elif FO <= 0.01:
            return 0.0
        else:
            return FO


class FLTB:
    """Implements partitioning to leaves

    FLTBa defines the steepness of the function (default 5)
    FLTBb defines the location of the centre point (default 1.0)
    FLTBc defines the initial amount of partitioning to leaves at DVS=0 (default 0.85)
    FLTBd defines the final amount of partitioning to leaves at DVS=2 (default 0)

    Note that allowing a fraction of partitioning to leaves at DVS=2 (e.g. setting FLTBd=A)
    also requires setting FOTBc=(1-A). Otherwise FOTB will override the partitioning to
    leaves.
    """

    def __init__(self, FLTBa=5, FLTBb=1.0, FLTBc=0.85, FLTBd=0.0):
        self.FLTBa = FLTBa
        self.FLTBb = FLTBb
        self.FLTBc = FLTBc
        self.FLTBd = FLTBd

    def __call__(self, x):
        FL = logistic_sigmoid(x, a=(self.FLTBc - self.FLTBd),
                              b=-self.FLTBa, c=self.FLTBb, d=self.FLTBd)
        if FL > 0.99:
            return 1.0
        elif FL < 0.01:
            return 0.0
        else:
            return FL


def compute_partition_functions(FOTBpar={}, FLTBpar={}, warn=True, stepsize=0.01):
    """Computes the set of partition tables FOTB, FLTB and FSTB based
    on the individual simmoid function parameters for FOTB and FLTB.

    FSTB is computed as the remainder: 1.0 - FOTB - FSTB

    Checks for negative partitioning are being applied. Note that the
    order of precedence is FOTB, FLTB, FSTB
    """
    fltb = FLTB(**FLTBpar)
    fotb = FOTB(**FOTBpar)
    sigm_DVS = np.arange(0, 2.1, stepsize)
    sigm_FOTB = np.array([fotb(d) for d in sigm_DVS])
    sigm_FLTB = np.array([fltb(d) for d in sigm_DVS])
    if any((sigm_FLTB + sigm_FOTB) > 1.0):
        sigm_FLTB = np.minimum(sigm_FLTB, 1.0 - sigm_FOTB)
        if warn:
            print("Warning: adjusting for negative partioning values")

    sigm_FSTB = 1. - sigm_FOTB - sigm_FLTB
    # Convert to X, Y pairs as array
    sigm_FLTB2 = np.array(list(zip(sigm_DVS, sigm_FLTB))).flatten()
    sigm_FSTB2 = np.array(list(zip(sigm_DVS, sigm_FSTB))).flatten()
    sigm_FOTB2 = np.array(list(zip(sigm_DVS, sigm_FOTB))).flatten()

    return sigm_FOTB2, sigm_FLTB2, sigm_FSTB2