import numpy as np
import random

class SpinConfig:
    """
    Base class for different types of spin configurations (i.e., 1D, 2D, etc)
    """
    def __init__(self):
        print(" Why are you instantiating this?\n")



class SpinConfig1D(SpinConfig):
    """
    1-D spin configuration
    """
    def __init__(self,N=10,pbc=0):
        """
        Initialize instance

        Parameters
        ----------
        N   : Int, default: 10
            Number of sites
        pbc : bool, default: true
            Should we use periodic boundary conditions?

        Returns
        -------
    """    
    self.config = np.zeros(N, dtype=int)
    self.N = N
    self.pbc = pbc

    def initialize(self, M=0):
        """
        Initialize spin configuration with specified magnetization
        
        Parameters
        ----------
        M   : Int, default: 0
            Total number of spin up sites 
        """
        self.config = np.zeros(N, dtype=int) 
        randomlist = random.sample(range(0, self.N-1), M)
        for i in randomlist:
            config[i] = 1

        print(config)





class tmp:
    """
    test class
    """
    def __init__(self):
        pass
    def print(self,a):
        """
        print for tmp class
        """
        print(a)

"""
montecarlo.py
Introduction to the Monte Carlo method

Handles the primary functions
"""
def canvas2(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format)

    Replace this function and doc string for your own project

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote

def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format)

    Replace this function and doc string for your own project

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote

def testa():
    """
    This is just a test to see if I can do LaTeX

    .. math::
        \sum_{i=1}^{\\infty} x_{i}


    Parammeters
    -----------

    Returns
    -------
    """
    print(" testa\n")


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())


