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
    def __init__(self, N=10, pbc=True):
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
        self.config = np.zeros(self.N, dtype=int) 
        randomlist = random.sample(range(0, self.N-1), M)
        for i in randomlist:
            self.config[i] = 1

        print(" Initialized config to: ", self.config)

    def __getitem__(self,i):
        return self.config[i]

    def flip_site(self, i):
        """
        flip spin at site, i 
        
        Parameters
        ----------
        i   : int
            site to flip 
        """
        if self.config[i] == 1:
            self.config[i] = 0
        else:
            self.config[i] = 1
        

class IsingHamiltonian1D:
    """
    Class for 1D Hamiltonian
        .. math::
            H = -\sum_{\left<ij\right>} \sigma_i\sigma_j - \mu\sum_i\sigma_i
    """
    def __init__(self, J, h, mu, pbc=True):
        self.J = J
        self.h = h
        self.mu = mu
        self.pbc = pbc

    def expectation_value(self, config):
        """
        Compute energy of configuration for the following Hamiltonian
        .. math::
            E = -\sum_{\left<ij\right>} \sigma_i\sigma_j - \mu\sum_i\sigma_i
        
        Parameters
        ----------
        config   : SpinConfig1D
            input configuration 
        
        Returns
        -------
        energy  : float
            Energy of the input configuration
        """
        e = 0.0
        for i in range(config.N-1):
            if config[i] == config[i+1]:
                e -= self.J
            else:
                e += self.J
        if self.pbc:
            if config[config.N-1] == config[0]:
                e -= self.J
            else:
                e += self.J
        
        for i in range(config.N):
            if config[i]:
                e -= self.mu * self.h[i]
            else:
                e += self.mu * self.h[i]
        return e



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


