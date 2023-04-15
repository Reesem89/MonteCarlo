import numpy as np
from numpy.testing import assert_almost_equal
import random
import copy as cp


class BitString:
    """
    Bit string for encoding a spin configuration
    """
    def __init__(self, N=10, pbc=True):
        """
        Initialize instance

        Parameters
        ----------
        N   : int, default: 10
            Number of sites
        pbc : bool, default: true
            Should we use periodic boundary conditions?

        Returns
        -------
        """    
        self.config = np.zeros(N, dtype=int)
        self.N = N
        self.pbc = pbc
        self.n_dim = 2**self.N

    def __repr__(self):
        print(self.config)
        
    
    def __str__(self):
        return "".join(str(e) for e in self.config)
        
    def initialize(self, M=0, verbose=0):
        """
        Initialize spin configuration with specified magnetization
        
        Parameters
        ----------
        M   : Int, default: 0
            Total number of spin up sites 
        """
        self.config = np.zeros(self.N, dtype=int) 
        randomlist = random.sample(range(0, self.N), M)
        for i in randomlist:
            self.config[i] = 1


    def __getitem__(self,i):
        return self.config[i]

    def flip_site(self, i):
        """
        flip spin at site, i 
        
        Parameters
        ----------
        i   : int
            site to flip 
            
        Returns
        -------
        """
        if self.config[i] == 1:
            self.config[i] = 0
        else:
            self.config[i] = 1

    def get_rand_config(self):
        """
        get random configuration 
        
        Parameters
        ----------
        
        Returns
        -------
        config : list[int]
            random bitstring
        """
        return np.array([int(i) for i in np.binary_repr(random.randrange(0,self.n_dim), width=self.N)])
    
    def set_rand_config(self):
        """
        set configuration to a random configuration 
        
        Parameters
        ----------
            
        Returns
        -------
        """
        self.config = np.array([int(i) for i in np.binary_repr(random.randrange(0,self.n_dim), width=self.N)])   
    
    def set_int_config(self, int_index):
        """
        set configuration to bitstring of `int`
        
        Parameters
        ----------
        int_index : int, required
            integer whose bit representation corresponds to desired spin configuration
        
        Returns
        -------
        """
        self.config = np.array([int(i) for i in np.binary_repr(int_index, width=self.N)])
        
    def get_magnetization(self):
        """
        Return net magnetization of current configuration 
        
        Parameters
        ----------
        
        Returns
        -------
        m : float
            magnetization
        """
        return np.sum(2*self.config-1)
    
    def set_config(self, conf):
        """
        set configuration that specified from `conf`
        
        Parameters
        ----------
        conf : [int], required
            list of 1/0 specifying desired spin configuration
        
        Returns
        -------
        """
        assert(len(conf) == self.N)
        self.config = np.array(conf)
        
    def x_gate(self, i):
        self.flip_site(i)

    def and_gate(self, conf):
        out = cp.deepcopy(conf)
        for i in range(len(self.config)):
            out.config[i] = int(self.config[i] == 1 & conf.config[i] == 1)
        return out

