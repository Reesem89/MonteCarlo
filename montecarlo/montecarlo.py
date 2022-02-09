import numpy as np
from numpy.testing import assert_almost_equal
import random
import copy as cp

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

    def __repr__(self):
        print(self.config)
        
    
    def __str__(self):
        return "".join(str(e) for e in self.config)
        
    def initialize(self, M=0, seed=2):
        """
        Initialize spin configuration with specified magnetization
        
        Parameters
        ----------
        M   : Int, default: 0
            Total number of spin up sites 
        """
        random.seed(seed)
        self.config = np.zeros(self.N, dtype=int) 
        randomlist = random.sample(range(0, self.N), M)
        for i in randomlist:
            self.config[i] = 1

        self.n_dim = 2**self.N
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
    
    def set_rand_config(self, seed=1):
        """
        set configuration to a random configuration 
        
        Parameters
        ----------
        seed : int
            seed for random numbers
            
        Returns
        -------
        """
        random.seed(seed)
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
        
        
        
class IsingHamiltonian1D:
    """Class for 1D Hamiltonian
        
        .. math::
            H = -J\\sum_{\\left<ij\\right>} \\sigma_i\\sigma_j - \\mu\\sigma_i

    """

    def __init__(self, J, mu, pbc=True):
        """ Constructor 
    
        Parameters
        ----------
        J: float, required
            Strength of coupling
        mu: float, required
            Chemical potential 
        pbc: bool, optional, default=true
            Do PBC?
        """
        self.J = J
        self.mu = mu
        self.pbc = pbc

    def expectation_value(self, config):
        """Compute energy of configuration, `config` 
            
            .. math::
                E = \\left<\\hat{H}\\right> 

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
       
        e -= self.mu * config.get_magnetization()
        
        return e

    def delta_e_for_flip(self, i, config):
        """Compute the energy change incurred if one were to flip the spin at site i

        Parameters
        ----------
        i        : int
            Index of site to flip
        config   : :class:`SpinConfig1D`
            input configuration 
        
        Returns
        -------
        energy  : list[SpinConfig1D, float]
            Returns both the flipped config and the energy change
        """
        config_trial = cp.deepcopy(config) 
        config_trial.flip_site(i)
        
#         del_e = 0.0
        
#         # assume PBC
#         iright = (i+1)%config.N
#         ileft  = (i-1)%config.N
#         if config.config[ileft] == config.config[iright]:
#             if config.config[ileft] == config.config[i]:
#                 del_e = 4.0*self.J
#             else:
#                 del_e = -4.0*self.J
                
#         del_e += 2*self.mu * (2*config.config[i]-1)

        # print("check")
        # print(del_e)
        # print(self.expectation_value(config_trial) - self.expectation_value(config))
#         assert_almost_equal(del_e, self.expectation_value(config_trial) - self.expectation_value(config)) # make this a test
        
        return config_trial, self.expectation_value(config_trial) - self.expectation_value(config)
        # return config_trial, del_e


        
    def metropolis_sweep(self,conf, T=1.0, seed=1):
        """Perform a single sweep through all the sites and return updated configuration

        Parameters
        ----------
        conf   : :class:`SpinConfig1D`
            input configuration 
        T      : int
            Temperature
        seed   : int
            seed for random numbers
        
        Returns
        -------
        conf  : :class:`SpinConfig1D`
            Returns updated config
        """

        random.seed(seed)
        
        for site_i in range(conf.N):
        
            # new_conf, delta_e = ham.delta_e_for_flip(site_i, conf)      
            delta_e = 0.0
        
            # assume PBC
            iright = (site_i+1)%conf.N
            ileft  = (site_i-1)%conf.N
            if conf.config[ileft] == conf.config[iright]:
                if conf.config[ileft] == conf.config[site_i]:
                    delta_e = 4.0*self.J
                else:
                    delta_e = -4.0*self.J
                
            delta_e += 2*self.mu * (2*conf.config[site_i]-1)

            # prob_trans = 1.0 # probability of transitioning
            accept = True
            if delta_e > 0.0:
                # prob_trans = np.exp(-delta_e/T)
                rand_comp = random.random()
                if rand_comp > np.exp(-delta_e/T):
                    accept = False
                # print("nick: %12.8f %12.8f %12s" %(rand_comp, np.exp(-delta_e/T), accept))
            if accept:
                if conf.config[site_i] == 0:
                    conf.config[site_i] = 1
                else:
                    conf.config[site_i] = 0
            # print("%12s %12s %12.8f %12.8f" %(conf, accept, delta_e, np.exp(-delta_e/T)))
        return conf



if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())


