import numpy as np
from numpy.testing import assert_almost_equal
import random
import copy as cp
        
        
class IsingHamiltonian1D:
    """Class for 1D Hamiltonian
        
        .. math::
            H = -J\\sum_{\\left<ij\\right>} \\sigma_i\\sigma_j + \\mu\\sum_i\\sigma_i

    """

    def __init__(self, J=1.0, mu=0.0, pbc=True):
        """ Constructor 
    
        Parameters
        ----------
        J: float, optional
            Strength of coupling
        mu: float, optional
            Chemical potential 
        pbc: bool, optional, default=true
            Do PBC?
        """
        self.J = J
        self.mu = mu
        self.pbc = pbc

    def energy(self, config):
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
       
        e += self.mu * np.sum(2*config.config-1)
        #e += self.mu * config.get_magnetization()
        
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
        
        
        del_e = 0.0
        
        # assume PBC
        iright = (i+1)%config.N
        ileft  = (i-1)%config.N
        if config.config[ileft] == config.config[iright]:
            if config.config[ileft] == config.config[i]:
                del_e = 4.0*self.J
            else:
                del_e = -4.0*self.J
                
        del_e += 2*self.mu * (2*config.config[i]-1)


        return config_trial, del_e
    
    def delta_e_for_flip_slow(self, i, config):
        """Compute the energy change incurred if one were to flip the spin at site i (slow)

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
        
        return config_trial, self.expectation_value(config_trial) - self.expectation_value(config)


        
    def metropolis_sweep(self, conf, T=1.0):
        """Perform a single sweep through all the sites and return updated configuration

        Parameters
        ----------
        conf   : :class:`SpinConfig1D`
            input configuration 
        T      : int
            Temperature
        
        Returns
        -------
        conf  : :class:`SpinConfig1D`
            Returns updated config
        """

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
                #print("nick: %12.8f %12.8f %12s" %(rand_comp, np.exp(-delta_e/T), accept))
            if accept:
                if conf.config[site_i] == 0:
                    conf.config[site_i] = 1
                else:
                    conf.config[site_i] = 0
            #print("%12s %12s deltaE = %12.8f prob = %12.8f" %(conf, accept, delta_e, np.exp(-delta_e/T)))
        return conf

    
    def compute_average_values(self, conf, T):
        """ Compute Average values exactly

        Parameters
        ----------
        conf   : :class:`SpinConfig1D`
            input configuration 
        T      : int
            Temperature
        
        Returns
        -------
        E  : float 
            Energy
        M  : float
            Magnetization
        HC : float
            Heat Capacity
        MS : float
            Magnetic Susceptability
        """
        E  = 0.0
        M  = 0.0
        Z  = 0.0
        EE = 0.0
        MM = 0.0

        for i in range(conf.n_dim):
            conf.set_int_config(i)
            Ei = self.energy(conf)
            Zi = np.exp(-Ei/T)
            E += Ei*Zi
            EE += Ei*Ei*Zi
            Mi = np.sum(2*conf.config-1)
            M += Mi*Zi
            MM += Mi*Mi*Zi
            Z += Zi
        
        E = E/Z
        M = M/Z
        EE = EE/Z
        MM = MM/Z
        
        HC = (EE - E*E)/(T*T)
        MS = (MM - M*M)/T
        return E, M, HC, MS
