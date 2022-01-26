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



def canvas(with_attribution=True):
    """This is a conceptual class representation of a simple BLE device
    (GATT Server). It is essentially an extended combination of the
    :class:`bluepy.btle.Peripheral` and :class:`bluepy.btle.ScanEntry` classes

    :param client: A handle to the :class:`simpleble.SimpleBleClient` client
        object that detected the device
    :type client: class:`simpleble.SimpleBleClient`
    :param addr: Device MAC address, defaults to None
    :type addr: str, optional
    :param addrType: Device address type - one of ADDR_TYPE_PUBLIC or
        ADDR_TYPE_RANDOM, defaults to ADDR_TYPE_PUBLIC
    :type addrType: str, optional
    :param iface: Bluetooth interface number (0 = /dev/hci0) used for the
        connection, defaults to 0
    :type iface: int, optional
    :param data: A list of tuples (adtype, description, value) containing the
        AD type code, human-readable description and value for all available
        advertising data items, defaults to None
    :type data: list, optional
    :param rssi: Received Signal Strength Indication for the last received
        broadcast from the device. This is an integer value measured in dB,
        where 0 dB is the maximum (theoretical) signal strength, and more
        negative numbers indicate a weaker signal, defaults to 0
    :type rssi: int, optional
    :param connectable: `True` if the device supports connections, and `False`
        otherwise (typically used for advertising ‘beacons’).,
        defaults to `False`
    :type connectable: bool, optional
    :param updateCount: Integer count of the number of advertising packets
        received from the device so far, defaults to 0
    :type updateCount: int, optional
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote



if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())


