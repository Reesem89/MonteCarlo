from numba import jit

import numpy as np
import random
import copy as cp
import montecarlo


@jit(nopython=True)
def delta_e_for_flip_fast(i, config, nodes, J, mu):
    """
    Parameters
    ----------
    i        : int
        Index of site to flip
    config   : list
        list of 0's and 1's
    nodes    : [[]]]
        for each node, list of nodes connected
    J        : [[]]
        for each node, list of J couplings to connected nodes
    mu       : list
        for each node, strength of local field, mu

    Returns
    -------
    del_e  : float
        Returns the energy change
    """

    del_e = 0.0
    del_si = 2
    if config[i] == 1:
        del_si = -2

    for jidx, j in enumerate(nodes):
        del_e += (2.0 * config[j] - 1.0) * J[jidx] * del_si

    del_e += mu[i] * del_si
    return del_e


class IsingHamiltonian:
    """Class for an Ising Hamiltonian of arbitrary dimensionality

    .. math::
        H = \\sum_{\\left<ij\\right>} J_{ij}\\sigma_i\\sigma_j + \\sum_i\\mu_i\\sigma_i

    """

    def __init__(self, J=[[()]], mu=np.zeros(1)):
        """Constructor

        Parameters
        ----------
        J: list of lists of tuples, optional
            Strength of coupling, e.g,
            [(4, -1.1), (6, -.1)]
            [(5, -1.1), (7, -.1)]
        mu: vector, optional
            local fields
        """
        self.J = J
        self.mu = mu

        self.nodes = []
        self.js = []

        for i in range(len(self.J)):
            self.nodes.append(np.zeros(len(self.J[i]), dtype=int))
            self.js.append(np.zeros(len(self.J[i])))
            for jidx, j in enumerate(self.J[i]):
                self.nodes[i][jidx] = j[0]
                self.js[i][jidx] = j[1]
        self.mu = np.array([i for i in self.mu])
        self.N = len(self.J)

    def energy(self, config):
        """Compute energy of configuration, `config`

            .. math::
                E = \\left<\\hat{H}\\right>

        Parameters
        ----------
        config   : BitString
            input configuration

        Returns
        -------
        energy  : float
            Energy of the input configuration
        """
        if len(config.config) != len(self.J):
            error("wrong dimension")

        e = 0.0
        for i in range(config.N):
            # print()
            # print(i)
            for j in self.J[i]:
                if j[0] < i:
                    continue
                # print(j)
                if config[i] == config[j[0]]:
                    e += j[1]
                else:
                    e -= j[1]

        e += np.dot(self.mu, 2 * config.config - 1)

        return e

    def delta_e_for_flip(self, i, config):
        """Compute the energy change incurred if one were to flip the spin at site i

        Parameters
        ----------
        i        : int
            Index of site to flip
        config   : :class:`BitString`
            input configuration

        Returns
        -------
        energy  : list[BitString, float]
            Returns both the flipped config and the energy change
        """

        del_e = 0.0
        del_si = 2
        if config.config[i] == 1:
            del_si = -2

        for j in self.J[i]:
            # if config.config[i] == config.config[j[0]]:
            del_e += (2.0 * config.config[j[0]] - 1.0) * j[1] * del_si

        del_e += self.mu[i] * del_si
        return del_e

    def delta_e_for_flip_slow(self, i, config):
        """Compute the energy change incurred if one were to flip the spin at site i (slow)

        Parameters
        ----------
        i        : int
            Index of site to flip
        config   : :class:`BitString`
            input configuration

        Returns
        -------
        energy  : list[BitString, float]
            Returns both the flipped config and the energy change
        """
        config_trial = cp.deepcopy(config)
        config_trial.flip_site(i)

        return self.energy(config_trial) - self.energy(config)

    def metropolis_sweep(self, conf, T=1.0):
        """Perform a single sweep through all the sites and return updated configuration

        Parameters
        ----------
        conf   : :class:`BitString`
            input configuration
        T      : int
            Temperature

        Returns
        -------
        conf  : :class:`BitString`
            Returns updated config
        """

        # print(self.mu)
        # error('here')
        for site_i in range(conf.N):

            # print(type(self.nodes[site_i]))
            # print(type(self.js[site_i]))
            # print(type(self.mu))
            # error('here')

            delta_e = delta_e_for_flip_fast(
                site_i, conf.config, self.nodes[site_i], self.js[site_i], self.mu
            )
            # delta_e = self.delta_e_for_flip(site_i, conf)

            # prob_trans = 1.0 # probability of transitioning
            accept = True
            if delta_e > 0.0:
                # prob_trans = np.exp(-delta_e/T)
                rand_comp = random.random()
                if rand_comp > np.exp(-delta_e / T):
                    accept = False
                # print("nick: %12.8f %12.8f %12s" %(rand_comp, np.exp(-delta_e/T), accept))
            if accept:
                if conf.config[site_i] == 0:
                    conf.config[site_i] = 1
                else:
                    conf.config[site_i] = 0
            # print("%12s %12s deltaE = %12.8f prob = %12.8f" %(conf, accept, delta_e, np.exp(-delta_e/T)))
        return conf

    def compute_average_values(self, T):
        """Compute Average values exactly

        Parameters
        ----------
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
        E = 0.0
        M = 0.0
        Z = 0.0
        EE = 0.0
        MM = 0.0

        conf = montecarlo.BitString(self.N)

        for i in range(conf.n_dim):
            conf.set_int_config(i)
            Ei = self.energy(conf)
            Zi = np.exp(-Ei / T)
            E += Ei * Zi
            EE += Ei * Ei * Zi
            Mi = np.sum(2 * conf.config - 1)
            M += Mi * Zi
            MM += Mi * Mi * Zi
            Z += Zi

        E = E / Z
        M = M / Z
        EE = EE / Z
        MM = MM / Z

        HC = (EE - E * E) / (T * T)
        MS = (MM - M * M) / T
        return E, M, HC, MS


# if __name__ == "__main__":
#     N=10
#     conf = montecarlo.BitString(N=N)
#     random.seed(3)
#     conf.initialize(M=4)
#     print(conf)

#     Jval = 1.0
#     muval = 0.2

#     J = []
#     mu = []
#     for i in range(N):
#         J.append([((i+1) % N, Jval), ((i-1) % N, Jval)])

#     for i in range(N):
#         mu.append(muval)

#     for i in J:
#         print(i)

#     ham = IsingHamiltonianND(J = J, mu = mu)
#     ham1d = montecarlo.IsingHamiltonian1D(J = -Jval, mu = muval)

#     print(ham.energy(conf))
#     print(ham1d.energy(conf))
#     print()
#     tmp = cp.deepcopy(conf)
#     tmp.flip_site(0)
#     print(tmp)
#     print(ham.energy(tmp))
#     print(ham1d.energy(tmp))
#     print()
#     print(ham.energy(tmp) - ham.energy(conf))
#     print(ham1d.energy(tmp) - ham1d.energy(conf))
#     print(ham.delta_e_for_flip(0,conf)[1])
#     #print(ham1d.delta_e_for_flip(0,conf)[1])
#     assert(abs(ham.energy(conf) - ham1d.energy(conf)) < 1e-9)
#     assert(abs(ham.delta_e_for_flip_slow(0,conf)[1] - ham.delta_e_for_flip(0,conf)[1]) < 1e-9)

#     print(ham.compute_average_values(conf,1.0))
#     print(ham1d.compute_average_values(conf,1.0))
