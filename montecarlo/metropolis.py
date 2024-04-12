import numpy as np
import copy as cp
import random


def metropolis_montecarlo(ham, conf, T=1, nsweep=1000, nburn=100):
    """Run montecarlo with metropolis sampling for the Hamiltonian

    Parameters
    ----------
    ham: IsingHamiltonian, required 
    conf: BitString, required
        configuration - probably not needed 
    T: float, optional
        Temperature 

    Returns
    E_samples: np.array
        Energy samples collected
    M_samples: np.array
        Magnetization samples collected
    -------
    """
    E_samples = np.zeros(nsweep)
    M_samples = np.zeros(nsweep)

    # thermalization
    for si in range(nburn):
        ham.metropolis_sweep(conf, T=T)


    # accumulation
    ham.metropolis_sweep(conf, T=T)
    Ei = ham.energy(conf)
    Mi = conf.get_magnetization()
    M_samples[0] = Ei 
    E_samples[0] = Mi 
    for si in range(1, nsweep):
        # conf_proposed = cp.deepcopy(conf)
        # ham.metropolis_sweep(conf_proposed, T=T)
        # Eproposed = ham.energy(conf_proposed)

        # # prob_trans = 1.0 # probability of transitioning
        # delta_e = Eproposed - Ei
        # accept = True
        # if delta_e > 0.0:
        #     rand_comp = random.random()
        #     if rand_comp > np.exp(-delta_e / T):
        #         accept = False
        # if accept:
        #     conf = conf_proposed
        #     Ei = ham.energy(conf_proposed)
        #     Mi = conf.get_magnetization()
        # conf_proposed = cp.deepcopy(conf)
        # ham.metropolis_sweep(conf_proposed, T=T)
        # Eproposed = ham.energy(conf_proposed)
        
        ham.metropolis_sweep(conf, T=T)

        Ei = ham.energy(conf)
        Mi = conf.get_magnetization()

        # Collect 
        E_samples[si]  = Ei
        M_samples[si]  = Mi

    Eavg = np.mean(E_samples)
    Estd = np.std(E_samples) 
    print("E(avg): %12.8f ± %5.2e" %(Eavg, Estd/np.sqrt(nsweep)*3))
    
    Mavg = np.mean(M_samples)
    Mstd = np.std(M_samples) 
    print("M(avg): %12.8f ± %5.2e" %(Mavg, Mstd/np.sqrt(nsweep)*3))
    return E_samples, M_samples

def running_average(data):
    N = len(data)
    r_avg = np.zeros(N)

    if N == 0:
        return r_avg

    r_avg[0] = data[0]
    for i in range(1, N):
        r_avg[i] = (r_avg[i - 1] * (i) + data[i]) / (i + 1)
    
    return r_avg