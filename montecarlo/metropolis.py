import montecarlo
import random
import numpy as np
# from bitstring import BitStream, BitArray
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import copy as cp



def metropolis_montecarlo(ham, conf, T=1, nsweep=1000, nburn=100):
    E_samples = np.zeros(nsweep) 
    M_samples = np.zeros(nsweep) 
    EE_samples = np.zeros(nsweep) 
    MM_samples = np.zeros(nsweep) 
    
    # thermalization
    for si in range(nburn):
        ham.metropolis_sweep(conf, T=T)
    
    # accumulation
    ham.metropolis_sweep(conf, T=T)
    Ei = ham.energy(conf)
    Mi = conf.get_magnetization()
    M_samples[0]  = Mi 
    E_samples[0]  = Ei
    MM_samples[0]  = Mi*Mi 
    EE_samples[0]  = Ei*Ei
    for si in range(1,nsweep):
        ham.metropolis_sweep(conf, T=T)
        Ei = ham.energy(conf)
        Mi = conf.get_magnetization()
        
        # E_samples[si]  = Ei 
        # EE_samples[si]  = Ei*Ei
        E_samples[si]  = (E_samples[si-1]*(si) + Ei)/(si+1)
        EE_samples[si] = (EE_samples[si-1]*(si) + Ei*Ei)/(si+1)
        
        # M_samples[si]  = Mi 
        # MM_samples[si]  = Mi*Mi
        M_samples[si]  = (M_samples[si-1]*(si) + Mi)/(si+1)
        MM_samples[si] = (MM_samples[si-1]*(si) + Mi*Mi)/(si+1)
    

    #tmp1 = cp.deepcopy(E_samples)
    #tmp2 = cp.deepcopy(M_samples)
    #for si in range(1,nsweep):
    #    E_samples[si] = np.mean(tmp1[0:si])
    #    M_samples[si] = np.mean(tmp2[0:si])

    return E_samples, M_samples, EE_samples, MM_samples


# if __name__ == "__main__":
#     N=40
#     conf = montecarlo.BitString(N=N)
#     conf.initialize(M=20)
#     ham = montecarlo.IsingHamiltonian1D(J=-1.0, mu=0.1)
    
#     random.seed(3)
    
#     eavg = []
#     e_vs_T = []
#     m_vs_T = []
#     ee_vs_T = []
#     mm_vs_T = []
#     heat_cap_vs_T = []
#     magn_sus_vs_T = []
#     T_range = []
#     for Ti in range(1,50):
#         T = .1*Ti
#         e, m, ee, mm = metropolis_montecarlo(ham, cp.deepcopy(conf), T=T, nsweep=40000, nburn=2000)
#         T_range.append(T)
#         e_vs_T.append(e[-1])
#         m_vs_T.append(m[-1])
#         ee_vs_T.append(ee[-1])
#         mm_vs_T.append(mm[-1])
#         #print(e)
    
#         #plt.plot(e)
#         #plt.plot(m)
    
#         E  = e[-1]
#         EE = ee[-1]
#         M  = m[-1]
#         MM = mm[-1]
    
#         heat_cap = (EE-E*E)/(T*T)
#         magn_sus = (MM-M*M)/T
#         heat_cap_vs_T.append(heat_cap)
#         magn_sus_vs_T.append(magn_sus)
    
#         print("T= %12.8f E= %12.8f M=%12.8f Heat Capacity= %12.8f Mag. Suscept.=%12.8f" %(T, e[-1], m[-1], heat_cap, magn_sus))
#     plt.plot(T_range,e_vs_T, label="Energy")
#     plt.plot(T_range,m_vs_T, label="Magnetization")
#     plt.plot(T_range,magn_sus_vs_T, label="Susceptibility")
#     plt.plot(T_range,heat_cap_vs_T, label="Heat Capacity")
#     plt.legend()
    
#     #print(" %12.8f"%(np.dot(Eseries,Zseries)/sum(Zseries)))
#     plt.savefig('test.pdf')
    
