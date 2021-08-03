import montecarlo
import random
import numpy as np
from bitstring import BitStream, BitArray
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import copy as cp

random.seed(2)
N = 100
conf = montecarlo.SpinConfig1D(N=N)
conf.initialize(M=5)
ham = montecarlo.IsingHamiltonian1D(1.0, [.1 for i in range(N)], 1.01)

nS = 100000
Z = 0.0
E = 0.0
T = 2.1
Eavg = np.zeros(nS)
Eseries = np.zeros(nS)
Zseries = np.zeros(nS)
nburn = 1000

Ei = ham.expectation_value(conf)

# avg(E) = sum_i E_i P_i

site_i = 0
print(" Start with regular Monte Carlo\n")
beta = 1.0/T
for si in range(nS):
    site_i = random.randint(0,conf.N-1)
    
    new_conf, delta_e = ham.delta_e_for_flip(site_i, conf)

    prob_trans = 1.0
    if delta_e >= 0:
        prob_trans = np.exp(-delta_e*beta)


    accept = random.choices([True,False],[prob_trans, 1.0-prob_trans])[0]
    
    #print(conf.config, "%12.8f"%Ei, "%12.8f"%delta_e, "%12.8f"%prob_trans, accept)
   
    if accept:
        conf = cp.deepcopy(new_conf)
        #print(conf.config, "%12.8f"%delta_e, site_i, "%12.8f"%prob_trans, accept)
        Ei = ham.expectation_value(conf)

    #Zi = np.exp(-Ei*beta)
    #Z += Zi
    #E += Ei
    #E += Ei*Zi/Z
    Eseries[si] = Ei
    if si > nburn:
        Eavg[si] = np.mean(Eseries[nburn:si]) 
    #Zseries[si] = Zi

plt.plot(Eseries)
plt.plot(Eavg)
#plt.plot(Eseries/sum(Zseries))
print(" %12.8f"%np.mean(Eseries))
#print(" %12.8f"%(np.dot(Eseries,Zseries)/sum(Zseries)))
plt.savefig('test.pdf')

