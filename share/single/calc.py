#!/usr/bin/env python
import numpy as np
from pbpl.common.units import *

m0 = me
gamma0 = 20000.0
beta0 = np.sqrt(1-1/gamma0**2)
lambda_u = 10*mm
Tu = lambda_u/(c_light*beta0)
dt = Tu/20
duration = 10*Tu
ku = twopi/lambda_u
K = 1.0
B = K * ku * m0 * c_light / eplus
rho = m0 * gamma0 * c_light / (eplus * B)
lambda0 = (lambda_u/(2*gamma0**2))*(1 + 0.5*K**2)
nu0 = c_light/lambda0
T0 = 1/nu0
E0 = planck*c_light/lambda0
print('gamma0 = {}'.format(gamma0))
print('1/gamma0 = {} mrad'.format((1/gamma0)/mrad))
print('K = {}'.format(K))
print('lambda_u = {} mm'.format(lambda_u/mm))
print('Tu = {} ns'.format(Tu/ns))
print('dt = {} s'.format(dt/sec))
print('duration = {} s'.format(duration/sec))
print('B = {} T'.format(B/tesla))
print('rho = {} m'.format(rho/meter))
print('E0 = {} eV'.format(E0/eV))
print('T0 = {} ns'.format(T0/ns))
