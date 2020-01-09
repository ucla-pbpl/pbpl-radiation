#!/usr/bin/env python
import sys, math, os, glob
import argparse
import toml
import numpy as np
from scipy.integrate import ode
from numpy.random import randn
import h5py
from pbpl.common.units import *
from mpi4py import MPI

def get_parser():
    parser = argparse.ArgumentParser(
        description='Calculate trajectories')
    parser.add_argument(
        'conf', metavar='CONF',
        help='Configuration file (e.g., calc-trajectories.toml)')
    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    args.conf = toml.load(args.conf)
    return args

def get_beam(conf):
    q0 = conf['q0'] * eplus
    m0 = conf['m0'] * me
    gamma0 = conf['gamma0']
    beta0 = np.sqrt(1-1/gamma0**2)
    p0 = gamma0 * beta0 * m0 * c_light
    sigma_x = np.array(
        (conf['sigma_x'], conf['sigma_y'], conf['sigma_z']))*meter
    eps_x = conf['eps_x'] / (beta0 * gamma0)
    eps_y = conf['eps_y'] / (beta0 * gamma0)
    dp_p0 = np.array((eps_x/sigma_x[0], eps_y/sigma_x[1], conf['dp_p0']))
    sigma_p = p0 * dp_p0
    def result():
        x = randn(3) * sigma_x
        p = randn(3) * sigma_p
        return (q0, m0, gamma0, np.concatenate((x, p)))
    return result

def get_undulator(conf):
    lambda_u = conf['Wavelength'] * meter
    ku = twopi/lambda_u
    K = conf['K']
    B0 = K*ku*me*c_light/eplus
    def B(r):
        y = r[1]
        z = r[2]
        z += 0.5*pi/ku
        return B0 * np.array((
            np.zeros_like(y),
            np.sin(ku*z) * np.cosh(ku*y),
            np.cos(ku*z) * np.sinh(ku*y)))
    return (None, B)

def calc_trajectory(times, beam, E, B):
    q0, m0, gamma0, y0 = beam
    beta0 = np.sqrt(1-1/gamma0**2)
    v0 = beta0 * c_light
    p0 = gamma0 * m0 * v0

    def yprime(t, y):
        x = y[:3]   # x, y, zeta
        p = y[3:]   # px, py, deltap
        xlab = x + np.array((0,0,v0*t))   # x, y, z
        plab = p + np.array((0,0,p0))     # px, py, pz
        p2 = np.dot(plab, plab)
        energy = np.sqrt(m0**2*c_light**4 + p2*c_light**2)
        dxlab = plab*c_light**2/energy
        dx = p*c_light**2/energy
        dx[2] += (p0*c_light**2/energy) - v0
        dp = np.zeros(3)
        if len(E)>0:
            E_xlab = np.array([f(xlab) for f in E]).sum(axis=0)
            dp += q0 * E_xlab
        if len(B)>0:
            B_xlab = np.array([f(xlab) for f in B]).sum(axis=0)
            dp += q0 * np.cross(dxlab, B_xlab)
        return np.concatenate((dx, dp))

    solver = ode(yprime)
    solver.set_integrator('dopri5', max_step=1e-4)
    solver.set_initial_value(y0)
    trajectory = []
    for t in times:
        trajectory.append(np.concatenate(([t], solver.integrate(t))))
    trajectory = np.array(trajectory)
    t = trajectory[:,0]
    dr = trajectory[:,1:4]
    dp = trajectory[:,4:]
    r = (v0*t)[:, np.newaxis] * np.array((0, 0, 1)) + dr
    p = np.array((0,0,p0)) + dp
    trajectory[:,1:4] = r
    trajectory[:,4:] = p
    return q0, m0, gamma0, trajectory

def run(conf, beam, fields):
    filename = conf['Filename']
    num_trajectories = conf['NumTrajectories']
    duration = conf['Duration'] * sec
    dt = conf['dt'] * sec

    E = [f[0] for f in fields if f[0]]
    B = [f[1] for f in fields if f[1]]

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    times = np.arange(0.0, duration, dt)
    with h5py.File(filename, 'w', driver='mpio', comm=MPI.COMM_WORLD) as fout:
        # HDF5 parallel model : all structural API calls are collective
        for i in range(num_trajectories):
            g = fout.create_group(str(i))
            g.create_dataset('trajectory', (len(times), 7), dtype=np.float64)
            for dset in ['q0', 'm0', 'gamma0']:
                g.create_dataset(dset, (), dtype=np.float64)

        for i in range(num_trajectories):
            if i % size == rank:
                q0, m0, gamma0, trajectory = calc_trajectory(
                    times, beam(), E, B)
                units = np.array((sec, *[meter]*3, *[me*c_light]*3))
                g = fout[str(i)]
                g['trajectory'][:] = (trajectory/units).astype(np.float64)
                g['q0'][()] = q0/eplus
                g['m0'][()] = m0/me
                g['gamma0'][()] = gamma0

def main():
    args = get_args()
    beam = get_beam(args.conf['Beam'])
    field_types = {}
    field_types['Undulator'] = get_undulator
    fields = [field_types[x['Type']](x) for x in args.conf['Fields']]
    run(args.conf['Output'], beam, fields)


if __name__ == '__main__':
    sys.exit(main())
