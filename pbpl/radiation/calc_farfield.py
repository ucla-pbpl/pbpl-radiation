#!/usr/bin/env python
import sys, math, os, glob
import argparse
import toml
import time
import tqdm
from collections import deque
import numpy as np
from numpy import cross, dot
from numpy.linalg import norm
from scipy.integrate import ode
from scipy.integrate import simps, trapz
from scipy.interpolate import make_interp_spline
import scipy.signal as signal
from scipy.fftpack import fft, fftfreq
from numpy.random import randn
import h5py
import asteval
import pbpl
from pbpl.common.units import *
from mpi4py import MPI

def get_parser():
    parser = argparse.ArgumentParser(
        description='Calculate radiation')
    parser.add_argument(
        'conf', metavar='CONF',
        help='Configuration file (e.g., calc-radiation.toml)')
    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    args.conf = toml.load(args.conf)
    return args

def doit(q0, m0, t, r, p, energy_vals, thetax, thetay):
    pmag = norm(p, axis=1)
    energy = np.sqrt(pmag**2*c_light**2 + m0**2*c_light**4)
    gamma = energy/(m0*c_light**2)
    beta_mag = np.sqrt(1-1/gamma**2)
    beta = p*(beta_mag/pmag)[:,np.newaxis]
    betadot = make_interp_spline(t, beta, k=3).derivative(nu=1)(t)
    window = np.blackman(len(t))

    result = np.zeros(
        (len(energy_vals), len(thetax), len(thetay), 3), dtype=complex)
    for i, nx in enumerate(np.sin(thetax)):
        for j, ny in enumerate(np.sin(thetay)):
            nz = np.sqrt(1 - nx**2 - ny**2)
            n = np.array((nx, ny, nz))
            kappa = 1 - dot(beta, n)
            delta_t = t - dot(r, n)/c_light
            g_t = (1/kappa**2)[:,np.newaxis] * (
                cross(n, cross(n-beta, betadot)))
            for k, energy in enumerate(energy_vals):
                omega = energy / hbar
                phase = omega*delta_t
                h_t = g_t * np.exp(1j*phase)[:,np.newaxis]
                h_t *= window[:,np.newaxis]
                A_omega = trapz(h_t, t, axis=0)
                A_omega *= q0/np.sqrt(32*pi**3*eps0*c_light)
                result[k, i, j] = A_omega
    return result

def main():
    args = get_args()

    # create safe interpreter for evaluation of ranges
    aeval = asteval.Interpreter(use_numpy=True)
    for x in pbpl.common.units.__all__:
        aeval.symtable[x] = pbpl.common.units.__dict__[x]
    thetax, thetay, energy = [
        aeval(args.conf['Output'][x]) for x in [
            'thetax', 'thetay', 'energy']]
    A_omega = np.zeros(
        (len(energy), len(thetax), len(thetay), 3), dtype=complex)
    total_charge = 0.0
    fin = h5py.File(args.conf['Input']['Filename'], 'r')

    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    if rank == 0:
        worker_q = deque(list(range(1, size)))
        job_q = deque(list(fin.keys()))

        num_trajectories = len(job_q)
        fmt = ('{desc:>30s}:{percentage:3.0f}% ' +
               '|{bar}| {n_fmt:>9s}/{total_fmt:<9s}')
        bar = tqdm.tqdm(
            total=num_trajectories, bar_format=fmt, desc='calc_farfield')

        while len(job_q):
            if len(worker_q):
                # worker is available
                A = job_q.popleft()
                comm.send(A, dest=worker_q.popleft(), tag=True)
                bar.update(1)
            else:
                # wait for worker to become available
                worker_q.append(comm.recv())
        # notify workers that there are no more jobs
        for i in range(1, size):
            comm.send(None, dest=i, tag=False)

        # collect results from workers
        A_omega = comm.reduce(A_omega, op=MPI.SUM, root=0)
        total_charge = comm.reduce(total_charge, op=MPI.SUM, root=0)
        d2W = (2/hbar)*(A_omega*A_omega.conjugate()).real.sum(axis=3)

        with h5py.File(args.conf['Output']['Filename'], 'w') as fout:
            scale = 1.0
            if 'ScaledCharge' in args.conf['Input']:
                scaled_charge = args.conf['Input']['ScaledCharge'] * nC
                if scaled_charge != 0:
                    scale = scaled_charge/total_charge
            unit = (joule/(mrad**2*MeV))/scale
            fout['d2W'] = d2W/unit
            fout['thetax'] = thetax/mrad
            fout['thetay'] = thetay/mrad
            fout['energy'] = energy/MeV

    else:
        while 1:
            status = MPI.Status()
            dset_name = comm.recv(status=status)
            continue_working = status.Get_tag()
            if not continue_working:
                break
            g = fin[dset_name]
            q0 = g['q0'][()] * eplus
            m0 = g['m0'][()] * me
            trajectory = g['trajectory'][:]
            t = trajectory[:,0] * sec
            dr = trajectory[:,1:4] * meter
            dp = trajectory[:,4:] * m0 * c_light
            A_omega += doit(q0, m0, t, dr, dp, energy, thetax, thetay)
            total_charge += q0
            # tell boss that worker needs another job
            comm.send(rank, dest=0)

        # finalize
        comm.reduce(A_omega, op=MPI.SUM, root=0)
        comm.reduce(total_charge, op=MPI.SUM, root=0)

    fin.close()
