# -*- coding: utf-8 -*-
import sys
import os
import itertools
import argparse
import toml
import asteval
from collections import namedtuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import numpy as np
import h5py
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import simps, trapz
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import pbpl
import pbpl.common
from pbpl.common.units import *

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Plot farfield radiation',
        epilog='''\
Example:

.. code-block:: sh

  pbpl-radiation-plot-farfield plot-farfield.toml
''')
    parser.add_argument(
        'config_filename', metavar='conf-file',
        help='Configuration file')
    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    args.conf = toml.load(args.config_filename)
    return args

Axis = namedtuple('Axis', 'val label unit xlim')

def get_axis(aeval, val, label, unit, xlim):
    xlim = aeval(xlim)
    if xlim is not None:
        xlim = np.array(xlim)
    return Axis(val, label, aeval(unit), xlim)

def plot_annotation(ax, aeval, conf):
    if 'Annotation' in conf:
        for aconf in conf['Annotation']:
            text = ''
            for s in aconf['Text']:
                text += aeval(s) + '\n'
            ax.text(
                *aconf['Location'], text, va='top',
                size=6.0, transform=ax.transAxes)

def plot_1D(output, aeval, conf, d2W, energy, thetax, thetay):
    axes = [get_axis(aeval, *conf[x]) for x in ['XAxis', 'YAxis']]

    fig = plot.figure(figsize=np.array(conf['FigSize'])/72)
    ax = fig.add_subplot(1, 1, 1)

    plot.xlabel(axes[0].label, labelpad=0.0)
    plot.ylabel(axes[1].label, labelpad=1.0)

    if axes[0].val == 'energy':
        density = simps(simps(d2W, thetax, axis=1), thetay, axis=1)
        xvals = energy
    elif axes[0].val == 'thetax':
        density = simps(simps(d2W, thetay, axis=2), energy, axis=0)
        xvals = thetax
    elif axes[0].val == 'thetay':
        density = simps(simps(d2W, thetax, axis=1), energy, axis=0)
        xvals = thetay
    else:
        raise Exception("unknown integration axis '{}'".format(axes[0].val))

    density_integral = simps(density, xvals)
    aeval.symtable['density_integral'] = density_integral

    if axes[1].val == 'energy_spectral_density':
        pass
    elif axes[1].val == 'photon_spectral_density':
        # this only makes sense if the x-axis is 'photon energy'
        assert(axes[0].val == 'energy')
        mask = (energy != 0)
        density[mask] = density[mask]/energy[mask]
    else:
        raise Exception("unknown axis '{}'".format(axes[1].val))

    ax.plot(xvals/axes[0].unit, density/axes[1].unit, linewidth=0.6)

    plot_annotation(ax, aeval, conf)

    ax.set_xlim(*axes[0].xlim)
    ax.set_ylim(*axes[1].xlim)

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    output.savefig(fig, transparent=True)


def plot_2D(output, aeval, conf, d2W, energy, thetax, thetay):
    axes = [get_axis(aeval, *conf[x]) for x in ['XAxis', 'YAxis', 'ZAxis']]
    num_contours = conf['NumContours']

    plot.rc('figure.subplot', right=0.95, top=0.97, bottom=0.15, left=0.15)
    fig = plot.figure(figsize=np.array(conf['FigSize'])/72)
    ax = fig.add_subplot(1, 1, 1)

    if 'AspectRatio' in conf:
        ax.set_aspect(conf['AspectRatio'])

    plot.xlabel(axes[0].label, labelpad=0.0)
    plot.ylabel(axes[1].label, labelpad=1.0)

    mapping = { 'energy' : energy, 'thetax' : thetax, 'thetay' : thetay }
    x, y, z = [mapping[x.val] for x in axes]

    if 'HarmonicCut' in conf:
        K, gamma0, lambda_u, cut = [
            aeval(str(conf['HarmonicCut'][x])) for x in [
                'K', 'gamma0', 'lambda_u', 'cut']]
        m0 = me
        Thetax, Thetay, Cut = np.meshgrid(thetax, thetay, cut)
        lambda_cut = (lambda_u/(Cut*2*gamma0**2)) * (
            1 + 0.5*K**2 + gamma0**2*(Thetax**2 + Thetay**2))
        energy_cut = planck*c_light/lambda_cut
        index_cut = np.searchsorted(energy, energy_cut)

        density = np.zeros((len(thetax), len(thetay)))
        for i in range(len(thetax)):
            for j in range(len(thetay)):
                i0 = slice(index_cut[i,j,0], index_cut[i,j,1])
                if i0.start == i0.stop:
                    density[i, j] = 0.0
                else:
                    density[i, j] = simps(d2W[i0, i, j], energy[i0])
    else:
        if axes[2].xlim is None:
            i0 = slice(None)
        else:
            integration_range = axes[2].xlim * axes[2].unit
            i0 = slice(*np.searchsorted(mapping[axes[2].val], integration_range))

        if axes[2].val == 'energy':
            density = simps(d2W[i0,:,:], energy[i0], axis=0)
        elif axes[2].val == 'thetax':
            density = simps(d2W[:,i0,:], thetax[i0], axis=1)
        elif axes[2].val == 'thetay':
            density = simps(d2W[:,:,i0], thetay[i0], axis=2)

    xgrid = np.linspace(*axes[0].xlim, conf['InterpolationGrid'][0])
    ygrid = np.linspace(*axes[1].xlim, conf['InterpolationGrid'][1])
    post_shape = np.array(list(itertools.product(xgrid, ygrid)))

    points = np.array(list(itertools.product(x/axes[0].unit, y/axes[1].unit)))
    outpoints = np.array(
        list(itertools.product(xgrid, ygrid)))

    smoothed_density = griddata(
        points, density.flatten(), post_shape, method='cubic').reshape(
            *conf['InterpolationGrid'])
    if 'Smoothing' in conf:
        smoothed_density = gaussian_filter(
            smoothed_density, sigma=conf['Smoothing'])

    epsilon = 1e-8
    mask = (smoothed_density < smoothed_density.max()*epsilon)
    smoothed_density[mask] = smoothed_density.max()*epsilon

    smoothed_density *= eval(conf['CBar'][0])
    density_unit = eval(conf['CBar'][2])

    contours = ax.contourf(
        xgrid, ygrid, smoothed_density.T/density_unit,
        levels=num_contours, cmap=pbpl.common.blue_cmap, vmin=0)
    ax.contour(
        xgrid, ygrid, smoothed_density.T/density_unit,
        levels=num_contours, colors='k', linewidths=0.4, vmin=0)

    cbar = plot.colorbar(contours)
    cbar.ax.set_ylabel(conf['CBar'][1])

    plot_annotation(ax, aeval, conf)

    if 'HarmonicLines' in conf:
        assert('energy' in [axes[0].val, axes[1].val])
        K, gamma0, lambda_u, plot_args, harmonics = [
            aeval(str(conf['HarmonicLines'][x])) for x in [
                'K', 'gamma0', 'lambda_u', 'plot_args', 'harmonics']]
        if axes[0].val == 'energy':
            theta = y
        else:
            theta = x
        m0 = me
        for h in harmonics:
            lambda_h = (lambda_u/(h*2*gamma0**2)) * (
                1 + 0.5*K**2 + gamma0**2*theta**2)
            energy_h = planck*c_light/lambda_h
            if axes[0].val == 'energy':
                ax.plot(
                    energy_h/axes[0].unit, theta/axes[1].unit, **plot_args)
            else:
                ax.plot(
                    theta/axes[0].unit, energy_h/axes[1].unit, **plot_args)

    ax.set_xlim(*axes[0].xlim)
    ax.set_ylim(*axes[1].xlim)

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    output.savefig(fig, transparent=True)

def main():
    args = get_args()
    conf = args.conf

    pbpl.common.setup_plot()
    with h5py.File(conf['Files']['Input'], 'r') as fin:
        d2W = fin['d2W'][:]*joule/(mrad**2*MeV)
        photon_energy = fin['energy'][:]*MeV
        thetax = fin['thetax'][:]*mrad
        thetay = fin['thetay'][:]*mrad

    # create safe interpreter for evaluation of configuration expressions
    aeval = asteval.Interpreter(use_numpy=True)
    for x in pbpl.common.units.__all__:
        aeval.symtable[x] = pbpl.common.units.__dict__[x]

    output = PdfPages(conf['Files']['Output'])
    for fig_conf in conf['Figure']:
        fig_type = fig_conf['Type']
        if fig_type == '1D':
            func = plot_1D
        elif fig_type == '2D':
            func = plot_2D
        else:
            raise Exception("unknown figure type '{}'".format(fig_type))
        func(output, aeval, fig_conf, d2W, photon_energy, thetax, thetay)

    output.close()
