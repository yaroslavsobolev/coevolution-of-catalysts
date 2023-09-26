from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from substrate_colors import *


def simpleaxis(ax=None):
    '''removes the top and right boundaries from pyplot plots for aesthetic reasons'''
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def potential(x, widthparameter=0.2):
    return 20 * (1 - np.exp(-0.2 * np.sqrt(widthparameter * (x ** 2))))


def P(x, T, factor=2.5, preterm=8):
    if preterm > 0:
        prefactor = preterm + np.log(T)
    else:
        prefactor = 1
    if factor==2.5:
        return prefactor * np.exp(-(potential(3 + x - factor*np.log(T))) / T)
    elif factor==0:
        return prefactor * np.exp(-(potential(x - 4.92 - 0.7 * np.log(T), widthparameter=0.01)) / T)


if __name__ == '__main__':
    # num=1
    # w, h = (5, 5)
    # fig = plt.figure(frameon=False)
    # fig.set_size_inches(w, h)
    # ax0 = plt.Axes(fig, [0., 0., 1., 1.])
    # ax0.set_axis_off()
    # ax0.set_yscale('log')
    # fig.add_axes(ax0)
    # plt.xlabel('Configuration space of all ligands')
    # plt.ylabel('Configuration space of all substrates', labelpad=15)
    #
    # x = np.linspace(-10, 10, 1000)
    #
    # y = np.logspace(-0.6, 1.4, 1000)
    # X, Y = np.meshgrid(x, y)
    # Z = P(X, Y, preterm = 15)
    # # Z[Z<1e-1] = np.nan
    # c = ax0.pcolormesh(X, Y, Z, cmap='viridis', vmin=0, vmax=11+7+5)
    # fig.savefig(f'figures/max_{num}.png', dpi=300)
    # plt.show()
    #
    #
    # w, h = (5, 5)
    # fig = plt.figure(frameon=False)
    # fig.set_size_inches(w, h)
    # ax0 = plt.Axes(fig, [0., 0., 1., 1.])
    # ax0.set_axis_off()
    # ax0.set_yscale('log')
    # fig.add_axes(ax0)
    # plt.xlabel('Configuration space of all ligands')
    # plt.ylabel('Configuration space of all substrates', labelpad=15)
    #
    # x = np.linspace(-10, 10, 1000)
    #
    # y = np.logspace(-0.6, 1.4, 1000)
    # X, Y = np.meshgrid(x, y)
    # Z = P(X, Y)
    # thresh = 2
    # Z[Z>thresh]=100
    # Z[Z<=thresh]=0
    # c = ax0.pcolormesh(X, Y, Z, cmap='Greys_r', vmin=0)
    # fig.savefig(f'figures/max_{num}_mask.png', dpi=300)
    # plt.show()

    num = 2
    thresh = 2
    w, h = (5, 5)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w, h)
    ax0 = plt.Axes(fig, [0., 0., 1., 1.])
    ax0.set_axis_off()
    ax0.set_yscale('log')
    fig.add_axes(ax0)
    plt.xlabel('Configuration space of all ligands')
    plt.ylabel('Configuration space of all substrates', labelpad=15)

    x = np.linspace(-10, 10, 1000)

    y = np.logspace(-0.6, 1.4, 1000)
    X, Y = np.meshgrid(x, y)
    Z = P(X, Y, factor=0, preterm=20)
    # Z[Z<1e-1] = np.nan
    c = ax0.pcolormesh(X, Y, Z, cmap='viridis', vmin=0)

    substrate_locations = [0.3, 1, 5, 21]

    bordercolor = 'C1'
    x0 = 6.86
    npoints = 2
    xs = np.array([x0]*npoints)
    ys = np.array(substrate_locations[-1*npoints:])
    z_here = P(xs, ys, factor=0, preterm=20)
    plt.plot(xs, ys, color=bordercolor, linewidth=8)
    plt.scatter(xs, ys, c=z_here, cmap='viridis', vmin=0, vmax=Z.max(), s=300, edgecolor=bordercolor,
                linewidth=5, zorder=10)

    bordercolor = 'orangered'
    x0 = 5.66
    npoints = 3
    xs = np.array([x0]*npoints)
    ys = np.array(substrate_locations[-1*npoints:])
    z_here = P(xs, ys, factor=0, preterm=20)
    plt.plot(xs, ys, color=bordercolor, linewidth=8)
    plt.scatter(xs, ys, c=z_here, cmap='viridis', vmin=0, vmax=Z.max(), s=300, edgecolor=bordercolor,
                linewidth=5, zorder=10)

    bordercolor = 'red'
    x0 = 4.4
    npoints = 4
    xs = np.array([x0]*npoints)
    ys = np.array(substrate_locations[-1*npoints:])
    z_here = P(xs, ys, factor=0, preterm=20)
    plt.plot(xs, ys, color=bordercolor, linewidth=8)
    plt.scatter(xs, ys, c=z_here, cmap='viridis', vmin=0, vmax=Z.max(), s=300, edgecolor=bordercolor,
                linewidth=5, zorder=10)

    fig.savefig(f'figures/max_{num}_markers.png', dpi=300)
    plt.show()

    w, h = (5, 5)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w, h)
    ax0 = plt.Axes(fig, [0., 0., 1., 1.])
    ax0.set_axis_off()
    ax0.set_yscale('log')
    fig.add_axes(ax0)
    plt.xlabel('Configuration space of all ligands')
    plt.ylabel('Configuration space of all substrates', labelpad=15)

    x = np.linspace(-10, 10, 1000)

    y = np.logspace(-0.6, 1.4, 1000)
    X, Y = np.meshgrid(x, y)
    Z = P(X, Y, factor=0)
    Z[Z > thresh] = 100
    Z[Z <= thresh] = 0
    c = ax0.pcolormesh(X, Y, Z, cmap='Greys_r', vmin=0, vmax=1.01)
    fig.savefig(f'figures/max_{num}_mask.png', dpi=300)
    plt.show()