from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
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


def P(x, T):
    return np.exp(-(potential(x - np.log(T))) / T)

if __name__ == '__main__':
    fig = plt.figure(tight_layout=False, figsize=(12, 8))
    gs = gridspec.GridSpec(4, 2)

    ax0 = fig.add_subplot(gs[:, 0])
    plt.xlabel('Space of all catalysts (generalized coordinate in arbitrary units)')
    plt.ylabel('Space of all substrates (generalized coordinate in arbitrary units)', labelpad=15)

    x = np.linspace(-10, 10, 1000)

    y = np.logspace(-0.6, 1.4, 1000)
    X, Y = np.meshgrid(x, y)
    Z = P(X, Y)
    # Z[Z<1e-1] = np.nan
    c = ax0.pcolormesh(X, Y, Z, cmap='viridis', vmin=0, vmax=1.01)
    ax0.set_yscale('log')
    ax0.axes.xaxis.set_ticklabels([])
    ax0.axes.yaxis.set_ticklabels([])
    # ax0.axes.xaxis.set_ticks([])
    ax0.axes.yaxis.set_ticks([])

    ax0.text(-0.13, 1, 'A', transform=ax0.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='left')

    list_of_values = [0.3, 0.5, 1.3, 12]
    for i, value in enumerate(list_of_values):
        color = substrate_colors[3 - i]
        plt.axhline(y=value, color=color, linestyle='--', linewidth=3)

    cbar = fig.colorbar(c, cax=fig.add_axes([0.20, 0.95, 0.2, 0.02]), label='Yield', orientation='horizontal')

    axs = [fig.add_subplot(gs[0, 1])]
    simpleaxis(axs[0])
    for i in range(1, 4, 1):
        axs.append(fig.add_subplot(gs[i, 1], sharex=axs[0]))
        simpleaxis(axs[i])
    for i in range(0, 3, 1):
        plt.setp(axs[i].get_xticklabels(), visible=False)

    panel_labels = ['B', 'C', 'D', 'E']
    for i in range(4):
        axs[i].text(-0.17, 1.02, panel_labels[i], transform=axs[i].transAxes,
                    fontsize=16, fontweight='bold', va='top', ha='left')
        axs[i].set_ylabel('Yield')

    axs[-1].set_xlabel('Space of all catalysts (generalized coordinate in arbitrary units)')
    axs[-1].axes.xaxis.set_ticklabels([])

    prev_max = 0
    for i, ax in enumerate(axs):
        ax.plot(x, P(x, list_of_values[::-1][i]), color=substrate_colors[i])
        ax.fill_between(x=x, y1=0, y2=P(x, list_of_values[::-1][i]), color=substrate_colors[i], alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_xlim(np.min(x), np.max(x))
        if i>0:
            x_of_prev_max = x[np.argmax(P(x, list_of_values[::-1][i-1]))]
            ax.scatter(x=x_of_prev_max, y=P(x_of_prev_max, list_of_values[::-1][i]), color=substrate_colors[i], s=20,
                       edgecolor='black', linewidth=2, zorder=10)


    fig.savefig('figures/ilustration_1.png', dpi=300)
    plt.show()
