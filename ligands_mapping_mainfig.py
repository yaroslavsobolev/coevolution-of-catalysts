import logging
import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy.orm
from matplotlib.colors import ListedColormap
from halides_selection import *
from halides_clustering import *


def make_combined_dataset_of_ligands():
    '''Combine the ligands from the two datasets: coevolution and commercial KRAKEN'''
    fp_colname = 'fp_ECFP6'

    df_coev = pd.read_csv('data/ligands.csv')
    # rename 'SMILES' column to 'smiles'
    df_coev = df_coev.rename(columns={'SMILES': 'smiles'})

    df_kraken = pd.read_excel('data/kraken_commercial.xlsx')

    for df_here in [df_coev]:
        df_here['smiles'] = df_here['smiles'].apply(Chem.CanonSmiles)
        # df_here[fp_colname] = df_here['smiles'].apply(fingerprint)
        # df_here[fp_colname + '_bv'] = df_here['smiles'].apply(fingerprint_bitvect)

    # Add a column 'in_kraken' to df_coev and fill it with zeros
    df_coev['in_kraken'] = 0

    # Iterate through the rows of df_kraken. If the smiles is in df_coev, set 'in_kraken' to 1. Otherwise, append the row to df_coev
    for idx, row in df_kraken.iterrows():
        if row['smiles'] in df_coev['smiles'].values:
            df_coev.loc[df_coev['smiles'] == row['smiles'], 'in_kraken'] = 1
        else:
            df_coev = df_coev.append(row, ignore_index=True)
            # in this appended row, set 'in_kraken' to 1
            df_coev.loc[df_coev['smiles'] == row['smiles'], 'in_kraken'] = 1

    df_here = df_coev
    df_here[fp_colname] = df_here['smiles'].apply(fingerprint)
    df_here[fp_colname + '_bv'] = df_here['smiles'].apply(fingerprint_bitvect)
    df_here.to_pickle('data/ligands_coev_plus_kraken.pickle')


def plot_embedding(n_neighbours, min_dist, withlegend=False, suffix=''):
    filepath_without_extension = 'ligands_coev_plus_kraken'
    db_filepath = 'data/ligands/' + f'{filepath_without_extension}_umap_nn{n_neighbours}_md{min_dist}.hdf'
    # load to df and plot 'x' vs 'y'
    df = pd.read_hdf(db_filepath)
    fig = plt.figure(figsize=(6, 6))
    # plt.scatter(df['x'], df['y'], c=df['in_kraken'], cmap=ListedColormap(['red', 'blue']), alpha=0.5)
    # plot those with 'in_kraken' == 1 with C0 blue fill
    if withlegend:
        label='Commercial (KRAKEN)'
    else:
        label=None
    sns.scatterplot(x="x", y="y", data=df[df['in_kraken'] == 1], edgecolor='none', alpha=0.4, color='C0',
                    label=label)

    # plot x,y for those that have not nan in the 'A' column, use no fill and black edge, empty circle
    if withlegend:
        label='This work'
    else:
        label=None
    kws = {"facecolor": "none", "linewidth": 2}
    sns.scatterplot(x="x", y="y", data=df[df['A'].notna()], edgecolor='black', alpha=0.7, label=label, **kws)

    if withlegend:
        label='Productive with substrate 4'
    else:
        label=None
    # plot x,y for those that have non-zero in 'C' column, use light yellow and low-zorder, large circular marker, and no edge
    sns.scatterplot(x="x", y="y", data=df[df['C'] > 0], edgecolor='none', alpha=1, color='khaki', zorder=-10, s=200,
                    label=label)

    plt.axis('equal')
    plt.axis('off')
    if withlegend:
        plt.legend(loc='upper left')
    # plt.tight_layout()
    fig.savefig(f'figures/embeddings/ligands/ligands_coev_plus_kraken_umap_nn{n_neighbours}_md{min_dist}{suffix}.png', dpi=300)
    fig.savefig(f'figures/embeddings/ligands/ligands_coev_plus_kraken_umap_nn{n_neighbours}_md{min_dist}{suffix}.svg')

    maxligand_ids = [1, 10, 31, 37]
    colors = ['C1', 'C2', 'C3', 'C4']
    for id in maxligand_ids:
        df_with_this_point = df[df['id'] == id]
        # scatter this point
        plt.scatter(df_with_this_point['x'], df_with_this_point['y'], color=colors[maxligand_ids.index(id)], s=10,
                    label=f'id={id}')
    plt.legend()
    fig.savefig(f'figures/embeddings/ligands/ligands_coev_plus_kraken_umap_nn{n_neighbours}_md{min_dist}{suffix}_marked_ligands.png')


def plot_embeddings_for_the_hyperparameter_grid(n_neighbours, min_dist, ax):
    filepath_without_extension = 'ligands_coev_plus_kraken'
    db_filepath = 'data/ligands/' + f'{filepath_without_extension}_umap_nn{n_neighbours}_md{min_dist}.hdf'
    # load to df and plot 'x' vs 'y'
    df = pd.read_hdf(db_filepath)
    # subtract mean from x and y
    df['x'] -= df['x'].mean()
    df['y'] -= df['y'].mean()
    # divide by the maximum absolute value of x and y
    maxval = max(df['x'].abs().max(), df['y'].abs().max())
    df['x'] /= maxval
    df['y'] /= maxval

    point_size = 20
    label='Commercial (KRAKEN)'
    sns.scatterplot(ax=ax, x="x", y="y", data=df[df['in_kraken'] == 1], edgecolor='none', alpha=0.4, color='C0',
                    label=label, s=point_size)

    # plot x,y for those that have not nan in the 'A' column, use no fill and black edge, empty circle
    label='This work'
    kws = {"facecolor": "none", "linewidth": 2}
    sns.scatterplot(ax=ax, x="x", y="y", data=df[df['A'].notna()], edgecolor='black', alpha=0.7, label=label,
                    s=point_size, **kws)

    label='Productive with substrate 4'
    # plot x,y for those that have non-zero in 'C' column, use light yellow and low-zorder, large circular marker, and no edge
    sns.scatterplot(ax=ax, x="x", y="y", data=df[df['C'] > 0], edgecolor='none', alpha=1, color='khaki', zorder=-10, s=200,
                    label=label)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # make_combined_dataset_of_ligands()

    # make plots for all UMAP hyperparameters
    for n_neighbours in [3, 10, 30, 100]:
        for min_dist in [0.01, 0.05, 0.1, 0.2, 0.5, 1]:
            plot_embedding(n_neighbours, min_dist)
            # plt.show()
            # clear all matplotlib figures
            plt.close('all')

    # make plots for specific UMAP hyperparameters
    n_neighbours = 30
    min_dist = 0.2
    plot_embedding(n_neighbours, min_dist)
    # plot_embedding(n_neighbours, min_dist, withlegend=True, suffix='_withlegend')
    plt.show()

    # make a grid of plots with different hyperparameters for the Supplementary Information
    # make a fig, axarr with grid
    fig, axarr = plt.subplots(6, 4, figsize=(22, 30), sharex=True, sharey=True)
    for i, n_neighbours in enumerate([3, 10, 30, 100]):
        for j, min_dist in enumerate([0.01, 0.05, 0.1, 0.2, 0.5, 1]):
            plot_embeddings_for_the_hyperparameter_grid(n_neighbours, min_dist, ax=axarr[j, i])
            # make a title with hyperparameters for this subplot
            # axarr[j, i].set_title(f'n_neighbours={n_neighbours}, min_dist={min_dist}')
            # remove all axes and ticks
            axarr[j, i].axis('off')
            # make equal aspect ratio
            axarr[j, i].set_aspect('equal')
            # remove legends
            axarr[j, i].get_legend().remove()
    # plt.tight_layout()
    # make a vertical label to the left of the grid to indicate min_dist
    font_size = 20
    for j, min_dist in enumerate([0.01, 0.05, 0.1, 0.2, 0.5, 1]):
        axarr[j, 0].text(-0.5, 0.5, f'min_dist={min_dist}', rotation=90, verticalalignment='center', horizontalalignment='center',
                        transform=axarr[j, 0].transAxes, fontsize=font_size)
    # make a title to the top of the first row of plot to indicate n_neighbours
    for i, n_neighbours in enumerate([3, 10, 30, 100]):
        axarr[0, i].text(0.5, 1.1, f'n_neighbors={n_neighbours}', verticalalignment='center', horizontalalignment='center',
                        transform=axarr[0, i].transAxes, fontsize=font_size)

    # plt.tight_layout()
    fig.savefig('figures/embeddings/ligands/ligands_coev_plus_kraken_umap_grid.png', dpi=150)
    plt.show()


