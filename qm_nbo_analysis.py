import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from rdkit import Chem
from halides_selection import fix_halide_smiles
import logging
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

logging.basicConfig(level=logging.INFO)


def locate_first_line_containing_text(filepath, string):
    # find the index of the line that has string 'Input orientation:'
    found = False
    i = -1  # setting this here for the IDE to stop complaining
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if string in line:
                found = True
                break
    if not found:
        i = -1
    return i


def locate_last_line_containing_text(filepath, string):
    # find the index of the line that has string 'Input orientation:'
    found = False
    i = -1  # setting this here for the IDE to stop complaining
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(reversed(lines)):
            if string in line:
                found = True
                break
    if not found:
        i = -1
    return len(lines) - i - 1


def gaussian_has_terminated_normally(filepath):
    line_index = locate_first_line_containing_text(filepath, 'Normal termination of Gaussian')
    return line_index != -1


def get_atom_coordinates(filepath):
    starting_row = locate_first_line_containing_text(filepath, 'Input orientation:') + 5
    ending_row = locate_first_line_containing_text(filepath, 'Distance matrix (angstroms):') - 1
    return pd.read_csv(filepath, skiprows=starting_row, nrows=ending_row - starting_row, delim_whitespace=True,
                       names=['center_id', 'atomic_number', 'atomic_type', 'x', 'y', 'z'])


def closest_carbon_to_bromine(df_coords):
    bromine_coords = df_coords[df_coords['atomic_number'] == 35]
    assert len(bromine_coords) == 1

    def distance_between_rows(i, j):
        return ((df_coords.iloc[i]['x'] - df_coords.iloc[j]['x']) ** 2 +
                (df_coords.iloc[i]['y'] - df_coords.iloc[j]['y']) ** 2 +
                (df_coords.iloc[i]['z'] - df_coords.iloc[j]['z']) ** 2) ** 0.5

    df_coords['distance_to_bromine'] = df_coords.apply(
        lambda row: distance_between_rows(row.name, bromine_coords.index[0]), axis=1)

    # find the center_id of carbon that is closest to the bromine
    df_coords = df_coords[df_coords['atomic_number'] == 6]
    df_coords = df_coords.sort_values(by='distance_to_bromine')
    center_id_of_carbon = df_coords.iloc[0]['center_id']

    return center_id_of_carbon


def get_npa_data(filepath):
    start_line = locate_last_line_containing_text(filepath, ' Summary of Natural Population Analysis:') + 6
    end_line = locate_last_line_containing_text(filepath, 'Natural Population') - 3
    return pd.read_csv(filepath, skiprows=start_line, nrows=end_line - start_line, delim_whitespace=True,
                       names=['Atom', 'center_id', 'natural_charge', 'core', 'valence', 'rydberg', 'total'])


def npa_charge_of_carbon_closest_to_bromine(filepath):
    df_coords = get_atom_coordinates(filepath)
    center_id_of_carbon = closest_carbon_to_bromine(df_coords)
    df_npa = get_npa_data(filepath)
    df_npa = df_npa[df_npa['center_id'] == center_id_of_carbon]
    return df_npa['natural_charge'].values[0]


def examples_of_gaussian_functions():
    print(gaussian_has_terminated_normally('data/qm/1.out'))
    df_coords = get_atom_coordinates('data/qm/1.out')
    print(df_coords.to_string())
    print(closest_carbon_to_bromine(df_coords))

    print(locate_last_line_containing_text('data/qm/1.out', ' Summary of Natural Population Analysis:'))

    df_npa = get_npa_data('data/qm/1.out')
    print(df_npa.to_string())

    print(npa_charge_of_carbon_closest_to_bromine('data/qm/1.out'))

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def plot_supplementary_figure_with_npa_charge_histograms():
    df = pd.read_csv('data/qm/unique_halides_and_bb_vacuum.csv')
    # df = df[df['npa'].notna()]
    df['smiles'] = df['smiles'].apply(Chem.CanonSmiles)
    # fill npa with random numbers from 0 to -0.4
    # df['npa'] = np.random.uniform(0, -0.4, len(df))

    df_rec = pd.read_pickle('data/unique_halides_reclassed.pickle')
    df_rec = fix_halide_smiles(df_rec, colname='smiles')
    df_rec['smiles'] = df_rec['smiles'].apply(Chem.CanonSmiles)
    # get counts by bb values and print them
    counts = df_rec['bb'].value_counts()
    print(counts.to_string())

    # # canonicalize smiles in df_rec
    # df_rec['smiles'] = df_rec['smiles'].apply(Chem.CanonSmiles)
    #
    # # canonicalize smiles in df
    # df['smiles'] = df['smiles'].apply(Chem.CanonSmiles)

    for i, row in df.iterrows():
        # find the row in df that matches the smiles
        df_row = df_rec[df_rec['smiles'] == row['smiles']]
        if len(df_row) >= 1:
            df.loc[i, 'bb'] = df_row['bb'].values[0]
        else:
            print('No match for', row['smiles'])

    print(df['bb'].value_counts().to_string())

    # df.drop(columns=['count', 'fp_ECFP6', 'filename', 'Unnamed: 0']).to_csv('D:/Docs/Dropbox/Lab/catalyst-coevolution/unique_halides_NPA_rc.csv', index=False)

    # df_rec = df = df[df['npa'].notna()]

    master_smiles = ['Br/C=C/C',
                     'Br/C=C\C',
                     'Br/C=C(C)/C',
                     'Br/C(C)=C/C',
                     'Br/C=C/C1=CC=CC=C1',
                     'Br/C=C(C(OC(C)(C)C)=O)\C',
                     'Br/C=C(C(N(C)C)=O)\C',
                     'Br/C=C(C)/C(C)=O',
                     'Br/C=C/C(OC(C)(C)C)=O',
                     'Br/C=C/C(N(C)C)=O',
                     'Br/C=C/C(C)=O',
                     'C=C(C)Br']
    master_smiles = [Chem.CanonSmiles(smiles) for smiles in master_smiles]
    bbs = [f'BB{n + 1}' for n in range(len(master_smiles))]

    select_BBs = ['BB1', 'BB3', 'BB9', 'BB11', 'BB12']
    bins = [40, 20, 2, 1, 20]

    # make 4 subplots, one for each BB group.
    fig, axarr = plt.subplots(5, 1, figsize=(6, 8), sharex=True)
    # make histogram of npa for each BB group
    for i, bb in enumerate(select_BBs):
        df_bb = df[df['bb'] == bb]
        # add gaps between the bins
        axarr[i].hist(df_bb['npa'], color='C0', alpha=0.5, bins=bins[i])
        axarr[i].invert_xaxis()
        simpleaxis(axarr[i])
        # overlay individual points on the histogram
        axarr[i].scatter(df_bb['npa'], np.random.uniform(3, 10, len(df_bb)),
                         marker='o', alpha=1, s=4, c='k', zorder=100)
        # get values of npa for the master smiles
        master_here = master_smiles[bbs.index(bb)]
        df_master = df[df['smiles'] == master_here]
        if len(df_master) == 1:
            assert len(df_master) == 1
            npa_master = df_master['npa'].values[0]
            # plot a vertical line at the value of npa for the master
            axarr[i].axvline(npa_master, color='C1', linewidth=5)
        axarr[i].set_ylabel('Count')
        if axarr[i].get_ylim()[1] == 1:
            # set only two ticks, at 0 and 1
            axarr[i].set_yticks([0, 1])
            # labels should have no decimal places
            axarr[i].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    plt.xlabel('Natural (NPA) charge on the bromine-bound carbon, e')

    # fig.tight_layout()
    fig.savefig('figures/npa_histograms.png', dpi=300)

    # leave only rows where '.' is not in smiles string

    plt.show()

if __name__ == '__main__':
    pass
    plot_supplementary_figure_with_npa_charge_histograms()

    df = pd.read_csv('data/qm/unique_halides_and_bb_vacuum.csv')
    df = fix_halide_smiles(df, colname='smiles')
    df = df.append({'smiles': 'Brc1ccccc1', 'npa': -0.120}, ignore_index=True)
    df['smiles'] = df['smiles'].apply(Chem.CanonSmiles)
    # append row with smiles 'Brc1ccccc1' and 'npa' = -0.120

    df_rec = pd.read_pickle('data/unique_halides_reclassed.pickle')
    df_rec = fix_halide_smiles(df_rec, colname='smiles')
    df_rec['smiles'] = df_rec['smiles'].apply(Chem.CanonSmiles)

    for i, row in df.iterrows():
        # find the row in df that matches the smiles
        df_row = df_rec[df_rec['smiles'] == row['smiles']]
        if len(df_row) >= 1:
            df.loc[i, 'bb'] = df_row['bb'].values[0]
        else:
            print('No match for', row['smiles'])

    substrate_colors = ['#d0d1d4',
                        '#f4e0eb',
                        '#d3b4d7',
                        '#a471b1']
    master_smiles = ['BrC1=CC=CC=C1',
                     'Br/C=C(C)/C(OC(C)(C)C)=O',
                     'Br/C=C(C)/C(N(C)C)=O',
                     'Br/C=C(C)\C']
    master_smiles = [Chem.CanonSmiles(smiles) for smiles in master_smiles]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    _, _, p = ax.hist(df['npa'], color='#009900', alpha=0.5, bins=80)
    ax.set_xlim(-0.306, -0.1)
    ax.invert_xaxis()
    simpleaxis(ax)

    # # color bars greater than mean_diff except the partial bar
    # for rectangle in p:
    #     if rectangle.get_x() >= -0.20:
    #         rectangle.set_facecolor('C1')

    # overlay individual points on the histogram

    # # vertical scatter version
    # ax.scatter(df['npa'], np.zeros(len(df)) + ax.get_ylim()[1] / 10.0,
    #                  marker='|', alpha=1, s=200, c='C2', zorder=100)
    ## version with np.random.uniform
    # ax.axhline(0, color='k', linewidth=1)
    ax.scatter(df['npa'], np.random.uniform(3, 15, len(df)),
                marker='o', alpha=1, s=4, c='k', zorder=100)

    # for i, master_here in enumerate(master_smiles):
    #     # get row where smiles matches master
    #     df_master = df[df['smiles'] == master_here]
    #     if len(df_master) >= 1:
    #         npa_master = df_master['npa'].values[0]
    #         # plot a vertical line at the value of npa for the master
    #         ax.axvline(npa_master, color=substrate_colors[i], linewidth=5, linestyle='--', alpha=1, zorder=-50,
    #                    ymax=0.85)
    #     else:
    #         print('No NPA for', master_here)

    ax.set_ylabel('Count of halides within bin')
    ax.set_ylim(0, 50)
    plt.xlabel('Natural (NPA) charge on the bromine-bound carbon, e')
    plt.tight_layout()

    # rect = plt.Rectangle((-0.01 - 0.1, -19),
    #                      width=0.012,  # 11.3,
    #                      height=18.5,
    #                      facecolor='white',
    #                      clip_on=False,
    #                      linewidth=0,
    #                      alpha=1.0, zorder=100)
    # ax.add_patch(rect)
    fig.savefig('figures/fig1_npa_histogram.png', dpi=300)
    fig.savefig('figures/fig1_npa_histogram.eps', dpi=300)

    plt.show()





