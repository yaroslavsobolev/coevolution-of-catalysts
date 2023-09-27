import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdCoordGen
from icecream import ic
from rdkit.DataStructs import TanimotoSimilarity as tanisim
from tqdm import tqdm
from sklearn.manifold import TSNE


def fingerprint(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    # if molecule is not nonetype then return the fingerprint
    if molecule:
        return AllChem.GetMorganFingerprint(molecule, 3)
    else:
        return np.NaN


def convert_full_DNP_to_dataframe_file(output_file='data/DNP_FULL_2016.pickle'):
    df = pd.read_csv('data/Classification/NPDB_linear_decs.tsv', sep='\t', names=['smiles', 'id', 'class']).astype(str)
    # append to df the dataframe from these additional files
    additional_file_list = [f'data/Classification/NPDB_{x}_decs.tsv' for x in
                            ['linked_cyclic', 'macro_cyclic', 'poly_cyclic']]
    for additional_file in additional_file_list:
        ic(additional_file)
        df = df.append(pd.read_csv(additional_file, sep='\t', names=['smiles', 'id', 'class'], usecols=[0, 1, 2]).astype(str),
                       ignore_index=True)
    ic(len(df))

    df['fp_ECFP6'] = df['smiles'].apply(fingerprint)

    # df.to_parquet(output_file)
    df.to_pickle(output_file)


def assert_that_select_polyketides_contain_intended_pattern(parquet_file='data/polyketides.parquet',
                                                            specific_molecular_pattern_SMARTS='[CH3]-[CH](-C=[C,c])-[CH](-[C,c])-O',
                                                            backup_general_pattern_SMARTS='C-C(-C=C)-C(-*)-O',
                                                            folder_for_output_images="figures/molecule_images/"):
    df = pd.read_parquet(parquet_file)
    df = df[df['synthesizable'] == 1]

    specific_molecular_pattern = Chem.MolFromSmarts(specific_molecular_pattern_SMARTS)
    img = Draw.MolToImage(specific_molecular_pattern)
    plt.imshow(img)
    plt.show()

    for i, row in df.iterrows():
        smi = row['smiles']
        ic(i)
        ic(smi)
        mol = Chem.MolFromSmiles(smi)
        if mol.HasSubstructMatch(specific_molecular_pattern):
            continue

        draw_pattern_within_molecule(backup_general_pattern_SMARTS, mol,
                                     image_filename=f'{folder_for_output_images}{i:04d}.png')

        print(f'{i} is not OK')
        break


def draw_pattern_within_molecule(pattern_smarts, molecule_smiles, image_filename):
    molecule = Chem.MolFromSmiles(molecule_smiles)
    pattern = Chem.MolFromSmarts(pattern_smarts)
    ic(len(molecule.GetSubstructMatches(pattern)))

    # Highlighting the part that matches the pattern
    hit_ats = list(molecule.GetSubstructMatch(pattern))
    hit_bonds = []
    for bond in pattern.GetBonds():
        aid1 = hit_ats[bond.GetBeginAtomIdx()]
        aid2 = hit_ats[bond.GetEndAtomIdx()]
        hit_bonds.append(molecule.GetBondBetweenAtoms(aid1, aid2).GetIdx())

    # Drawing the molecule
    d = rdMolDraw2D.MolDraw2DCairo(500, 500)
    rdCoordGen.AddCoords(molecule)
    rdMolDraw2D.PrepareAndDrawMolecule(d, molecule, highlightAtoms=hit_ats,
                                       highlightBonds=hit_bonds)
    png = d.GetDrawingText()
    with open(image_filename, 'wb') as f:
        f.write(png)


def compare_sets_and_update_DNP():
    fp_colname = 'fp_ECFP6'

    df_polyketides = pd.read_parquet('data/polyketides.parquet')
    df_polyketides = df_polyketides[df_polyketides['synthesizable'] == 1]
    df_polyketides = df_polyketides.reset_index(drop=True)
    df_polyketides[fp_colname] = df_polyketides['smiles'].apply(fingerprint)

    df_all = pd.read_pickle('data/DNP_FULL_2016.pickle')
    assert fingerprint(df_all.loc[10, 'smiles']) == df_all.loc[10, 'fp_ECFP6']
    # leave only rows where fp_ECFP6 is not nan
    df_all = df_all[~df_all['fp_ECFP6'].isna()]
    df_all = df_all.reset_index(drop=True)

    # iterate over polyketides df and find the most similar molecule in the full df
    absent_smileses = dict()
    for i, row in tqdm(df_polyketides.iterrows(), total=len(df_polyketides)):
        smiles = row['smiles']
        ic(i)
        fp = row[fp_colname]
        similarities = df_all[fp_colname].apply(lambda x: tanisim(x, fp))
        highest_similarity = similarities.max()
        # index in df_all of the most similar molecule
        highest_similarity_index = similarities.idxmax()
        # smiles in df_all at highest_similarity_index
        highest_similarity_smiles = df_all.loc[highest_similarity_index, 'smiles']
        # ic(highest_similarity, highest_similarity_index)
        print(f'Highest similarity {highest_similarity:.2f} at index {highest_similarity_index}')
        print('Polyketide smiles:')
        print(smiles)
        print('Highest similarity smiles:')
        print(highest_similarity_smiles)
        if highest_similarity < 1:
            absent_smileses[smiles] = fp
        else:
            # replace the smiles in df_all with the polyketide smiles
            df_all.loc[highest_similarity_index, 'smiles'] = smiles
    ic(len(absent_smileses))

    # make a dataframe from absent_smileses and append to df_all
    df_absent = pd.DataFrame.from_dict(absent_smileses, orient='index', columns=[fp_colname])
    df_absent['smiles'] = df_absent.index
    df_absent = df_absent.reset_index(drop=True)
    df_all_updated = df_all.append(df_absent, ignore_index=True)

    # save to pickle
    df_all_updated.to_pickle('data/DNP_FULL_2016_with_polyketides.pickle')


def calculate_distance_matrix(df, fp_colname = 'fp_ECFP6'):
    # define the distance metric
    if fp_colname == 'fp_ECFP6':
        def metric(a, b):
            return 1 - tanisim(a, b)
    else:
        raise ValueError('Unknown fingerprint type')

    # iterate over df rows and make a distance matrix
    dist_matrix = np.zeros((len(df), len(df)))
    for i in tqdm(range(len(df)), total=len(df), desc='Computing distance matrix'):
        for j in range(len(df)):
            dist_matrix[i, j] = metric(df.loc[i, fp_colname], df.loc[j, fp_colname])
    return dist_matrix


def distance_matrix(df, filename, fp_colname = 'fp_ECFP6', force_recalculate=False):
    # if file exists, then load by np.load. Else calculate and save
    if os.path.exists(filename) and not force_recalculate:
        dist_matrix = np.load(filename)
    else:
        dist_matrix = calculate_distance_matrix(df, fp_colname)
        np.save(filename, dist_matrix)
    return dist_matrix


def compute_tsne():
    fp_colname = 'fp_ECFP6'
    df_polyketides = pd.read_parquet('data/polyketides.parquet')
    df_polyketides = df_polyketides[df_polyketides['synthesizable'] == 1]
    df_polyketides = df_polyketides.reset_index(drop=True)
    df_polyketides[fp_colname] = df_polyketides['smiles'].apply(fingerprint)

    df = df_polyketides

    # iterate over df rows and make a distance matrix
    dist_matrix = np.zeros((len(df), len(df)))
    for i in tqdm(range(len(df)), total=len(df), desc='Computing distance matrix'):
        for j in range(len(df)):
            distance = 1 - tanisim(df.loc[i, fp_colname], df.loc[j, fp_colname])

            # power = 1/7
            # frac = 1 - 1/1000
            # dist_matrix[i, j] = (1 / (1 - tmap_distance*frac))**(power) - 1
            # dist_matrix[i, j] /= (1 / (1 - frac))**(power) - 1

            dist_matrix[i, j] = distance

    tsne = TSNE(n_components=2, verbose=1, random_state=125, metric='precomputed', early_exaggeration=1,
                perplexity=30, init='random', n_iter=10000, n_iter_without_progress=300, method='exact')
    z = tsne.fit_transform(dist_matrix)
    xs = z[:, 0]
    ys = z[:, 1]
    # angle = np.pi / 2
    # df["x"] = xs * np.cos(angle) - ys * np.sin(angle)
    # df["y"] = xs * np.sin(angle) + ys * np.cos(angle)
    df["x"] = xs
    df["y"] = ys

    # save embedding df to pickle
    df.to_pickle('data/polyketides_embedding.pickle')


if __name__ == '__main__':
    # assert_that_select_polyketides_contain_intended_pattern()
    # convert_full_DNP_to_dataframe_file()
    # compute_tsne()


    df = pd.read_pickle('data/polyketides_embedding.pickle')
    col_name = 'BB3'
    # in this column, change nans to zero
    df[col_name] = df[col_name].fillna(0)
    palette_name = 'bwr'

    fig = plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(x="x", y="y", data=df, hue=col_name, palette=palette_name, edgecolor=None, alpha=0.5, vmin=0, vmax=1)

    # filter only those that have not nan in col 'BB6'
    df2 = df[df['BB6'].notna()]
    ax2 = sns.scatterplot(ax=ax, x="x", y="y", data=df2, hue='BB6', palette='winter_r', edgecolor=None, alpha=0.5, vmin=0,
                         vmax=1)
    # kws = {"facecolor": "none", "linewidth": 1}
    # sns.scatterplot(x="x", y="y", data=df, edgecolor='black', alpha=0.5, ax=ax, **kws)
    # ax.set(title="Ligands T-SNE projection")
    ax.get_legend().remove()
    # ax.figure.colorbar(sm, ax=ax)
    plt.axis('equal')
    plt.axis('off')
    plt.show()