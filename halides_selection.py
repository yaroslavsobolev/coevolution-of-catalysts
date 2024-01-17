import glob
import importlib
import logging
import os
import pickle
import random
import time
import numpy as np
import pandas as pd
import seaborn as sns
import tmap as tm
from icecream import ic
from joblib import Parallel, delayed
from map4 import MAP4Calculator
from matplotlib import pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, rdCoordGen
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.DataManip.Metric import GetTanimotoSimMat
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.DataStructs import TanimotoSimilarity as tanisim
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# pickle.HIGHEST_PROTOCOL = 4

# set logging level to info and set format to include time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Wrapper:
    def __init__(self, method_name, module_name):
        self.method_name = method_name
        self.module = importlib.import_module(module_name)

    @property
    def method(self):
        return getattr(self.module, self.method_name)

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)

def fix_halide_smiles(df, colname):
    pairs = [(Chem.CanonSmiles('C.CO/C(=C\\C(C)=C\\Br)C([O])=O'), Chem.CanonSmiles('CC(/C=C(OC)/C(OC)=O)=C\\Br')),
             (Chem.CanonSmiles('C.COC(=O)/C=C/C(C)=C/Br'), Chem.CanonSmiles('COC(/C=C/C(C)=C/Br)=O')),
             (Chem.CanonSmiles('C/C=C/C(=O)C(C)(C)C(=O)NC'), Chem.CanonSmiles('Br/C=C/C(=O)C(C)(C)C(=O)NC'))
             ]
    for look_for, replace_with in pairs:
        # in rows where colname converted to CanonSmiles is equal to look_for, replace with replace_with
        for i, row in df.iterrows():
            if row[colname]:
                if Chem.CanonSmiles(row[colname]) == look_for:
                    df.loc[i, colname] = replace_with
                    logging.info(f'fixed halide smiles in row {i}, smiles {row[colname]} turned into {replace_with}')
    return df


wrapped_mol_from_smiles = Wrapper("MolFromSmiles", "rdkit.Chem")

def parallel_mol_from_smiles(smiles, n_jobs=70):
    mols = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(wrapped_mol_from_smiles)(x) for x in smiles
    )
    return mols

def fingerprint(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    # if molecule is not nonetype then return the fingerprint
    if molecule:
        return AllChem.GetMorganFingerprint(molecule, 3)
    else:
        return np.NaN

def fingerprint_bitvect(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    # if molecule is not nonetype then return the fingerprint
    if molecule:
        return AllChem.GetMorganFingerprintAsBitVect(molecule, 3)
    else:
        return np.NaN

def chiral_fingerprint(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    # if molecule is not nonetype then return the fingerprint
    if molecule:
        return AllChem.GetMorganFingerprint(molecule, 3, useChirality=True)
    else:
        return np.NaN

def chiral_fingerprint_bitvect(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    # if molecule is not nonetype then return the fingerprint
    if molecule:
        return AllChem.GetMorganFingerprintAsBitVect(molecule, 3, useChirality=True)
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


def draw_pattern_within_molecule(pattern_smarts, molecule_smiles, image_filename, useChirality=True):
    molecule = Chem.MolFromSmiles(molecule_smiles)
    pattern = Chem.MolFromSmarts(pattern_smarts)
    ic(len(molecule.GetSubstructMatches(pattern, useChirality=True)))

    # Highlighting the part that matches the pattern
    hit_ats = list(molecule.GetSubstructMatch(pattern, useChirality=True))
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


def enrich_DNP_with_synthesizeable(parent_df_all_filename='data/DNP_FULL_2016.pickle',
                                   output_file='data/DNP_FULL_2016_with_polyketides.pickle'):
    fp_colname = 'fp_ECFP6'
    df_polyketides = pd.read_parquet('data/polyketides.parquet')
    df_polyketides = df_polyketides[df_polyketides['synthesizable'] == 1]
    df_polyketides = df_polyketides.reset_index(drop=True)
    df_polyketides[fp_colname] = df_polyketides['smiles'].apply(fingerprint)

    df_all = pd.read_pickle(parent_df_all_filename)
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
    df_all_updated.to_pickle(output_file)


def worker(input):
    i, fp, fps_list = input
    similarities = np.array(BulkTanimotoSimilarity(fp, fps_list))
    highest_similarity = similarities.max()
    highest_similarity_index = similarities.argmax()
    return i, fp, highest_similarity, highest_similarity_index

def update_DNP_with_manual_molecules(df_left, df_right, colname_for_flag='is_pk', force_calculate_fps=False,
                                     max_workers=70, chunksize=200):
    fp_colname = 'fp_ECFP6'
    fp_colname_bv = fp_colname + '_bv'

    # if "SMILES" column is in df_right, rename
    if "SMILES" in df_right.columns:
        df_right = df_right.rename(columns={'SMILES': 'smiles'})
    df_right = df_right.reset_index(drop=True)
    df_right = df_right[~df_right['smiles'].isna()]
    if force_calculate_fps:
        df_right[fp_colname] = df_right['smiles'].apply(fingerprint)
        df_right[fp_colname_bv] = df_right['smiles'].apply(fingerprint_bitvect)
        ic(len(df_right[df_right[fp_colname].isna()]))
        df_right = df_right[~df_right[fp_colname].isna()]

    # # Debugging: use only first 100 rows of df_polyketides
    # df_polyketides = df_polyketides.iloc[:10, :]

    assert fingerprint(df_left.loc[10, 'smiles']) == df_left.loc[10, 'fp_ECFP6']
    # leave only rows where fp_ECFP6 is not nan
    df_left = df_left[~df_left['fp_ECFP6'].isna()]
    df_left = df_left.reset_index(drop=True)
    if fp_colname_bv not in df_left.columns:
        logging.info(f'Calculating bitvect fingerprints for df_left...')
        df_left[fp_colname_bv] = df_left['smiles'].apply(fingerprint_bitvect)

    # iterate over polyketides df and find the most similar molecule in the full df
    # precompute similarity indices for all molecules in df_right
    fps_list = df_left[fp_colname_bv]
    # target_fp_list = df_right[fp_colname_bv]
    # arglist = [(i, fp, fps_list) for i, fp in enumerate(target_fp_list)]
    arglist = []
    for i, row in tqdm(df_right.iterrows(), total=len(df_right)):
        fp = row[fp_colname_bv]
        arglist.append((i, fp, fps_list))
    results = process_map(worker, arglist, max_workers=max_workers, chunksize=chunksize)

    results_dict = dict()
    for r in results:
        i, fp, highest_similarity, highest_similarity_index = r
        results_dict[i] = (fp, highest_similarity, highest_similarity_index)

    absent_smileses = dict()
    for i, row in tqdm(df_right.iterrows(), total=len(df_right)):
        smiles = row['smiles']
        fp = row[fp_colname_bv]
        # similarities = df_left[fp_colname].apply(lambda x: tanisim(x, fp))
        # highest_similarity = similarities.max()
        # highest_similarity_index = similarities.idxmax()

        # # faster version
        # similarities = np.array(BulkTanimotoSimilarity(fp, fps_list))
        # highest_similarity = similarities.max()
        # highest_similarity_index = similarities.argmax()

        # from parallel-preprocessed results
        copy_of_fp, highest_similarity, highest_similarity_index = results_dict[i]
        if not (tanisim(fp, copy_of_fp) == 1):
            print('Fingerprints are not equal!')
            print(fp)
            print(copy_of_fp)
        assert (tanisim(fp, copy_of_fp) == 1)
        highest_similarity_smiles = df_left.loc[highest_similarity_index, 'smiles']
        print(f'Highest similarity {highest_similarity:.2f} at index {highest_similarity_index}')
        print('Target smiles:')
        print(smiles)
        print('Highest similarity smiles:')
        print(highest_similarity_smiles)
        if highest_similarity < 1:
            absent_smileses[smiles] = fp
        else:
            # replace the smiles in df_all with the polyketide smiles
            print('Perfect similarity. Replacing smiles in df_all with target smiles')
            df_left.loc[highest_similarity_index, 'smiles'] = smiles
            df_left.loc[highest_similarity_index, colname_for_flag] = 1
    ic(len(absent_smileses))

    # make a dataframe from absent_smileses and append to df_all
    df_absent = pd.DataFrame.from_dict(absent_smileses, orient='index', columns=[fp_colname_bv])
    df_absent['smiles'] = df_absent.index
    df_absent[colname_for_flag] = 1
    df_absent = df_absent.reset_index(drop=True)
    df_all_updated = df_left.append(df_absent, ignore_index=True)

    # sort df_all_updated by is_pk

    # count the number of df_all molecules that are polyketides
    ic(len(df_all_updated[df_all_updated[colname_for_flag] == 1]))
    ic(len(df_right))

    # in is_pk column, set all nans to 0
    df_all_updated[colname_for_flag] = df_all_updated[colname_for_flag].fillna(0)
    # set types of column 'is_pk' to int
    df_all_updated[colname_for_flag] = df_all_updated[colname_for_flag].astype(int)

    # save to pickle
    return df_all_updated

def update_DNP_with_manual_polyketides(output_file='data/DNP_FULL_2016_with_manpk.pickle'):
    df_all_updated = update_DNP_with_manual_molecules(df_left=pd.read_pickle('data/DNP_FULL_2016.pickle'),
                                                      df_right=pd.read_pickle('data/manual_polyketides.pickle'),
                                                      colname_for_flag='is_pk',
                                                      force_calculate_fps=True)
    df_all_updated.to_pickle(output_file)


def compute_polyketides_fingerprints():
    fp_colname = 'fp_ECFP6'
    df_polyketides = pd.read_parquet('data/polyketides.parquet')
    df_polyketides = df_polyketides[df_polyketides['synthesizable'] == 1]
    df_polyketides = df_polyketides.reset_index(drop=True)
    df_polyketides[fp_colname] = df_polyketides['smiles'].apply(fingerprint)
    df_polyketides[fp_colname+'_bv'] = df_polyketides['smiles'].apply(fingerprint_bitvect)
    df_polyketides.to_pickle('data/polyketides_bitvect_fp.pickle')


def calculate_distance_matrix(df, fp_colname = 'fp_ECFP6_bv'):
    # define the distance metric
    if fp_colname == 'fp_ECFP6_bv':
        def metric(a, b):
            return 1 - tanisim(a, b)
        # iterate over df rows and make a distance matrix
        dist_matrix = np.zeros((len(df), len(df)), dtype=np.float16)
        tuple_of_fingerprints = tuple(df[fp_colname].to_list())
        for i in tqdm(range(len(df)), total=len(df), desc='Computing distance matrix'):
            for j in range(len(df)):
                dist_matrix[i, j] = metric(tuple_of_fingerprints[i], tuple_of_fingerprints[j])

    elif fp_colname == 'fp_MAP4_bv':
        logging.info('Converting np.array MAP4 fingerprints to tm.VectorUint...')
        tuple_of_fingerprints = tuple([tm.VectorUint(fp) for fp in df[fp_colname].to_list()])
        dist_matrix = np.zeros((len(df), len(df)), dtype=np.float16)
        for i in tqdm(range(len(df)), total=len(df), desc='Computing distance matrix'):
            for j in range(len(df)):
                dist_matrix[i, j] = ENC.get_distance(tuple_of_fingerprints[i], tuple_of_fingerprints[j])

    return dist_matrix

map4_dim = 1024
MAP4 = MAP4Calculator(dimensions=map4_dim)
ENC = tm.Minhash(map4_dim)


def map4_distance_with_unpacking(args_tuple):
    fp1, fp2 = args_tuple
    # since the MAP4 fingerprints are stored as np.arrays, we need to convert them to tmap.VectorUInt first
    return ENC.get_distance(tm.VectorUint(fp1), tm.VectorUint(fp2))


def map4_bulk_distance_function_for_single_column(args_tuple):
    fp, fps_list = args_tuple
    # print(fp[0])
    dist_column = np.zeros(len(fps_list), dtype=np.float16)
    fp1 = tm.VectorUint(fp)
    for j in range(len(fps_list)):
        dist_column[j] = ENC.get_distance(fp1, tm.VectorUint(fps_list[j]))
    return dist_column


def map4_bulk_distance_function(args_tuple):
    start_index, end_index, fps_list = args_tuple
    range_length = end_index - start_index
    # Converting to tmap.VectorUint because the MAP4 fingerprints are stored as np.arrays.
    # The reason for storing them as np.arrays is that tmap.VectorUint is not pickleable
    fps_list = tuple([tm.VectorUint(fp) for fp in fps_list])
    dist_submatrix = np.zeros((range_length, len(fps_list)), dtype=np.float16)
    for i in range(range_length):
        for j in range(len(fps_list)):
            dist_submatrix[i, j] = ENC.get_distance(fps_list[i + start_index], fps_list[j])
    return dist_submatrix


def distance_matrix(df, cache_filename, fp_colname ='fp_ECFP6_bv', force_recalculate=False, algo='loop',
                    memmap_mode=None, chunksize = 1000, max_workers=70):
    # if file exists, then load by np.load. Else calculate and save
    if os.path.exists(cache_filename) and not force_recalculate:
        logging.info(f'Loading distance matrix from {cache_filename} with memmap mode {memmap_mode}')
        dist_matrix = np.load(cache_filename, mmap_mode=memmap_mode)
    else:
        logging.info(f'Calculating distance matrix and saving to {cache_filename}')
        if fp_colname == 'fp_ECFP6_bv':
            def bulk_distance_function(fp, fps_list):
                return 1 - np.array(BulkTanimotoSimilarity(fp, fps_list))
        elif fp_colname == 'fp_MAP4_bv':
            def bulk_distance_function(fp, fps_list):
                return map4_bulk_distance_function(fp, fps_list)
        else:
            raise ValueError('Unknown fingerprint type')

        if algo=='matrix':
            logging.info('Extracting list of Morgan fingerprints..')
            t0 = time.time()
            morganfps = df[fp_colname].to_list()
            logging.info(f'Extracted list of Morgan fingerprints in {time.time() - t0:.2f} seconds')
            logging.info('Calculating Tanimoto similarity matrix..')
            t0 = time.time()
            tri_matrix = 1 - GetTanimotoSimMat(morganfps).astype(np.float16)
            logging.info(f'Calculated Tanimoto similarity matrix in {time.time() - t0:.2f} seconds')
            logging.info('Converting to squareform..')
            t0 = time.time()
            # dist_matrix = tri2mat2(tri_matrix) ## This version is slow due to use of Python loops
            dist_matrix = np.fliplr(np.flipud(squareform(np.flip(tri_matrix)))).astype(np.float16)
            logging.info(f'Converted to squareform in {time.time() - t0:.2f} seconds')
        elif algo=='loop':
            logging.info('Loading fingerprints..,')
            t0 = time.time()
            # morganfps = tuple([tm.VectorUint(fp) for fp in df[fp_colname].to_list()])
            morganfps = df[fp_colname]
            logging.info(f'Extracted list of fingerprints in {time.time() - t0:.2f} seconds')
            logging.info('Calculating distance matrix...')
            t0 = time.time()
            # iterate over df rows and make a distance matrix
            if fp_colname == 'fp_ECFP6_bv':
                dist_matrix = np.zeros((len(df), len(df)), dtype=np.float16)
                for i in tqdm(range(len(df)), total=len(df), desc='Computing distance matrix'):
                    dist_matrix[i, :] = bulk_distance_function(morganfps[i], morganfps)
            elif fp_colname == 'fp_MAP4_bv':
                logging.info('Making packed arglist for parallel map...')
                # Here, outer loop over columns is parallelized: each parallel process computes a "chunk" -- a number
                # of columns equal to chunksize. Therefore, each parallel process does a simple slow Python "outer"
                # loop over columns of this chunk and "inner" loop over all rows. Start and end indices for each
                # chunk, but in such a way that the chunk index does not exceed the length of the list of fingerprints.
                arglist = [(i, min(i + chunksize, len(df)), morganfps) for i in range(0, len(df), chunksize)]
                logging.info('Calculating approximate Jaccard distance matrix in parallel...')
                t0 = time.time()
                dist_matrix = np.concatenate(process_map(map4_bulk_distance_function, arglist,
                                                         max_workers=max_workers, chunksize=1), axis=0).astype(np.float16)
            logging.info(f'Calculated distance matrix in {time.time() - t0:.2f} seconds')
            # logging.info('Converting to squareform..')
            # t0 = time.time()
            # dist_matrix = tri2mat2(tri_matrix) ## This version is slow due to use of Python loops
            # logging.info(f'Converted to squareform in {time.time() - t0:.2f} seconds')

        elif algo=='two_loops':
            dist_matrix = calculate_distance_matrix(df, fp_colname)
        else:
            raise ValueError('Unknown algorithm')
        np.save(cache_filename, dist_matrix)
    return dist_matrix

def tri2mat2(tri_arr):
    n = len(tri_arr)
    m = int((np.sqrt(1 + 4 * 2 * n) + 1) / 2)
    arr = np.zeros([m, m])
    counter=0
    for i in range(m):
        for j in range(i):
            arr[i,j] = tri_arr[counter]
            arr[j,i] = tri_arr[counter]
            counter+=1
    return arr

def rotate_coordinates(xs, ys, angle):
    res_xs = xs * np.cos(angle) - ys * np.sin(angle)
    res_ys = xs * np.sin(angle) + ys * np.cos(angle)
    return res_xs, res_ys


def compute_tsne_draft():
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

    df["x"] = xs
    df["y"] = ys

    # save embedding df to pickle
    df.to_pickle('data/polyketides_embedding.pickle')


def compute_fingerprints_for_df(db_filepath, fp_colname = 'fp_ECFP6', force_recalculate=False,
                                N_PROC=70, output_filepath=None):
    db_filepath_without_extension = db_filepath.split('.')[0]
    if output_filepath is None:
        output_filepath = f'{db_filepath_without_extension}_map4fingerprints.pickle'
    if db_filepath.endswith('.pickle'):
        df = pd.read_pickle(db_filepath)
    elif db_filepath.endswith('.hdf'):
        df = pd.read_hdf(db_filepath, key='df')
        if 'smiles' not in df.columns:
            # convert index to smiles column
            df['smiles'] = df.index
            # reindex with ints as index
            df_all = df.reset_index(drop=True)
    # print number of rows
    logging.info(f'Loaded {len(df)} rows from {db_filepath}')
    # if colump fp_colname does not exist, compute fingerprints
    if force_recalculate or (fp_colname not in df.columns):
        logging.info(f'Computing fingerprints and saving to column {fp_colname}')
        if fp_colname == 'fp_ECFP6':
            df[fp_colname] = df['smiles'].apply(fingerprint)
            df[fp_colname + '_bv'] = df['smiles'].apply(fingerprint_bitvect)
            df.to_pickle(f'{db_filepath_without_extension}_fingerprints.pickle')
        elif fp_colname == 'fp_MAP4':
            logging.info('Computing rdkit Mol objects from smiles...')
            # array_of_mols = [Chem.MolFromSmiles(smi) for smi in df['smiles']]
            # Parallel version with pool.map:
            # with Pool(N_PROC) as pool:
            #     array_of_mols = pool.map(wrapped_mol_from_smiles, df['smiles'].to_list())
            array_of_mols = parallel_mol_from_smiles(df['smiles'].to_list(), N_PROC)
            indices_of_NoneType_in_array_of_mols = [i for i, x in enumerate(array_of_mols) if x is None]
            print(f'Nones : {indices_of_NoneType_in_array_of_mols}')
            for i in indices_of_NoneType_in_array_of_mols:
                print(df.loc[i, 'smiles'])
            logging.info('Computing MAP4 fingerprints...')
            fps = MAP4.calculate_many(array_of_mols)
            logging.info('Converting fingerprints to np.arrays because tmap.VectorUint is not pickleable...')
            fps = [np.array(fp) for fp in fps]
            df[fp_colname] = fps
            df[fp_colname + '_bv'] = fps
            df.to_pickle(output_filepath)

    else:
        logging.info(f'Fingerprints already exist in column {fp_colname}. Checking if they are correct in the first 10 rows')
        # assert that the fingerprints are correct in the first 10 rows
        for i in range(10):
            assert df.loc[i, fp_colname] == fingerprint(df.loc[i, 'smiles'])
            assert df.loc[i, fp_colname + '_bv'] == fingerprint_bitvect(df.loc[i, 'smiles'])


def compute_tsne_with_distmatrix_and_save(db_filepath, distmatrix_cache_filename='auto', output_filename='auto',
                                          fp_colname = 'fp_ECFP6_bv', tsne_params = None, memmap_mode=None,
                                          limit_rows=None, change_dtypes_to_str=True):
    # extract the filepath without extension
    db_filepath_without_extension = db_filepath.split('.')[0]
    if distmatrix_cache_filename == 'auto':
        distmatrix_cache_filename = db_filepath_without_extension + '_distance_matrix.npy'
    if tsne_params is None: # use default parameters
        tsne_params = {'n_components': 2, 'verbose': 2, 'random_state': 137, 'metric': 'precomputed',
                       'early_exaggeration': 1, 'perplexity': 300, 'init': 'random', 'n_iter': 5000,
                       'n_iter_without_progress': 500, 'method': 'barnes_hut', 'n_jobs':1,
                       'learning_rate': 1000}
    if output_filename == 'auto':
        output_filename = db_filepath_without_extension + '_tsne.hdf'
    df = pd.read_pickle(db_filepath)
    if limit_rows is not None:
        df = df.iloc[:limit_rows]
    dist_matrix = distance_matrix(df, cache_filename=distmatrix_cache_filename, fp_colname=fp_colname, memmap_mode=memmap_mode)
    if limit_rows is not None:
        dist_matrix = dist_matrix[:limit_rows, :limit_rows]
    logging.info(f'Loaded distance matrix from {distmatrix_cache_filename}')
    tsne = TSNE(**tsne_params)
    # write TSNE with default parameters instead of tsne_params dictionary
    # tsne = TSNE(n_components=2, verbose=2, random_state=137, metric='precomputed', early_exaggeration=1,
    #             perplexity=30, init='random', n_iter=5000, n_iter_without_progress=100, method='barnes_hut', n_jobs=70)

    z = tsne.fit_transform(dist_matrix)
    df["x"] = z[:, 0]
    df["y"] = z[:, 1]

    # drop these columns because they are objects and cannot be saved to hdf without pickle, but pickle
    # has compatibitily issues betweeb versions of python/pickle/pandas
    if 'fp_ECFP6_bv' in df.columns:
        df = df.drop(columns=['fp_ECFP6_bv', 'fp_ECFP6'])
    if 'fp_MAP4_bv' in df.columns:
        df = df.drop(columns=['fp_MAP4_bv', 'fp_MAP4'])
    # change drypes to string
    if change_dtypes_to_str:
        df['smiles'] = df['smiles'].astype(str)
        df['id'] = df['id'].astype(str)
        df['class'] = df['class'].astype(str)
    df.to_hdf(output_filename, 'df')

    logging.info(f'Saved tsne embedding to {output_filename}')


def compute_tsne_with_callable_metric_and_save(db_filepath, force_recalculate_distances=False, fp_colname = 'fp_ECFP6'):
    # extract the filepath without extension
    db_filepath_without_extension = db_filepath.split('.')[0]

    # load dataframe and convert fingerprints to numpy arrays
    df = pd.read_pickle(db_filepath)
    final_array = []
    # iterate over rows of fp_colname column
    for fp in tqdm(df[fp_colname+'_bv'].to_list()):
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        final_array.append(np.copy(arr))
    final_array = np.array(final_array)

    def callable_metric(arr_x, arr_y):
        # since fingerprints are in numpy arrays, we need to convert them to bitstrings and then to RDKIt version back
        return 1 - tanisim(DataStructs.cDataStructs.CreateFromBitString("".join(arr_x.astype(str))),
                           DataStructs.cDataStructs.CreateFromBitString("".join(arr_y.astype(str))))

    tsne = TSNE(n_components=2, verbose=100, random_state=125, metric=callable_metric, early_exaggeration=1,
                perplexity=2, init='random', n_iter=250, n_iter_without_progress=30, n_jobs=10, method='barnes_hut')
    logging.info('Starting TSNE')
    z = tsne.fit_transform(final_array)
    df["x"] = z[:, 0]
    df["y"] = z[:, 1]
    # save embedding df to pickle
    df.to_pickle(db_filepath_without_extension + '_tsne.pickle')


def make_tsne_on_polyketides(filepath_to_df='data/polyketides_embedding.pickle'):
    df = pd.read_pickle(filepath_to_df)
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


def merge_columns_from_polyketides_to_parent():
    df_all = pd.read_pickle('data/DNP_FULL_2016_with_polyketides_fingerprints.pickle')
    df_pk = pd.read_pickle('data/polyketides_bitvect_fp.pickle')
    # make index of df_all from 'smiles' column
    df_all = df_all.set_index('smiles')
    df_pk = df_pk.set_index('smiles')

    # drop columns 'ecfp6' and 'ecfp6_bv' from df_pk
    df_pk = df_pk.drop(columns=['fp_ECFP6_bv', 'fp_ECFP6'])

    # join df_pk to df_all on index
    df_joined = df_all.join(df_pk, how='left')

    # save to pickle
    df_joined.to_pickle('data/DNP_FULL_2016_with_polyketides_allcols_fingerprints.pickle')

    return df_joined


def decimate_DNP(target_size):
    df_all = pd.read_pickle('data/DNP_FULL_2016_with_polyketides_fingerprints.pickle')
    df_pk = pd.read_pickle('data/polyketides_bitvect_fp.pickle')
    ref_smileses = set(df_pk['smiles'].to_list())
    # iterate through df_all and make a list of indices where smiles are in ref_smileses and a list of indices where
    # smiles in df_all are not in ref_smileses
    indices_to_keep = []
    indices_to_random_choose = []
    for i, row in tqdm(df_all.iterrows(), total=len(df_all)):
        if row['smiles'] in ref_smileses:
            indices_to_keep.append(i)
        else:
            indices_to_random_choose.append(i)
    logging.info(f'Number of indices to keep certainly: {len(indices_to_keep)}')
    final_list = indices_to_keep + random.sample(indices_to_random_choose, target_size - len(indices_to_keep))
    df_decimated = df_all.iloc[final_list]
    df_decimated.to_pickle(f'data/DNP_FULL_2016_with_polyketides_fingerprints_len{int(round(target_size/1000)):d}k.pickle')


def decimate_DNP_v2(target_size, source_df_filename):
    # df_all = pd.read_pickle(source_df_filename)
    df_all = pd.read_hdf(source_df_filename)
    # convert index to smiles column
    df_all['smiles'] = df_all.index
    # reindex with ints as index
    df_all = df_all.reset_index(drop=True)
    # df_pk is the subset of df_all that has is_pk that is 1 (is a polyketide) or synthesizable that is 1
    if 'is_steroid' not in df_all.columns:
        df_pk = df_all[(df_all['is_pk'] == 1) | (df_all['synthesizable'] == 1)]
    else:
        df_pk = df_all[(df_all['is_pk'] == 1) | (df_all['synthesizable'] == 1) | (df_all['is_steroid'] == 1) |
                       (df_all['is_terpene'] == 1) | (df_all['is_alkaloid'] == 1) | (df_all['is_flavonoid'])]
    ref_smileses = set(df_pk['smiles'].to_list())
    # iterate through df_all and make a list of indices where smiles are in ref_smileses and a list of indices where
    # smiles in df_all are not in ref_smileses
    indices_to_keep = []
    indices_to_random_choose = []
    for i, row in tqdm(df_all.iterrows(), total=len(df_all)):
        if row['smiles'] in ref_smileses:
            indices_to_keep.append(i)
        else:
            indices_to_random_choose.append(i)
    logging.info(f'Number of indices to keep certainly: {len(indices_to_keep)}')
    final_list = indices_to_keep + random.sample(indices_to_random_choose, target_size - len(indices_to_keep))
    df_decimated = df_all.iloc[final_list]

    df_decimated.to_pickle(f'data/DNP_FULL_2016_with_polyketides_len{int(round(target_size/1000)):d}k.pickle')


def convert_from_server_pickle_to_hdf():
    filepath_to_df = 'data/DNP_FULL_2016_with_polyketides_fingerprints_tsne_px30.pickle'
    df = pd.read_pickle(filepath_to_df)
    # df = df.drop(columns=['fp_ECFP6_bv', 'fp_ECFP6'])
    import pickle
    pickle.HIGHEST_PROTOCOL = 4
    df.to_hdf('data/DNP_FULL_2016_with_polyketides_fingerprints_tsne_px30_withfps.hdf', 'df')


def server_df_to_merged_df(filepath,
                           polyketides_df_pickle_filepath='data/polyketides_bitvect_fp.pickle',
                           output_filename=None):
    # if filepath ends with .hdf, read it with pandas.read_hdf
    if filepath.endswith('.hdf'):
        logging.info(f'Reading df from hdf {filepath}')
        df_all = pd.read_hdf(filepath)
    elif filepath.endswith('.pickle'):
        logging.info(f'Reading df from pickle {filepath}')
        df_all = pd.read_pickle(filepath)
    db_filepath_without_extension = filepath.split('.')[0]
    df_pk = pd.read_pickle(polyketides_df_pickle_filepath)
    # make index of df_all from 'smiles' column
    df_all = df_all.set_index('smiles')
    df_pk = df_pk.set_index('smiles')

    # drop columns 'ecfp6' and 'ecfp6_bv' from df_pk
    for col_here in ['fp_ECFP6_bv', 'fp_ECFP6']:
        if (col_here in df_all.columns):
            df_pk = df_pk.drop(columns=col_here)

    # join df_pk to df_all on index
    df_joined = df_all.join(df_pk, how='left')

    # # set c-types for certain columns to str
    # for colname in ['id', 'class', 'name', 'formula', 'Halide SMILES', 'Halide 2 SMILES', 'Halide 3 smiles']:
    #     df_joined[colname] = df_joined[colname].astype(str)


    # # mark alpha-methyl-beta-hydroxy
    # specific_molecular_pattern_SMARTS = '[CH3]-[CH](-[C,c])-[CH](-[C,c])-O'

    # # all polyketides
    # specific_molecular_pattern_SMARTS = 'C-C(~O)-C-C(~O)-C'
    #
    # specific_molecular_pattern = Chem.MolFromSmarts(specific_molecular_pattern_SMARTS)
    # for smi in tqdm(df_joined.index):
    #     mol = Chem.MolFromSmiles(smi)
    #     # if mol.HasSubstructMatch(specific_molecular_pattern):
    #     if len(mol.GetSubstructMatches(specific_molecular_pattern)) > 1:
    #         # mark in 'polyketideome' column as 1
    #         df_joined.loc[smi, 'polyketideome'] = 1
    #
    # print('Number of polyketides in the dataset: ', df_joined['polyketideome'].sum())

    # save to pickle
    if output_filename is None:
        df_joined.to_hdf(f'{db_filepath_without_extension}_pkmerged.hdf', 'df')
    else:
        df_joined.to_hdf(f'{output_filename}', 'df')

    return df_joined

def plot_tsne_from_hdf(hdf_filepath, title='', do_show=True, size=2, mainalpha=0.1):
    df = pd.read_hdf(hdf_filepath)
    #filename without extension
    db_filename = hdf_filepath.split('.')[0].split('/')[-1]

    # defining the colors into the 'color' column. If synthesizable column is 1, then color is red else blue
    # df['color'] = df['synthesizable'].apply(lambda x: 'PK' if x == 1 else 'Other')

    fig = plt.figure(figsize=(10, 10))
    ax = sns.scatterplot(x="x", y="y", s=size, data=df[df['is_pk']==0], color='black', edgecolor=None, alpha=mainalpha,
                         linewidth=0)
    sns.scatterplot(ax=ax, x="x", y="y", s=7, data=df[(df['is_pk']==1) & (df['synthesizable']==0)], color='C0', edgecolor=None, alpha=0.3,
                    label='Polyketideome', zorder=9, linewidth=0)
    sns.scatterplot(ax=ax, x="x", y="y", s=20, data=df[df['synthesizable']==1], color='C1', edgecolor=None, alpha=1,
                    label='Products you extracted vinyl halides from', zorder=10, linewidth=0)
    # sns.scatterplot(ax=ax, x="x", y="y", s=20, data=df[df['BB3'].notna()], color='C3', edgecolor=None, alpha=1,
    #                 label='BB3', zorder=20, linewidth=0)
    # sns.scatterplot(ax=ax, x="x", y="y", s=20, data=df[df['BB6'].notna()], color='C2', edgecolor=None, alpha=1,
    #                 label='BB6', zorder=30, linewidth=0)
    plt.tight_layout()
    plt.axis('equal')
    plt.axis('off')
    # find rad_lim radius limit such that 99% of points are within the circle
    rad_lim = np.percentile(np.sqrt((df['x'] - df['x'].mean()) ** 2 + (df['y']-df['y'].mean()) ** 2), 99)
    plt.xlim(-rad_lim, rad_lim)
    plt.ylim(-rad_lim, rad_lim)
    plt.legend()
    # add margin on the top for the title
    plt.subplots_adjust(top=0.95)
    plt.title(title)
    fig.savefig(f'figures/embeddings/{db_filename}.png', dpi=300)
    if do_show:
        plt.show()
    else:
        # delete figure fig
        plt.close(fig)
        plt.clf()
        plt.cla()


def select_cluster(filename = 'DNP_FULL_2016_with_polyketides_fingerprints_len100k_tsne_px1060_lr24735_pkmerged.hdf',
                    x0 = -32.9682, y0 = -65.603, ymax = -62.766):
    rad = ymax-y0
    df = pd.read_hdf(f'data/{filename}')
    # select only synthesizable
    df = df[df['synthesizable'] == 1]
    # select rows whose distance from x0, y0 is less than rad
    df = df[(df['x'] - x0) ** 2 + (df['y'] - y0) ** 2 < rad ** 2]
    # save this df to hdf
    df.to_hdf(f'data/{filename[:-4]}_selected.hdf', key='df')


def draw_smarts_found_in_many_smiles(pickle_file,
                                pattern_SMARTS,
                                folder_for_output_images="figures/molecule_images/"):
    df = pd.read_pickle(pickle_file)

    specific_molecular_pattern = Chem.MolFromSmarts(pattern_SMARTS)
    img = Draw.MolToImage(specific_molecular_pattern)
    plt.imshow(img)
    plt.show()

    for i, row in df.iterrows():
        smi = row['smiles']
        ic(i)
        ic(smi)
        mol = Chem.MolFromSmiles(smi)
        if not mol.HasSubstructMatch(specific_molecular_pattern):
            continue

        draw_pattern_within_molecule(pattern_SMARTS, smi,
                                     image_filename=f'{folder_for_output_images}{i:04d}.png')


def draw_smarts_found_in_smiles(SMILES,
                                pattern_SMARTS,
                                folder_for_output_images="figures/molecule_images/"):

    specific_molecular_pattern = Chem.MolFromSmarts(pattern_SMARTS)
    img = Draw.MolToImage(specific_molecular_pattern)
    plt.imshow(img)
    plt.show()

    mol = Chem.MolFromSmiles(SMILES)
    if not mol.HasSubstructMatch(specific_molecular_pattern):
        print('No match')
    else:
        draw_pattern_within_molecule(pattern_SMARTS, SMILES,
                                     image_filename=f'{folder_for_output_images}lala.png')


def convert_and_view_server_df(server_hdf):
    server_df_to_merged_df(filepath=f'data/{server_hdf}',
                           polyketides_df_pickle_filepath='data/polyketides_bitvect_fp.pickle')
    server_hdf_without_extension = server_hdf[:-4]
    filename = f'{server_hdf_without_extension}_pkmerged.hdf'
    equivalent_perplexity = int(round(int(filename.split("_")[8][2:]) * 283 / 1000)) * 10
    plot_tsne_from_hdf(hdf_filepath=f'data/{filename}', title=f'Perplexity: {equivalent_perplexity}',
                       do_show=True)


def convert_and_view_server_df(server_hdf):
    server_df_to_merged_df(filepath=f'data/{server_hdf}',
                           polyketides_df_pickle_filepath='data/polyketides_bitvect_fp.pickle')
    server_hdf_without_extension = server_hdf[:-4]
    filename = f'{server_hdf_without_extension}_pkmerged.hdf'
    equivalent_perplexity = int(round(int(filename.split("_")[8][2:]) * 283 / 1000)) * 10
    plot_tsne_from_hdf(hdf_filepath=f'data/{filename}', title=f'Perplexity: {equivalent_perplexity}',
                       do_show=True)


def load_many_excel_with_molecules_into_df(source_folder, target_df_filename='data/manual_polyketides.pickle'):
    # find all .xls files in the folder
    # filenames = [f for f in os.listdir(source_folder) if f.endswith('.xls')]
    filenames = glob.glob(f'{source_folder}/**/*.xls', recursive=True)
    df = pd.DataFrame()
    for filename in filenames:
        logging.info(f'Loading {filename}')
        # df_from_excel = pd.read_excel(f'{source_folder}/{filename}', usecols=['Chemical Name', 'Molecular Formula', 'SMILES'])
        df_from_excel = pd.read_excel(filename,
                                      usecols=['Chemical Name', 'Molecular Formula', 'SMILES'])
        df = df.append(df_from_excel)
    df = df.reset_index(drop=True)
    logging.info(f'Loaded {len(df)} molecules from {len(filenames)} files')
    # save to pickle
    df.to_pickle(target_df_filename)
    return df


def fix_column_types(input_filename='data/DNP_FULL_2016_with_polyketides_len100k_fingerprints.pickle',
                     output_filename='data/DNP_FULL_2016_with_polyketides_fingerprints_len100k.pickle'):
    df = pd.read_pickle(input_filename)
    # rename column ''Halide 3 smiles' to 'Halide 3 SMILES'
    df = df.rename(columns={'Halide 3 smiles': 'Halide 3 SMILES'})
    for colname in ['id', 'class', 'name', 'formula', 'smiles', 'Halide SMILES', 'Halide 2 SMILES', 'Halide 3 SMILES']:
        # replace nan and None with empty string
        df[colname] = df[colname].fillna('')
        df[colname] = df[colname].astype(str)

    # fill nans with zero in certain columns ('synthesizable', 'is_pk', abd from 'BB1' to 'BB16')
    for colname in ['synthesizable', 'is_pk', 'BB_count'] + [f'BB{i}' for i in range(1, 17)] + ['is_steroid', 'is_terpene', 'is_flavonoid', 'is_alkaloid']:
        df[colname] = df[colname].fillna(0)
        # set dtype to int
        df[colname] = df[colname].astype(int)

    # in rows where synthesizable column is 1, set is_pk to 1 as well
    df.loc[df['synthesizable'] == 1, 'is_pk'] = 1

    # reindex
    df = df.reset_index(drop=True)
    # set smiles as first column, is_pk as second, synthesizable as third
    df = df[['smiles', 'is_pk', 'synthesizable'] +
                                [colname for colname in df.columns if colname not in ['smiles', 'is_pk', 'synthesizable']]]

    # save to pickle
    df.to_pickle(output_filename)



def get_unique_halides(unique_by='canonical_SMILES'):
    df_polyketides = pd.read_pickle('data/polyketides_bitvect_fp.pickle')
    fp_colname = 'fp_ECFP6'
    df_polyketides = df_polyketides[df_polyketides['synthesizable'] == 1]
    df_polyketides = df_polyketides.reset_index(drop=True)
    df_polyketides[fp_colname] = df_polyketides['smiles'].apply(fingerprint)

    df_polyketides = df_polyketides[~df_polyketides['fp_ECFP6'].isna()]
    df_polyketides = df_polyketides[df_polyketides['BB_count'] <= 3]
    df_polyketides = df_polyketides[df_polyketides['BB_count'] >= 1]
    df_polyketides = df_polyketides.reset_index(drop=True)

    list_of_bb_column_names = [f'BB{i}' for i in range(1, 17)]
    list_of_halide_smiles_columns = ['Halide SMILES', 'Halide 2 SMILES', 'Halide 3 smiles']
    # df_halides is a dataframe with columns 'smiles' and 'bb'
    df_halides = pd.DataFrame(columns=['smiles', 'bb'])

    # iterate over polyketides df and find the most similar molecule in the full df
    for i, row in tqdm(df_polyketides.iterrows(), total=len(df_polyketides)):
        # get list of BB columns where value is 1
        if row['BB_count'] > 1:
            continue
        bbs_present = [colname for colname in list_of_bb_column_names if row[colname] == 1]
        assert len(bbs_present) == 1
        for k, bb_column in enumerate(bbs_present):
            # if there is no smiles for this halide, skip
            if (row[list_of_halide_smiles_columns[k]] is None) or (row[list_of_halide_smiles_columns[k]] == ''):
                logging.info(f'No halide smiles for row {i}')
                continue
            # add 'Halide SMILES' to df_halides as 'smiles' column
            df_halides = df_halides.append({'smiles': row[list_of_halide_smiles_columns[k]],
                                            'bb': bb_column}, ignore_index=True)

    # Remove repeating halides
    if unique_by == 'fingerprint':
        # add column "to_drop" to df_halides
        df_halides['to_drop'] = False

        if fp_colname == 'fp_ECFP6':
            df_halides[fp_colname] = df_halides['smiles'].apply(fingerprint)
            df_halides[fp_colname + '_bv'] = df_halides['smiles'].apply(fingerprint_bitvect)

        # iterate over polyketides df and find the most similar molecule in the full df
        for i, row in tqdm(df_halides.iterrows(), total=len(df_halides)):
            if df_halides.loc[i, 'to_drop']:
                continue
            smiles = row['smiles']
            fp = row[fp_colname]
            similarities = df_halides[fp_colname].apply(lambda x: tanisim(x, fp))
            # in all the indices where similarity is exactly 1, mark for dropping
            df_halides.loc[similarities == 1, 'to_drop'] = True
            df_halides.loc[similarities == 1, 'sim_id'] = i
            df_halides.loc[i, 'count'] = len(df_halides[similarities == 1])
            df_halides.loc[i, 'to_drop'] = False
        df_halides = df_halides[df_halides['to_drop'] == False]
        df_halides = df_halides.reset_index(drop=True)
        # remove columns 'to_drop' and 'sim_id'
        df_halides = df_halides.drop(columns=['to_drop', 'sim_id'])

    elif unique_by == 'canonical_SMILES':
        # canonicalize the smiles column
        df_halides['smiles'] = df_halides['smiles'].apply(Chem.CanonSmiles)
        # remove duplicates by 'smiles' column (keep first), but add column that says the count of duplicates
        df_halides_without_duplicates = df_halides.groupby('smiles').first().reset_index()
        for i, row in df_halides_without_duplicates.iterrows():
            smiles_here = row['smiles']
            number_of_rows_in_df_halides_that_have_this_smiles = len(df_halides[df_halides['smiles'] == smiles_here])
            df_halides_without_duplicates.loc[i, 'count'] = number_of_rows_in_df_halides_that_have_this_smiles

        df_halides = df_halides_without_duplicates
        if fp_colname == 'fp_ECFP6':
            df_halides[fp_colname] = df_halides['smiles'].apply(chiral_fingerprint)
            df_halides[fp_colname + '_bv'] = df_halides['smiles'].apply(chiral_fingerprint_bitvect)



    df_halides.to_pickle('data/unique_halides.pickle')

def dft_npa_halides():
    df_halides = pd.read_pickle('data/unique_halides.pickle')
    smiles_to_add = ['Br/C=C/C',
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
    smiles_to_add = [Chem.CanonSmiles(smiles) for smiles in smiles_to_add]
    bbs = [f'BB{n+1}' for n in range(len(smiles_to_add))]
    for i, smiles in enumerate(smiles_to_add):
        if smiles not in df_halides['smiles'].values:
            df_halides = df_halides.append({'smiles': smiles, 'bb': bbs[i]}, ignore_index=True)
    df_halides['npa'] = np.nan
    df_halides['filename'] = np.nan

    df_halides.to_pickle('data/unique_halides_and_bb.pickle')
    return df_halides


def classify_diene(smiles):
    mol = Chem.MolFromSmiles(smiles)



def reclassify_BB4(folder_for_output_images='figures/molecule_images/'):
    df_halides = pd.read_pickle('data/unique_halides.pickle')
    for i, row in df_halides.iterrows():
        if not (row['bb'] == 'BB4'):
            continue
        smi = row['smiles']
        mol = Chem.MolFromSmiles(smi)

        patterns_smarts = {'BB1': 'Br/[CH]=[CH]/C',
                           'BB2': 'Br/[CH]=[CH]\C',
                           'BB3': 'Br/[CH]=C(*)*',
                           'BB4': 'Br/C(-*)=[CH]/*',
                           'BB12': 'BrC(*)=[CH2]'}

        found_something = False
        for bb in patterns_smarts.keys():
            if mol.HasSubstructMatch(Chem.MolFromSmarts(patterns_smarts[bb]), useChirality=True):
                # change 'bb' column in df_halides to this specific bb
                df_halides.loc[i, 'bb'] = bb
                draw_pattern_within_molecule(patterns_smarts[bb], smi,
                                             image_filename=f'{folder_for_output_images}{i:04d}_{bb}.png')
                found_something = True
                break

        if not found_something:
            img = Draw.MolToImage(mol)
            plt.imshow(img)
            plt.title('UNCLASSIFIED')
            plt.show()

    # classify this one as BB4, even though it's cis relative to bromine, and BB4 are supposed to be trans
    df_halides.loc[df_halides['smiles'] == 'C/C(Br)=C/C(C)C(C)O', 'bb'] = 'BB4'

    df_halides.to_pickle('data/unique_halides_reclassed.pickle')


def canonicalize_some_smiles():
    indices = [69417, 69418, 69419, 74138, 75652, 75657, 75678, 75791, 75792, 75793, 75943, 75976, 76144, 76145, 76217, 76218,
     76219, 76644, 76699, 76700, 76701, 76702, 76703, 76704, 76705, 76706, 76707, 76708, 76709, 76710, 76711, 76712,
     76713, 76714, 76715, 77030, 77045, 77046, 77047, 77048, 77049, 77050, 77051, 77052, 77588, 78919, 79379, 79913,
     83840, 83841, 83868, 83869, 83870, 86125, 86126, 86150, 86475, 88747, 89298, 89299, 89300, 89301, 89302, 89303,
     89304, 89305, 89306, 89307, 89308, 89309, 91345, 95089, 107528, 108495, 120000, 120059, 126100, 126101, 136245,
     136946, 136975, 137138, 140217, 140218, 140219, 140234, 141695, 141696, 142165, 142721, 158536, 158537, 158538,
     158539, 158540, 158541, 162785, 164013, 167332, 178953, 178954, 178955, 192989, 193751, 193830, 299358]
    df = pd.read_pickle('data/DNP_FULL_2016_with_polyketides_fingerprints_fixed.pickle')
    for i, row in df.iterrows():
        if i in indices:
            df.loc[i, 'smiles'] = Chem.CanonSmiles(row['smiles'])
    df.to_pickle('data/DNP_FULL_2016_with_polyketides_fingerprints_fixed2.pickle')


if __name__ == '__main__':
    # canonicalize_some_smiles()
    # reclassify_BB4()

    ################# After adding polyketideome manually selected by Antonio
    # load_many_excel_with_molecules_into_df(source_folder='data/DNP All Polyketides', target_df_filename='data/manual_polyketides.pickle')
    # update_DNP_with_manual_polyketides(output_file='data/DNP_FULL_2016_with_manpk.pickle')
    # enrich_DNP_with_synthesizeable(parent_df_all_filename='data/DNP_FULL_2016_with_manpk.pickle',
    #                                output_file='data/DNP_FULL_2016_with_polyketides_not_all_cols.pickle')
    # server_df_to_merged_df(filepath='data/DNP_FULL_2016_with_polyketides_not_all_cols.pickle',
    #                        polyketides_df_pickle_filepath='data/polyketides_bitvect_fp.pickle',
    #                        output_filename='data/DNP_FULL_2016_with_polyketides.hdf')
    # decimate to 100 k
    # decimate_DNP_v2(source_df_filename='data/DNP_FULL_2016_with_polyketides.hdf', target_size=100000)
    # compute_fingerprints_for_df(db_filepath='data/DNP_FULL_2016_with_polyketides_len100k.pickle',
    #                             output_filepath='data/DNP_FULL_2016_with_polyketides_fingerprints_len100k.pickle',
    #                             force_recalculate=True)


    #################### Loading all the groups into the DB
    # load_many_excel_with_molecules_into_df(source_folder='data/other_groups/DNP all steroids', target_df_filename='data/manual_steroids.pickle')
    # load_many_excel_with_molecules_into_df(source_folder='data/other_groups/DNP all flavonoids',
    #                                        target_df_filename='data/manual_flavonoids.pickle')
    # load_many_excel_with_molecules_into_df(source_folder='data/other_groups/DNP All Alkaloids',
    #                                        target_df_filename='data/manual_alkaloids.pickle')
    # load_many_excel_with_molecules_into_df(source_folder='data/other_groups/DNP all terpenes',
    #                                        target_df_filename='data/manual_terpenes.pickle')

    # df_left = pd.read_pickle('data/DNP_FULL_2016_with_manpk.pickle')
    # fp_colname_bv = 'fp_ECFP6_bv'
    # if fp_colname_bv not in df_left.columns:
    #     logging.info(f'Calculating bitvect fingerprints for df_left...')
    #     df_left[fp_colname_bv] = df_left['smiles'].apply(fingerprint_bitvect)
    # df_left.to_pickle('data/DNP_FULL_2016_with_manpk.pickle')

    # # merge the manually selected sets into the master DB
    # df_updated = update_DNP_with_manual_molecules(df_left=pd.read_pickle('data/DNP_FULL_2016_with_manpk.pickle'),
    #                                                   df_right=pd.read_pickle('data/manual_steroids.pickle'),
    #                                                   colname_for_flag='is_steroid',
    #                                                   force_calculate_fps=True)
    # df_updated.to_pickle('data/DNP_FULL_2016_with_steroids.pickle')
    #
    # df_updated = update_DNP_with_manual_molecules(df_left=pd.read_pickle('data/DNP_FULL_2016_with_steroids.pickle'),
    #                                                   df_right=pd.read_pickle('data/manual_flavonoids.pickle'),
    #                                                   colname_for_flag='is_flavonoid',
    #                                                   force_calculate_fps=True)
    # df_updated.to_pickle('data/DNP_FULL_2016_with_pk_and_steroids_and_flavonoids.pickle')
    #
    # df_updated = update_DNP_with_manual_molecules(df_left=pd.read_pickle('data/DNP_FULL_2016_with_pk_and_steroids_and_flavonoids.pickle'),
    #                                                   df_right=pd.read_pickle('data/manual_alkaloids.pickle'),
    #                                                   colname_for_flag='is_alkaloid',
    #                                                   force_calculate_fps=True)
    # df_updated.to_pickle('data/DNP_FULL_2016_with_pk_and_steroids_and_flavonoids_and_alkaloids.pickle')
    #
    # df_updated = update_DNP_with_manual_molecules(df_left=pd.read_pickle('data/DNP_FULL_2016_with_pk_and_steroids_and_flavonoids_and_alkaloids.pickle'),
    #                                                   df_right=pd.read_pickle('data/manual_terpenes.pickle'),
    #                                                   colname_for_flag='is_terpene',
    #                                                   force_calculate_fps=True)
    # df_updated.to_pickle('data/DNP_FULL_2016_with_pk_and_steroids_and_flavonoids_and_alkaloids_and_terpenes.pickle')
    #
    # # If this is done on the server, fix the pickle protocol compatibiolity issues
    # df = pd.read_pickle('data/DNP_FULL_2016_with_pk_and_steroids_and_flavonoids_and_alkaloids_and_terpenes.pickle')
    # # drop columns fp_ECFP6 and fp_ECFP6_bv
    # df.drop(columns=['fp_ECFP6', 'fp_ECFP6_bv'], inplace=True)
    # # save to hdf
    # df.to_hdf('data/DNP_FULL_2016_with_pk_and_steroids_and_flavonoids_and_alkaloids_and_terpenes.hdf', key='df', mode='w')

    # fp_colname_bv = 'fp_ECFP6_bv'
    # fp_colname = 'fp_ECFP6'
    # df = pd.read_hdf('data/DNP_FULL_2016_with_pk_and_steroids_and_flavonoids_and_alkaloids_and_terpenes.hdf', key='df')
    # df[fp_colname_bv] = df['smiles'].apply(fingerprint_bitvect)
    # df[fp_colname] = df['smiles'].apply(fingerprint)
    # df.to_pickle('data/DNP_FULL_2016_with_pk_and_steroids_and_flavonoids_and_alkaloids_and_terpenes.pickle')

    # enrich_DNP_with_synthesizeable(parent_df_all_filename='data/DNP_FULL_2016_with_pk_and_steroids_and_flavonoids_and_alkaloids_and_terpenes.pickle',
    #                                output_file='data/DNP_FULL_2016_with_polyketides_not_all_cols.pickle')
    # server_df_to_merged_df(filepath='data/DNP_FULL_2016_with_polyketides_not_all_cols.pickle',
    #                        polyketides_df_pickle_filepath='data/polyketides_bitvect_fp.pickle',
    #                        output_filename='data/DNP_FULL_2016_with_polyketides.hdf')
    # compute_fingerprints_for_df(db_filepath='data/DNP_FULL_2016_with_polyketides.hdf',
    #                             output_filepath='data/DNP_FULL_2016_with_polyketides_fingerprints_notfixed.pickle',
    #                             force_recalculate=True)
    # fix_column_types(input_filename='data/DNP_FULL_2016_with_polyketides_fingerprints.pickle',
    #                  output_filename='data/DNP_FULL_2016_with_polyketides_fingerprints_fixed.pickle')

    # drop rows where fp is None
    # df = pd.read_pickle('data/DNP_FULL_2016_with_polyketides.pickle')
    # df = df[df['fp_ECFP6'].notnull()]
    # df.to_pickle('data/DNP_FULL_2016_with_polyketides.pickle')

    # ### decimate to 100 k
    # decimate_DNP_v2(source_df_filename='data/DNP_FULL_2016_with_polyketides.hdf', target_size=100000)
    # compute_fingerprints_for_df(db_filepath='data/DNP_FULL_2016_with_polyketides_len100k.pickle',
    #                             output_filepath='data/DNP_FULL_2016_with_polyketides_fingerprints_len100k.pickle',
    #                             force_recalculate=True)
    # fix_column_types()



    # compute_fingerprints_for_df(db_filepath='data/DNP_FULL_2016_with_polyketides.pickle',
    #                             fp_colname='fp_ECFP6')
    # compute_polyketides_fingerprints()
    # compute_tsne_with_callable_metric_and_save(db_filepath='data/polyketides_bitvect_fp.pickle',
    #                                            force_recalculate_distances=False, fp_colname='fp_ECFP6')

    # compute_fingerprints_for_df(db_filepath='data/DNP_FULL_2016_with_polyketides.pickle',
    #                             force_recalculate=True)

    # df = pd.read_pickle('data/DNP_FULL_2016_with_polyketides_fingerprints.pickle')
    # # get rows where fp_ECFP6_bv is a list
    # df = df[df['fp_ECFP6_bv'].apply(lambda x: not isinstance(x, DataStructs.cDataStructs.ExplicitBitVect))]

    # fp_colname = 'fp_ECFP6_bv'
    # df = pd.read_pickle('data/polyketides_bitvect_fp.pickle')
    # dm = distance_matrix(df, f'data/polyketides.parquet_distance_matrix.npy', fp_colname,
    #                      force_recalculate=True)


    #
    # dm = np.load('data/polyketides.parquet_distance_matrix.npy')

    # df = pd.read_pickle('data/polyketides_bitvect_fp.pickle')
    # dm2 = distance_matrix(df, cache_filename=f'data/polyketides_bitvect_fp_distance_matrix.npy', force_recalculate=True,
    #                       algo='loop')
    # dm3 = distance_matrix(df, cache_filename=f'data/polyketides_bitvect_fp_distance_matrix.npy', force_recalculate=True,
    #                       algo='two_loops')

    # df = pd.read_pickle('data/DNP_FULL_2016_with_polyketides_fingerprints.pickle')
    # distance_matrix(df, cache_filename=f'data/DNP_FULL_2016_with_polyketides_fingerprints_distance_matrix.npy',
    #                 force_recalculate=True, algo='loop')

    # compute_tsne_with_distmatrix_and_save(db_filepath='data/polyketides_bitvect_fp.pickle',
    #                                       distmatrix_cache_filename='data/polyketides_bitvect_fp_distance_matrix.npy',
    #                                       output_filename='data/polyketides_bitvect_fp_tsne.pickle',
    #                                       limit_rows=1000, memmap_mode='r')

    # df = pd.read_pickle('data/polyketides_bitvect_fp_tsne.pickle')

    ####### # only the polyketides
    # df = pd.read_pickle('data/DNP_FULL_2016_with_polyketides_fingerprints_len100k.pickle')
    # df = df[(df['is_pk'] == 1) | (df['synthesizable'] == 1)]
    # df = df.reset_index(drop=True)
    # df.to_pickle('data/only_polyketides_fingerprints_len100k.pickle')


    # ####### alpha-methyl-beta-hydroxy
    # df = pd.read_pickle('data/DNP_FULL_2016_with_polyketides_fingerprints_len100k.pickle')
    # # mark alpha-methyl-beta-hydroxy
    # specific_molecular_pattern_SMARTS = '[CH3]-[CH](-[*])-[CH](-[*])-O'
    #
    # specific_molecular_pattern = Chem.MolFromSmarts(specific_molecular_pattern_SMARTS)
    # img = Draw.MolToImage(specific_molecular_pattern)
    # plt.imshow(img)
    # plt.show()
    # # iterate over df with tqdm
    # for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    #     mol = Chem.MolFromSmiles(row['smiles'])
    #     # if mol.HasSubstructMatch(specific_molecular_pattern):
    #     if len(mol.GetSubstructMatches(specific_molecular_pattern)) > 1:
    #         # mark in 'polyketideome' column as 1
    #         df.loc[i, 'a-methyl-b-hydroxyl'] = 1
    #     else:
    #         df.loc[i, 'a-methyl-b-hydroxyl'] = 0
    #
    # print('Number of polyketides in the dataset: ', df['a-methyl-b-hydroxyl'].sum())
    # df = df[(df['is_pk'] == 1) & (df['a-methyl-b-hydroxyl'] == 1)]
    # df = df.reset_index(drop=True)
    # df.to_pickle('data/only_ab_polyketides_fingerprints.pickle')

    # get_unique_halides()

    # df_halides = dft_npa_halides()

    ####################### Server #######################
    # compute_tsne_with_distmatrix_and_save(db_filepath='data/DNP_FULL_2016_with_polyketides_fingerprints.pickle',
    #                                       distmatrix_cache_filename='data/DNP_FULL_2016_with_polyketides_fingerprints_distance_matrix.npy',
    #                                       output_filename='data/DNP_FULL_2016_with_polyketides_fingerprints_tsne_px300.hdf',
    #                                       memmap_mode='r')
    #
    # compute_tsne_with_distmatrix_and_save(db_filepath='data/DNP_FULL_2016_with_polyketides_fingerprints.pickle',
    #                                       distmatrix_cache_filename='data/DNP_FULL_2016_with_polyketides_fingerprints_distance_matrix.npy',
    #                                       output_filename='data/DNP_FULL_2016_with_polyketides_fingerprints_tsne_px3000.hdf',
    #                                       memmap_mode='r',
    #                                       tsne_params={'n_components': 2, 'verbose': 2, 'random_state': 137,
    #                                                    'metric': 'precomputed',
    #                                                    'early_exaggeration': 1, 'perplexity': 3000, 'init': 'random',
    #                                                    'n_iter': 5000, 'learning_rate':1000,
    #                                                    'n_iter_without_progress': 500, 'method': 'barnes_hut',
    #                                                    'n_jobs': 1}
    #                                       )
    #
    # compute_tsne_with_distmatrix_and_save(db_filepath='data/DNP_FULL_2016_with_polyketides_fingerprints.pickle',
    #                                       distmatrix_cache_filename='data/DNP_FULL_2016_with_polyketides_fingerprints_distance_matrix.npy',
    #                                       output_filename='data/DNP_FULL_2016_with_polyketides_fingerprints_tsne_px30000.hdf',
    #                                       memmap_mode='r',
    #                                       tsne_params={'n_components': 2, 'verbose': 2, 'random_state': 137,
    #                                                    'metric': 'precomputed',
    #                                                    'early_exaggeration': 1, 'perplexity': 30000, 'init': 'random',
    #                                                    'n_iter': 5000, 'learning_rate':1000,
    #                                                    'n_iter_without_progress': 500, 'method': 'barnes_hut',
    #                                                    'n_jobs': 1}
    #                                       )


    # make_tsne_on_polyketides(filepath_to_df='data/polyketides_bitvect_fp_tsne.pickle')

    # assert_that_select_polyketides_contain_intended_pattern()
    # convert_full_DNP_to_dataframe_file()
    # compute_tsne()

    #### plotting
    # filepath_to_df = 'data/DNP_FULL_2016_with_polyketides_fingerprints_tsne_px30.pickle'
    # # filepath_to_df = 'data/polyketides_bitvect_fp.pickle'
    # df = pd.read_pickle(filepath_to_df)


    # server_df_to_merged_df(filepath='data/DNP_FULL_2016_with_polyketides_fingerprints_tsne_px300_lr70000_fixed.hdf',
    #                        polyketides_df_pickle_filepath='data/polyketides_bitvect_fp.pickle')
    #
    # # get all filenames in /data that end with .hdf and have len100k_tsne in their name
    # for filename in [f for f in os.listdir('data') if f.endswith('.hdf') and
    #                                                   (not f.endswith('pkmerged.hdf')) and
    #                                                   (not f.endswith('selected.hdf')) and
    #                                                   'len100k_tsne' in f]:
    #     server_df_to_merged_df(filepath=f'data/{filename}',
    #                            polyketides_df_pickle_filepath='data/polyketides_bitvect_fp.pickle')


    # # ############### Plot all tsne embeddings
    # plot_tsne_from_hdf(filepath='data/DNP_FULL_2016_with_polyketides_fingerprints_tsne_px300_lr70000_fixed_pkmerged.hdf',
    #                    title='Perplexity: 300', do_show=False)
    #
    # plot_tsne_from_hdf(filepath='data/DNP_FULL_2016_with_polyketides_fingerprints_tsne_px30_pkmerged.hdf',
    #                    title='Perplexity: 30', do_show=False)
    # # plot_tsne_from_hdf(
    # #     filepath='data/DNP_FULL_2016_with_polyketides_fingerprints_len100k_tsne_px3534_lr24735_pkmerged.hdf')
    #
    # for filename in [f for f in os.listdir('data') if f.endswith('pkmerged.hdf') and 'len100k_tsne' in f]:
    #     logging.info(f'Plotting {filename}')
    #     equivalent_perplexity = int(round(int(filename.split("_")[8][2:]) * 283 / 1000))*10
    #     plot_tsne_from_hdf(filepath=f'data/{filename}', title=f'Perplexity: {equivalent_perplexity}',
    #                        do_show=False)

    ############ play with one embedding
    # filename = 'DNP_FULL_2016_with_polyketides_fingerprints_len100k_tsne_px1060_lr24735_pkmerged.hdf'
    # equivalent_perplexity = int(round(int(filename.split("_")[8][2:]) * 283 / 1000)) * 10
    # plot_tsne_from_hdf(filepath=f'data/{filename}', title=f'Perplexity: {equivalent_perplexity}',
    #                    do_show=True)

    # draw_smarts_found_in_many_smiles(pickle_file='data/DNP_FULL_2016_with_polyketides_fingerprints.pickle',
    #                             pattern_SMARTS='C-C(~O)-C-C(~O)-C'
    # )
    # pattern_SMARTS = '*-C(~O)-C'

    # draw_smarts_found_in_smiles(SMILES='C(C(C=C)O[H])=C',
    #                             pattern_SMARTS='*-C(~O)-C',
    #                             folder_for_output_images="figures/molecule_images/")
    #
    # draw_smarts_found_in_smiles(SMILES='C(C(C=C)=O)=C',
    #                             pattern_SMARTS='*-C(~O)-C',
    #                             folder_for_output_images="figures/molecule_images/")

    # draw_smarts_found_in_smiles(SMILES='[C](=[C](C=C)O[H])=[C]',
    #                             pattern_SMARTS='*-C(~O)-C',
    #                             folder_for_output_images="figures/molecule_images/")

    # server_df_to_merged_df(filepath='data/DNP_FULL_2016_with_polyketides_map4fingerprints_len100k_tsne_px353_lr24735.hdf',
    #                        polyketides_df_pickle_filepath='data/polyketides_bitvect_fp.pickle')
    # filename = 'DNP_FULL_2016_with_polyketides_map4fingerprints_len100k_tsne_px353_lr24735_pkmerged.hdf'
    # equivalent_perplexity = int(round(int(filename.split("_")[8][2:]) * 283 / 1000))*10
    # plot_tsne_from_hdf(filepath=f'data/{filename}', title=f'Perplexity: {equivalent_perplexity}',
    #                    do_show=True)

    # server_df_to_merged_df(filepath='data/DNP_FULL_2016_with_polyketides_map4fingerprints_len100k_tsne_px1060_lr24735.hdf',
    #                        polyketides_df_pickle_filepath='data/polyketides_bitvect_fp.pickle')
    # filename = 'DNP_FULL_2016_with_polyketides_map4fingerprints_len100k_tsne_px1060_lr24735_pkmerged.hdf'
    # equivalent_perplexity = int(round(int(filename.split("_")[8][2:]) * 283 / 1000)) * 10
    # plot_tsne_from_hdf(filepath=f'data/{filename}', title=f'Perplexity: {equivalent_perplexity}',
    #                    do_show=True)

    # filename='DNP_FULL_2016_with_polyketides_fingerprints_len100k_tsne_px1060_lr24735_5kiter.hdf'
    plot_tsne_from_hdf(hdf_filepath=f'data/DNP_FULL_2016_with_polyketides_fingerprints_fixed2_map4fingerprints_umap_nn30.hdf',
                       title=f'UMAP',
                       do_show=True, size=30, mainalpha=1)
    pass