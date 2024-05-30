from halides_selection import *
import umap
from sklearnex import patch_sklearn
patch_sklearn()

def compute_umap_with_distmatrix_and_save(db_filepath, distmatrix_cache_filename='auto', output_filename='auto',
                                          fp_colname = 'fp_ECFP6_bv', memmap_mode=None, umap_params=None,
                                          limit_rows=None, change_dtypes_to_str=True):
    # extract the filepath without extension
    db_filepath_without_extension = db_filepath.split('.')[0]
    if distmatrix_cache_filename == 'auto':
        distmatrix_cache_filename = db_filepath_without_extension + '_distance_matrix.npy'
    if umap_params is None:  # use default parameters
        umap_params = {'n_neighbors':15, 'n_components':2, 'metric':'precomputed'}
    if output_filename == 'auto':
        output_filename = db_filepath_without_extension + '_tsne.hdf'
    df = pd.read_pickle(db_filepath)
    if limit_rows is not None:
        df = df.iloc[:limit_rows]
    dist_matrix = distance_matrix(df, cache_filename=distmatrix_cache_filename, fp_colname=fp_colname,
                                  memmap_mode=memmap_mode)
    if limit_rows is not None:
        dist_matrix = dist_matrix[:limit_rows, :limit_rows]
    logging.info(f'Loaded distance matrix from {distmatrix_cache_filename}')


    reducer = umap.UMAP(**umap_params)
    z = reducer.fit_transform(dist_matrix)
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
    pickle.HIGHEST_PROTOCOL = 4
    df.to_hdf(output_filename, 'df')

    logging.info(f'Saved umap embedding to {output_filename}')


# Calculate distance matrix
df = pd.read_pickle('data/ligands_coev_plus_kraken.pickle')
distance_matrix(df, cache_filename=f'data/ligands_coev_plus_kraken_distance_matrix.npy',
                force_recalculate=True, algo='loop', fp_colname='fp_ECFP6_bv')


################## UMAP WITH MAP4
db_filepath = 'data/ligands_coev_plus_kraken.pickle'
umap_params = {'n_neighbors': 1000, 'n_components': 2, 'metric': 'precomputed', 'verbose': True, 'min_dist': 0.1,}

for n_neighbours in [3, 10, 30, 100]:
    for min_dist in [0.01, 0.05, 0.1, 0.2, 0.5, 1]:
        umap_params['n_neighbors'] = n_neighbours
        umap_params['min_dist'] = min_dist
        logging.info(f'UMAP with n_neighbors={n_neighbours}')
        filepath_without_extension = db_filepath.split('.')[0]
        distmatrix_cache_filename = f'data/ligands_coev_plus_kraken_distance_matrix.npy'
        compute_umap_with_distmatrix_and_save(db_filepath=db_filepath,
                                              distmatrix_cache_filename=distmatrix_cache_filename,
                                              output_filename=f'{filepath_without_extension}_umap_nn{umap_params["n_neighbors"]}_md{umap_params["min_dist"]}.hdf',
                                              memmap_mode='r', umap_params=umap_params,
                                              change_dtypes_to_str=False)