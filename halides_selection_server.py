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

    logging.info(f'Saved tsne embedding to {output_filename}')

#### OPERATIONS AFTER CUSTOM POLYKETIDES ADDED MANUALLY BY ANTONIO

######### FOR ALL DNP

######### nondecimated_dataset

# ECFP 4
# df = pd.read_pickle('data/DNP_FULL_2016_with_polyketides_fingerprints_fixed.pickle')
# distance_matrix(df, cache_filename=f'data/DNP_FULL_2016_with_polyketides_fingerprints_fixed_distance_matrix.npy',
#                 force_recalculate=True, algo='loop', fp_colname='fp_ECFP6_bv')

# MAP4
# compute_fingerprints_for_df(db_filepath='data/DNP_FULL_2016_with_polyketides_fingerprints_fixed2.pickle',
#                             force_recalculate=True, fp_colname='fp_MAP4', N_PROC=70)

# df = pd.read_pickle('data/DNP_FULL_2016_with_polyketides_fingerprints_fixed2_map4fingerprints.pickle')
# distmat = distance_matrix(df, cache_filename=f'data/DNP_FULL_2016_with_polyketides_fingerprints_fixed_map4fingerprints_distance_matrix.npy',
#                 force_recalculate=True, algo='loop', fp_colname='fp_MAP4_bv')
# print(distmat.shape)

# tsne_params = {'n_components': 2, 'verbose': 2, 'random_state': 137,
#                'metric': 'precomputed',
#                'early_exaggeration': 1, 'perplexity': 300, 'init': 'random',
#                'n_iter': 5000, 'learning_rate': 70000,
#                'n_iter_without_progress': 10000, 'method': 'barnes_hut',
#                'n_jobs': 1}
#
# for perplexity in [3000]:
#     for learning_rate in [70000]:
#         logging.info(f'learning_rate={learning_rate}, perplexity={perplexity}')
#         db_filepath = 'data/DNP_FULL_2016_with_polyketides_fingerprints_fixed2_map4fingerprints.pickle'
#         tsne_params['learning_rate'] = int(round(learning_rate / 283 * 300))
#         tsne_params['perplexity'] = int(round(perplexity / 283 * 300))
#         filepath_without_extension = db_filepath.split('.')[0]
#         distmatrix_cache_filename = f'data/DNP_FULL_2016_with_polyketides_fingerprints_fixed_map4fingerprints_distance_matrix.npy'
#         compute_tsne_with_distmatrix_and_save(db_filepath=db_filepath,
#                                               distmatrix_cache_filename=distmatrix_cache_filename,
#                                               output_filename=f'{filepath_without_extension}_tsne_px{tsne_params["perplexity"]}_lr{tsne_params["learning_rate"]}.hdf',
#                                               memmap_mode='r', tsne_params=tsne_params)

# ################## UMAP WITH MAP4
# db_filepath = 'data/DNP_FULL_2016_with_polyketides_fingerprints_fixed2_map4fingerprints.pickle'
# umap_params = {'n_neighbors': 1000, 'n_components': 2, 'metric': 'precomputed', 'verbose': True, 'min_dist': 0.1,}
#
# for n_neighbours in [1000]:
#     for min_dist in [0.01, 0.05, 0.2, 0.5, 1]:
#         umap_params['n_neighbors'] = n_neighbours
#         umap_params['min_dist'] = min_dist
#         logging.info(f'UMAP with n_neighbors={n_neighbours}')
#         filepath_without_extension = db_filepath.split('.')[0]
#         distmatrix_cache_filename = f'data/DNP_FULL_2016_with_polyketides_fingerprints_fixed_map4fingerprints_distance_matrix.npy'
#         compute_umap_with_distmatrix_and_save(db_filepath=db_filepath,
#                                               distmatrix_cache_filename=distmatrix_cache_filename,
#                                               output_filename=f'{filepath_without_extension}_umap_nn{umap_params["n_neighbors"]}_md{umap_params["min_dist"]}.hdf',
#                                               memmap_mode='r', umap_params=umap_params,
#                                               change_dtypes_to_str=False)


df = pd.read_pickle('data/DNP_FULL_2016_with_polyketides_fingerprints_len100k.pickle')
distance_matrix(df, cache_filename=f'data/DNP_FULL_2016_with_polyketides_fingerprints_len100k_distance_matrix.npy',
                force_recalculate=True, algo='loop', fp_colname='fp_ECFP6_bv')
tsne_params = {'n_components': 2, 'verbose': 2, 'random_state': 138,
               'metric': 'precomputed',
               'early_exaggeration': 1, 'perplexity': 300, 'init': 'random',
               'n_iter': 5000, 'learning_rate': 1000,
               'n_iter_without_progress': 1000, 'method': 'barnes_hut',
               'n_jobs': 1}

for perplexity in [1000, 3000]:#, [1000, 3000, 10000, 30000, 100000, 300]:
    for learning_rate in [70000]:
        logging.info(f'learning_rate={learning_rate}, perplexity={perplexity}')
        db_filepath = 'data/DNP_FULL_2016_with_polyketides_fingerprints_len100k.pickle'
        tsne_params['learning_rate'] = int(round(learning_rate / 283 * 100))
        tsne_params['perplexity'] = int(round(perplexity / 283 * 100))
        filepath_without_extension = db_filepath.split('.')[0]
        distmatrix_cache_filename = f'{filepath_without_extension}_distance_matrix.npy'
        compute_tsne_with_distmatrix_and_save(db_filepath=db_filepath,
                                              distmatrix_cache_filename=distmatrix_cache_filename,
                                              output_filename=f'{filepath_without_extension}_tsne_px{tsne_params["perplexity"]}_lr{tsne_params["learning_rate"]}_'
                                                              f'{int(round(tsne_params["n_iter"]/1000))}kiter.hdf',
                                              memmap_mode='r', tsne_params=tsne_params)

######### FOR ONLY PK
# df_filename = 'data/only_polyketides_fingerprints_len100k.pickle'
# df = pd.read_pickle(df_filename)
# distance_matrix(df, cache_filename=f'data/only_polyketides_fingerprints_len100k_distance_matrix.npy',
#                 force_recalculate=True, algo='loop', fp_colname='fp_ECFP6_bv')
#
# tsne_params = {'n_components': 2, 'verbose': 2, 'random_state': 137,
#                'metric': 'precomputed',
#                'early_exaggeration': 1, 'perplexity': 300, 'init': 'random',
#                'n_iter': 50000, 'learning_rate': 1000,
#                'n_iter_without_progress': 1000, 'method': 'barnes_hut',
#                'n_jobs': 1}
#
# for perplexity in [1000, 3000]:#, [1000, 3000, 10000, 30000, 100000, 300]:
#     for learning_rate in [70000]:
#         logging.info(f'learning_rate={learning_rate}, perplexity={perplexity}')
#         db_filepath = df_filename
#         tsne_params['learning_rate'] = int(round(learning_rate / 283 * 7.5))
#         tsne_params['perplexity'] = int(round(perplexity / 283 * 7.5))
#         filepath_without_extension = db_filepath.split('.')[0]
#         distmatrix_cache_filename = f'{filepath_without_extension}_distance_matrix.npy'
#         compute_tsne_with_distmatrix_and_save(db_filepath=db_filepath,
#                                               distmatrix_cache_filename=distmatrix_cache_filename,
#                                               output_filename=f'{filepath_without_extension}_tsne_px{tsne_params["perplexity"]}_lr{tsne_params["learning_rate"]}_'
#                                                               f'{int(round(tsne_params["n_iter"]/1000))}kiter.hdf',
#                                               memmap_mode='r', tsne_params=tsne_params)


# ##### for only PK that have a-methyl-b-hydroxy
# df_filename = 'data/only_ab_polyketides_fingerprints.pickle'
# df = pd.read_pickle(df_filename)
# distance_matrix(df, cache_filename=f'data/only_ab_polyketides_fingerprints_distance_matrix.npy',
#                 force_recalculate=True, algo='loop', fp_colname='fp_ECFP6_bv')
#
# tsne_params = {'n_components': 2, 'verbose': 2, 'random_state': 137,
#                'metric': 'precomputed',
#                'early_exaggeration': 1, 'perplexity': 300, 'init': 'random',
#                'n_iter': 50000, 'learning_rate': 1000,
#                'n_iter_without_progress': 10000, 'method': 'barnes_hut',
#                'n_jobs': 1}
#
# for perplexity in [3000]:#, [1000, 3000, 10000, 30000, 100000, 300]:
#     for learning_rate in [70000]:
#         logging.info(f'learning_rate={learning_rate}, perplexity={perplexity}')
#         db_filepath = df_filename
#         tsne_params['learning_rate'] = int(round(learning_rate / 283 * 2.86))
#         tsne_params['perplexity'] = int(round(perplexity / 283 * 2.86))
#         filepath_without_extension = db_filepath.split('.')[0]
#         distmatrix_cache_filename = f'{filepath_without_extension}_distance_matrix.npy'
#         compute_tsne_with_distmatrix_and_save(db_filepath=db_filepath,
#                                               distmatrix_cache_filename=distmatrix_cache_filename,
#                                               output_filename=f'{filepath_without_extension}_tsne_px{tsne_params["perplexity"]}_lr{tsne_params["learning_rate"]}_'
#                                                               f'{int(round(tsne_params["n_iter"]/1000))}kiter.hdf',
#                                               memmap_mode='r', tsne_params=tsne_params)


#### OPERATIONS BEFORE CUSTOM POLYKETIDES ADDED MANUALLY BY ANTONIO

###### ECFP6

# distance_matrix(df, cache_filename=f'data/DNP_FULL_2016_with_polyketides_map4fingerprints_distance_matrix.npy',
#                 force_recalculate=True, algo='two_loops', fp_colname='fp_MAP4_bv')

# df = pd.read_pickle('data/DNP_FULL_2016_with_polyketides_fingerprints_len100k.pickle')
# distance_matrix(df, cache_filename=f'data/DNP_FULL_2016_with_polyketides_fingerprints_len100k_distance_matrix.npy',
#                 force_recalculate=True, algo='loop')
#
# df = pd.read_pickle('data/DNP_FULL_2016_with_polyketides_fingerprints_len30k.pickle')
# distance_matrix(df, cache_filename=f'data/DNP_FULL_2016_with_polyketides_fingerprints_len30k_distance_matrix.npy',
#                 force_recalculate=True, algo='loop')

# tsne_params = {'n_components': 2, 'verbose': 2, 'random_state': 137,
#                'metric': 'precomputed',
#                'early_exaggeration': 1, 'perplexity': 300, 'init': 'random',
#                'n_iter': 50000, 'learning_rate': 1000,
#                'n_iter_without_progress': 10000, 'method': 'barnes_hut',
#                'n_jobs': 1}
# #
# # # # for perplexity in [3000, 30000]:
# # # # for perplexity in [10000, 100000]:
# for perplexity in [3000]:#, [1000, 3000, 10000, 30000, 100000, 300]:
#     for learning_rate in [70000]:
#         logging.info(f'learning_rate={learning_rate}, perplexity={perplexity}')
#         if perplexity < 400:
#             db_filepath = 'data/DNP_FULL_2016_with_polyketides_map4fingerprints.pickle'
#             tsne_params['perplexity'] = perplexity
#             tsne_params['learning_rate'] = learning_rate
#         else:
#             db_filepath = 'data/DNP_FULL_2016_with_polyketides_map4fingerprints_len100k.pickle'
#             tsne_params['learning_rate'] = int(round(learning_rate / 283 * 100))
#             tsne_params['perplexity'] = int(round(perplexity / 283 * 100))
#         filepath_without_extension = db_filepath.split('.')[0]
#         distmatrix_cache_filename = f'{filepath_without_extension}_distance_matrix.npy'
#         compute_tsne_with_distmatrix_and_save(db_filepath=db_filepath,
#                                               distmatrix_cache_filename=distmatrix_cache_filename,
#                                               output_filename=f'{filepath_without_extension}_tsne_px{tsne_params["perplexity"]}_lr{tsne_params["learning_rate"]}.hdf',
#                                               memmap_mode='r', tsne_params=tsne_params)




### MAP4

# compute_fingerprints_for_df(db_filepath='data/DNP_FULL_2016_with_polyketides.pickle',
#                             force_recalculate=True, fp_colname='fp_MAP4')
#
# df = pd.read_pickle('data/DNP_FULL_2016_with_polyketides_map4fingerprints.pickle')
# distmat = distance_matrix(df, cache_filename=f'data/DNP_FULL_2016_with_polyketides_map4fingerprints_distance_matrix.npy',
#                 force_recalculate=True, algo='loop', fp_colname='fp_MAP4_bv')
# print(distmat.shape)

# compute_fingerprints_for_df(db_filepath='data/DNP_FULL_2016_with_polyketides_fingerprints_len100k.pickle',
#                             force_recalculate=True, fp_colname='fp_MAP4',
#                             output_filepath='data/DNP_FULL_2016_with_polyketides_map4fingerprints_len100k.pickle')
#
# df = pd.read_pickle('data/DNP_FULL_2016_with_polyketides_map4fingerprints_len100k.pickle')
# distmat = distance_matrix(df, cache_filename=f'data/DNP_FULL_2016_with_polyketides_map4fingerprints_len100k_distance_matrix.npy',
#                 force_recalculate=True, algo='loop', fp_colname='fp_MAP4_bv')
# print(distmat.shape)