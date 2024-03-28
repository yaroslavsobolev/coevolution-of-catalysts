from sklearn.metrics import silhouette_score, calinski_harabasz_score

from halides_selection import *
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.cluster import DBSCAN, SpectralClustering
import skunk
from matplotlib.offsetbox import AnnotationBbox
from rdkit.Chem.Draw import rdMolDraw2D
from cairosvg import svg2png

def find_id_of_bromine(molecule):
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == 'Br':
            return atom.GetIdx()
    raise ValueError('No bromine found in molecule')


def similarity_from_bromine(molecule1, molecule2, radius=2):
    fp1 = AllChem.GetMorganFingerprintAsBitVect(molecule1, radius, useChirality=True, fromAtoms=[find_id_of_bromine(molecule1)])
    fp2 = AllChem.GetMorganFingerprintAsBitVect(molecule2, radius, useChirality=True, fromAtoms=[find_id_of_bromine(molecule2)])
    return tanisim(fp1, fp2)


def exp_similarity_from_bromine(molecule1, molecule2, decay_radius=2, radius_range=range(2, 8)):
    overall_similarity = 0
    for radius in radius_range:
        overall_similarity += similarity_from_bromine(molecule1, molecule2, radius) * np.exp(-radius/decay_radius)
    overall_similarity /= np.sum(np.exp(-1*np.array(radius_range)/decay_radius))
    return overall_similarity


def smiles_to_svg(input_smiles: str, svg_file_name:str,  size=(400, 200)):

    molecule= Chem.MolFromSmiles(input_smiles)
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:','')
    with open(svg_file_name, 'w') as f:
        f.write(svg)
    return


representative_smiles = [
    'Br/C=C/C',
    'Br/C=C\C',
    'Br/C=C(C)/C',
    'Br/C=C(C)/C',
    'Br/C=C/C1=CC=CC=C1',
    'Br/C=C(C(OC(C)(C)C)=O)\C',
    'Br/C=C(C(N(C)C)=O)\C',
    'Br/C=C(C)/C(C)=O',
    'Br/C=C/C(OC(C)(C)C)=O',
    'Br/C=C/C(N(C)C)=O',
    'Br/C=C/C(C)=O',
    'C=C(C)Br']
representative_smiles = [Chem.CanonSmiles(smiles) for smiles in representative_smiles]

def get_k_medoids(distmatrix, number_of_medoids):
    # Initialize medoids randomly
    initial_medoids = np.random.choice(distmatrix.shape[0], number_of_medoids, replace=False).tolist()
    print('initial randomized %i medoids: ' % number_of_medoids)
    print(initial_medoids)

    # Apply k-medoid clustering
    kmedoids_instance = kmedoids(distmatrix, initial_medoids, data_type='distance_matrix',
                                 itermax=10000, tolerance=0.00001)
    kmedoids_instance.process()
    return kmedoids_instance.get_medoids(), kmedoids_instance.get_clusters()


def get_dbscan_clusters(distmatrix, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', algorithm='brute').fit(distmatrix)
    clusters = [np.where(db.labels_ == i)[0] for i in np.unique(db.labels_) if i != -1
                and len(np.where(db.labels_ == i)[0]) > 1]
    # for each clister find the medoid
    medoids = []
    for cluster in clusters:
        submatrix = distmatrix[np.ix_(cluster, cluster)]
        # set submatrix diagonal to 1e12
        # np.fill_diagonal(submatrix, 1e12)
        medoid = np.argmin(submatrix.sum(axis=0))
        medoids.append(cluster[medoid])
    return medoids, clusters


def get_spectral_clusters(distmatrix, n_clusters):
    db = SpectralClustering(n_clusters=n_clusters, affinity='precomputed_nearest_neighbors',
                            assign_labels = 'discretize', random_state = 0).fit(distmatrix)

    clusters = [np.where(db.labels_ == i)[0] for i in np.unique(db.labels_) if i != -1
                and len(np.where(db.labels_ == i)[0]) > 1]
    # for each clister find the medoid
    medoids = []
    for cluster in clusters:
        submatrix = distmatrix[np.ix_(cluster, cluster)]
        # set submatrix diagonal to 1e12
        # np.fill_diagonal(submatrix, 1e12)
        medoid = np.argmin(submatrix.sum(axis=0))
        medoids.append(cluster[medoid])
    return medoids, clusters


def make_silhouette_gridsearch_for_dbscan(distmatrix, eps_list, min_samples_list):
    # make gridsearch
    silhouette_scores = []
    # divide distmatrix by max
    for eps in eps_list:
        for min_samples in min_samples_list:
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', algorithm='brute').fit(distmatrix)
            if len(np.unique(db.labels_)) > 1:
                silhouette_scores.append(silhouette_score(distmatrix, db.labels_, metric='precomputed'))
            else:
                silhouette_scores.append(0)
    silhouette_scores = np.array(silhouette_scores).reshape(len(eps_list), len(min_samples_list))
    return silhouette_scores

def make_calinski_harabasz_score_gridsearch_for_dbscan(distmatrix, eps_list, min_samples_list):
    # make gridsearch
    calinski_harabasz_scores = []
    # divide distmatrix by max
    for eps in eps_list:
        for min_samples in min_samples_list:
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', algorithm='brute').fit(distmatrix)
            if len(np.unique(db.labels_)) > 1:
                calinski_harabasz_scores.append(calinski_harabasz_score(distmatrix, db.labels_))
            else:
                calinski_harabasz_scores.append(0)
    calinski_harabasz_scores = np.array(calinski_harabasz_scores).reshape(len(eps_list), len(min_samples_list))
    return calinski_harabasz_scores

def make_number_of_clusters_gridsearch_for_dbscan(distmatrix, eps_list, min_samples_list):
    # make gridsearch
    number_of_clusters = []
    # divide distmatrix by max
    for eps in eps_list:
        for min_samples in min_samples_list:
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distmatrix)
            number_of_clusters.append(len(np.unique(db.labels_)))
    number_of_clusters = np.array(number_of_clusters).reshape(len(eps_list), len(min_samples_list))
    return number_of_clusters


def plot_gridsearch_for_dbscan(silhouette_scores, eps_list, min_samples_list, vmax=1):
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(silhouette_scores, annot=True, fmt='.2f', ax=ax, xticklabels=min_samples_list,
                yticklabels=eps_list, vmax=vmax)
    ax.set_xlabel('min_samples')
    ax.set_ylabel('eps')

    ax.set_title('Score for DBSCAN clustering')
    plt.tight_layout()
    plt.show()


def plot_clusters(hdf_filepath, title='', do_show=True, size_for_points=0, alpha=0.5, suffix='', percentile_for_rad=99,
                  rad_lim_factor=1, distmatrix_cache_filename=None, plot_clusters=False,
                  number_of_medoids=4, colors = ('C4', 'C2', 'C1', 'C3'), rmin=10,
                  out_svg_file='temp.svg', annotation_label_shift=0.9, figsize=(7, 5)):
    df = pd.read_hdf(hdf_filepath)
    #filename without extension
    db_filename = hdf_filepath.split('.')[0].split('/')[-1]
    df['count'] = df['count'].astype(int)

    df['x'] = df['x'] - df['x'].mean()
    df['y'] = df['y'] - df['y'].mean()

    fig, ax = plt.subplots(figsize=figsize)

    fp_colname = 'fp_ECFP4'
    df[fp_colname] = df['smiles'].apply(chiral_fingerprint)

    # make and plot medoids
    distmatrix = np.load(distmatrix_cache_filename)
    assert distmatrix.shape[0] == len(df)
    # medoids, clusters = get_k_medoids(distmatrix, number_of_medoids)
    # medoids, clusters = get_dbscan_clusters(distmatrix, eps=0.62, min_samples=4)
    medoids, clusters = get_spectral_clusters(distmatrix, n_clusters=number_of_medoids)
    # make dictionary of medoid:cluster
    cluster_dict = {medoids[i]:clusters[i] for i in range(len(medoids))}
    # sort medoids list
    medoids.sort()
    clusters = [cluster_dict[medoid] for medoid in medoids]

    max_abs_x = np.max(np.abs(df['x']).to_numpy())

    medoids_ys = df.loc[medoids, 'y']
    ys_sorted = np.sort(medoids_ys)
    def is_odd_sign(i):
        # if i is odd return -1
        if i % 2 == 0:
            return 1
        else:
            return -1
    y_to_sign_dict = {y:is_odd_sign(i) for i, y in enumerate(ys_sorted)}

    for i,medoid in enumerate(medoids):
        # get x,y from df
        x = df.loc[medoid, 'x']
        y = df.loc[medoid, 'y']
        plt.scatter(x, y, s=100, marker='x', linewidth=3, zorder=100, color=colors[i])

        # sknunk box with id sk2
        box = skunk.Box(70, 35, f'sk{i}')
        ab = AnnotationBbox(box, (x, y),
                            xybox=(y_to_sign_dict[y]*max_abs_x*1.6, y),
                            xycoords='data',
                            arrowprops=dict(arrowstyle="->"),
                            bboxprops=dict(edgecolor=colors[i]))

        plt.gca().add_artist(ab)

        smiles_to_svg(df.loc[medoid, 'smiles'], f'figures/temp3/sk{i}.svg', size=(400, 200))

    # turn the column 'smiles' to canonical
    for j, rep in enumerate(representative_smiles):
        # location of this smiles in the df
        df_one_rep = df[df['smiles'] == rep]
        assert len(df_one_rep) == 1, f'len(df_one_rep) = {len(df_one_rep)} for rep {rep}'
        x = df_one_rep['x'].values[0]
        y = df_one_rep['y'].values[0]
        if (j+1) in [1, 3, 9, 11]:
            alpha_here = 1
        else:
            alpha_here = 0.5
        plt.scatter(x, y, s=100, marker='o', linewidth=3, zorder=100, facecolors='none', edgecolors='black', alpha=alpha_here)
        # annotate with BBn for n-th representative with vertical shift of 0.1
        plt.text(x + annotation_label_shift, y + annotation_label_shift, f'{j + 1}', fontsize=12, ha='left', va='bottom', alpha=0.5)

    # ax = None
    for i, cluster in enumerate(clusters):
        print(f'medoid {medoids[i]}: {len(cluster)} members')
        df_here = df[df.index.isin(cluster)]
        if size_for_points == 0:
            sizes = rmin + df_here['count'] * 10
        else:
            sizes = [size_for_points] * len(df_here)
        markers = []
        for j, row in df_here.iterrows():
            if is_trans(row['smiles']) == 'trans':
                markers.append('o')
            elif is_trans(row['smiles']) == 'cis':
                markers.append('^')
            elif is_trans(row['smiles']) == 'both':
                markers.append('s')
            elif is_trans(row['smiles']) == 'neither':
                markers.append('D')

        # markers = 's'
        # if ax is None:
        xs = df_here['x'].to_list()
        ys = df_here['y'].to_list()
        for j in range(len(xs)):
            ax.scatter(x=xs[j], y=ys[j], s=sizes[j],
                        edgecolor=None, alpha=alpha, label=f'{i}', zorder=30, linewidth=0, marker=markers[j], color=colors[i])
        # else:
        #     sns.scatterplot(ax=ax, x="x", y="y", s=sizes, data=df_here,
        #                     edgecolor=None, alpha=alpha, label=f'{i}', zorder=30, linewidth=0, markers=markers)

    # plot all the points that are not in clusters
    df_not_in_clusters = df[~df.index.isin(np.concatenate(clusters))]
    if size_for_points == 0:
        sizes = rmin + df_not_in_clusters['count'] * 10
    else:
        sizes = size_for_points

    # iterate over df_not_in_clusters and choose marker based on cis-trans
    markers = []
    for i, row in df_not_in_clusters.iterrows():
        if is_trans(row['smiles']) == 'trans':
            markers.append('o')
        elif is_trans(row['smiles']) == 'cis':
            markers.append('^')
        else:
            markers.append('s')

    # sns.scatterplot(ax=ax, x="x", y="y", s=sizes, data=df_not_in_clusters, color='black',
    #                 edgecolor=None, alpha=alpha, label=f'not in clusters', zorder=30, linewidth=0, markers=markers)
    for j in range(len(df_not_in_clusters)):
        ax.scatter(x=df_not_in_clusters.iloc[j]['x'], y=df_not_in_clusters.iloc[j]['y'], s=sizes[j],
                    edgecolor=None, alpha=alpha, label=f'not in clusters', zorder=30, linewidth=0, marker=markers[j], color='black')

    plt.axis('equal')
    plt.axis('off')
    # find rad_lim radius limit such that 99% of points are within the circle
    rad_lim = np.percentile(np.sqrt(df['x'] ** 2 + df['y'] ** 2), percentile_for_rad)
    rad_lim *= rad_lim_factor
    plt.xlim(-rad_lim, rad_lim)
    plt.ylim(-rad_lim, rad_lim)

    # ax.get_legend().remove()

    plt.subplots_adjust(top=0.95)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(f'figures/embeddings/halide_clustering_{suffix}.png', dpi=300)
    # fig.savefig(f'figures/embeddings/panelH_{suffix}.eps', dpi=300)

    # insert current figure into itself at sk1
    # insert svg file in sk2
    svg = skunk.insert(
        {
            f'sk{i}': f'figures/temp3/sk{i}.svg'
        for i in range(len(medoids))
        }
    )

    # write to file
    with open(out_svg_file, 'w') as f:
        f.write(svg)

    svg_code = open(out_svg_file, 'rt').read()
    svg2png(bytestring=svg_code, write_to=f'{out_svg_file}.png', dpi=300)

    if do_show:
        plt.show()
    else:
        # delete figure fig
        plt.close(fig)
        plt.clf()
        plt.cla()


if __name__ == '__main__':
    # ## HYBRID METRIC
    # hyw_str = '1p00'
    # db_filepath = f'data/unique_halides_reclassed_plus_bbs_hybridW1p00-{hyw_str}_tsne_px40_lr70_50kiter.hdf'
    # distmatrix_cache_filename = 'data/unique_halides_reclassed_plus_bbs_distance_matrix_hybridW1p00-1p00.npy'
    # for nclusters in range(4, 9):
    #     plot_clusters(db_filepath,
    #                     title=f'',
    #                     suffix=f'_hybridW1p00-{hyw_str}_kmedoids',
    #                     rad_lim_factor=1.4,
    #                     distmatrix_cache_filename=distmatrix_cache_filename,
    #                     plot_clusters=True,
    #                     number_of_medoids=nclusters,
    #                     size_for_points=20,
    #                     colors = [f'C{i}' for i in range(10)],
    #                     out_svg_file=f'figures/embeddings/halide_clustering_nclusters{nclusters}.svg',
    #                     do_show=False
    #                   )

    ## BROMINE-CENTERED LOCAL METRIC
    decayrad_str = '4p0'
    px = 60
    db_filepath = f'data/unique_halides_reclassed_plus_bbs_localmetric_decayrad{decayrad_str}_tsne_px{px}_lr70_50kiter.hdf'
    distmatrix_cache_filename = f'data/unique_halides_reclassed_plus_bbs_distance_matrix_localmetric_decayrad{decayrad_str}.npy'
    for nclusters in range(4, 9):
        plot_clusters(db_filepath,
                        title=f'',
                        suffix=f'_localmetric_decayrad{decayrad_str}_px{px}',
                        rad_lim_factor=1.4,
                        distmatrix_cache_filename=distmatrix_cache_filename,
                        plot_clusters=True,
                        number_of_medoids=nclusters,
                        size_for_points=20,
                        colors = [f'C{i}' for i in range(10)],
                        out_svg_file=f'figures/embeddings/halide_clustering_localmetric_decayrad{decayrad_str}_px{px}_nclusters{nclusters}.svg',
                        do_show=True,
                        annotation_label_shift=0.4,
                        alpha=0.3,
                        figsize=(6, 7)
                      )

    # # dbscan param gridsearch
    # distmatrix = np.load(distmatrix_cache_filename)
    # eps_list = np.linspace(0.5, 1.2, 30)
    # min_samples_list = np.arange(2, 10)
    # # silh = make_silhouette_gridsearch_for_dbscan(distmatrix, eps_list, min_samples_list)
    # # plot_gridsearch_for_dbscan(silh, eps_list, min_samples_list)
    # # nclust = make_number_of_clusters_gridsearch_for_dbscan(distmatrix, eps_list, min_samples_list)
    # # plot_gridsearch_for_dbscan(nclust, eps_list, min_samples_list, vmax=6)
    # carb = make_calinski_harabasz_score_gridsearch_for_dbscan(distmatrix, eps_list, min_samples_list)
    # plot_gridsearch_for_dbscan(carb, eps_list, min_samples_list, vmax=110)


