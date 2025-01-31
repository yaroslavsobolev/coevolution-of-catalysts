import logging

import matplotlib.pyplot as plt
import sqlalchemy.orm
from matplotlib.colors import ListedColormap

from halides_selection import *

from faerun import Faerun

from pyclustering.cluster.kmedoids import kmedoids

from halides_clustering import *

def mark_ab(hdf_filepath):
    df = pd.read_hdf(f'data/{hdf_filepath}')
    # filename without extension
    db_filename = hdf_filepath.split('.')[0].split('/')[-1]

    # mark alpha-methyl-beta-hydroxy
    specific_molecular_pattern_SMARTS = '[CH3]-[CH](-[*])-[CH](-[*])-O'

    specific_molecular_pattern = Chem.MolFromSmarts(specific_molecular_pattern_SMARTS)
    img = Draw.MolToImage(specific_molecular_pattern)
    plt.imshow(img)
    plt.show()
    # iterate over df with tqdm
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        mol = Chem.MolFromSmiles(row['smiles'])
        # if mol.HasSubstructMatch(specific_molecular_pattern):
        if len(mol.GetSubstructMatches(specific_molecular_pattern)) > 1:
            # mark in 'polyketideome' column as 1
            df.loc[i, 'a-methyl-b-hydroxyl'] = 1
        else:
            df.loc[i, 'a-methyl-b-hydroxyl'] = 0

    print('Number of polyketides in the dataset: ', df['a-methyl-b-hydroxyl'].sum())

    df.to_hdf(f'data/{db_filename}_with_ab.hdf', 'df')



# def plot_panelA2_from_hdf(hdf_filepath, title='', do_show=True):
#     df = pd.read_hdf(hdf_filepath)
#     # filename without extension
#     db_filename = hdf_filepath.split('.')[0].split('/')[-1]
#
#     # # mark alpha-methyl-beta-hydroxy
#     # specific_molecular_pattern_SMARTS = '[CH3]-[CH](-[C,c])-[CH](-[C,c])-O'
#     #
#     # specific_molecular_pattern = Chem.MolFromSmarts(specific_molecular_pattern_SMARTS)
#     # # iterate over df with tqdm
#     # for i, row in tqdm(df.iterrows(), total=df.shape[0]):
#     #     mol = Chem.MolFromSmiles(row['smiles'])
#     #     # if mol.HasSubstructMatch(specific_molecular_pattern):
#     #     if len(mol.GetSubstructMatches(specific_molecular_pattern)) > 1:
#     #         # mark in 'polyketideome' column as 1
#     #         df.loc[i, 'a-methyl-b-hydroxyl'] = 1
#     #     else:
#     #         df.loc[i, 'a-methyl-b-hydroxyl'] = 0
#     #
#     # print('Number of ab in the dataset: ', df['a-methyl-b-hydroxyl'].sum())
#
#     # defining the colors into the 'color' column. If synthesizable column is 1, then color is red else blue
#     # df['color'] = df['synthesizable'].apply(lambda x: 'PK' if x == 1 else 'Other')
#
#     fig = plt.figure(figsize=(5, 5))
#     ax = sns.scatterplot(x="x", y="y", s=7, data=df[df['a-methyl-b-hydroxyl'] == 0], color='C2', edgecolor=None, alpha=0.05,
#                          linewidth=0)
#     sns.scatterplot(ax=ax, x="x", y="y", s=7, data=df[df['a-methyl-b-hydroxyl'] == 1], color='C0', edgecolor=None, alpha=0.05,
#                     label='a-methyl-b-hydroxyl', zorder=9, linewidth=0)
#     # sns.scatterplot(ax=ax, x="x", y="y", s=7, data=df[df['synthesizable'].notna()], color='C3', edgecolor=None, alpha=0.2,
#     #                 label='Non-red polyketides\nfrom your Excel', zorder=10, linewidth=0)
#
#     # sns.scatterplot(ax=ax, x="x", y="y", s=20, data=df[df['BB3'].notna()], color='C3', edgecolor=None, alpha=1,
#     #                 label='BB3', zorder=20, linewidth=0)
#     # sns.scatterplot(ax=ax, x="x", y="y", s=20, data=df[df['BB6'].notna()], color='C2', edgecolor=None, alpha=1,
#     #                 label='BB6', zorder=30, linewidth=0)
#
#     plt.axis('equal')
#     plt.axis('off')
#     # find rad_lim radius limit such that 99% of points are within the circle
#     rad_lim = np.percentile(np.sqrt(df['x'] ** 2 + df['y'] ** 2), 99)
#     plt.xlim(-rad_lim, rad_lim)
#     plt.ylim(-rad_lim, rad_lim)
#     # plt.legend()
#     ## remove legend
#     ax.get_legend().remove()
#     # add margin on the top for the title
#     plt.subplots_adjust(top=0.95)
#     plt.title(title)
#     plt.tight_layout()
#     fig.savefig(f'figures/embeddings/panelA.png', dpi=300)
#     if do_show:
#         plt.show()
#     else:
#         # delete figure fig
#         plt.close(fig)
#         plt.clf()
#         plt.cla()


def plot_panelA_from_hdf(hdf_filepath, title='', do_show=True, radlim_percentile=99, do_center_of_mass_centering=True,
                         suffix='', figfactor=1, legend=True, mark_molecules=False, marklabels=True,
                         custom_x_limits=None, figsize=(5, 6)):
    df = pd.read_hdf(hdf_filepath)
    #filename without extension
    db_filename = hdf_filepath.split('.')[0].split('/')[-1]
    # # for all x column values that are larger than 146, decrease then by 30
    # df.loc[df['x'] > 146, 'x'] = df.loc[df['x'] > 146, 'x'] - 50

    if do_center_of_mass_centering:
        df['x'] = df['x'] - df['x'].mean()
        df['y'] = df['y'] - df['y'].mean()
    # # mark alpha-methyl-beta-hydroxy
    # specific_molecular_pattern_SMARTS = '[CH3]-[CH](-[C,c])-[CH](-[C,c])-O'
    #
    # specific_molecular_pattern = Chem.MolFromSmarts(specific_molecular_pattern_SMARTS)
    # for smi in tqdm(df.index):
    #     mol = Chem.MolFromSmiles(smi)
    #     # if mol.HasSubstructMatch(specific_molecular_pattern):
    #     if len(mol.GetSubstructMatches(specific_molecular_pattern)) > 1:
    #         # mark in 'polyketideome' column as 1
    #         df.loc[smi, 'a-methyl-b-hydroxyl'] = 1
    #     else:
    #         df.loc[smi, 'a-methyl-b-hydroxyl'] = 0
    #
    # print('Number of polyketides in the dataset: ', df['a-methyl-b-hydroxyl'].sum())

    # defining the colors into the 'color' column. If synthesizable column is 1, then color is red else blue
    # df['color'] = df['synthesizable'].apply(lambda x: 'PK' if x == 1 else 'Other')
    substance_classes = ['is_pk', 'is_steroid',
       'is_flavonoid', 'is_alkaloid', 'is_terpene']
    fig = plt.figure(figsize=(figsize[0]*figfactor, figsize[1]*figfactor))

    ax = sns.scatterplot(x="x", y="y", s=7, data=df[df['is_pk']==1], color='C2', edgecolor=None, alpha=0.05,
                    zorder=9, linewidth=0)
    sns.scatterplot(ax=ax, x="x", y="y", s=7, data=df[df['is_steroid']==1], color='C4', edgecolor=None, alpha=0.05,
                    zorder=9, linewidth=0)
    sns.scatterplot(ax=ax, x="x", y="y", s=7, data=df[df['is_flavonoid']==1], color='C3', edgecolor=None, alpha=0.05,
                    zorder=9, linewidth=0)
    sns.scatterplot(ax=ax, x="x", y="y", s=7, data=df[df['is_alkaloid']==1], color='C1', edgecolor=None, alpha=0.05,
                    zorder=9, linewidth=0)
    sns.scatterplot(ax=ax, x="x", y="y", s=7, data=df[df['is_terpene']==1], color='C0', edgecolor=None, alpha=0.05,
                    zorder=9, linewidth=0)
    sns.scatterplot(ax=ax, x="x", y="y", s=7, data=df[df[substance_classes].eq(0).all(axis=1)], color='grey', edgecolor=None, alpha=0.01,
                         linewidth=0, zorder=-10)
    # sns.scatterplot(ax=ax, x="x", y="y", s=7, data=df[df['synthesizable'].notna()], color='C3', edgecolor=None, alpha=0.2,
    #                 label='Non-red polyketides\nfrom your Excel', zorder=10, linewidth=0)

    # sns.scatterplot(ax=ax, x="x", y="y", s=20, data=df[df['BB3'].notna()], color='C3', edgecolor=None, alpha=1,
    #                 label='BB3', zorder=20, linewidth=0)
    # sns.scatterplot(ax=ax, x="x", y="y", s=20, data=df[df['BB6'].notna()], color='C2', edgecolor=None, alpha=1,
    #                 label='BB6', zorder=30, linewidth=0)

    if mark_molecules:
        mark_names = ['discodermolide',
                      # 'artemisinin',
                      'taxol',
                      'caffeine',
                      'quercetin',
                      'ergosterol',
                      'roselipin 1A']

        list_of_smiles_to_mark = ['O[C@H]([C@@H](C)/C=C/C=C/C=C/C=C/C=C/C=C/C=C/[C@H](O[C@@]1([H])[C@@H](O)[C@@H](N)[C@H](O)[C@@H](C)O1)C[C@@H]([C@H](C(O)=O)[C@@H](O)C2)O[C@]2(O)C[C@@H](O)C[C@@H](O)[C@H](O)CC[C@@H](O)C[C@@H](O)C3)[C@@H](C)[C@H](C)OC3=O',
                                  # 'O=C(O[C@H]1[C@@]23[C@H]4CC[C@](OO3)(C)O1)[C@H](C)[C@@H]2CC[C@H]4C',
                                  'CC1=C2C(OC(C)=O)C(C3(C)C(O)CC4C(CO4)(OC(C)=O)C3C(OC(C5=CC=CC=C5)=O)C(CC1OC(C(O)C(NC(C6=CC=CC=C6)=O)C7=CC=CC=C7)=O)(O)C2(C)C)=O',
                                  'O=C(N(C)C(N1C)=O)C2=C1N=CN2C',
                                  'O=C(C1=C(O)C=C(O)C=C1O2)C(O)=C2C3=CC(O)=C(O)C=C3',
                                  'O[C@H](C1)CC[C@@]2(C)C1=CC=C3[C@]2([H])CC[C@@]4(C)[C@@]3([H])CCC4[C@@H](/C=C/[C@@H](C(C)C)C)C',
                                  'CC[C@H](C)C[C@H](C)C[C@H](C)[C@@H](O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]1O)[C@@H](C)/C=C(\\C)[C@@H](O)[C@@H](C)/C=C(\\C)[C@@H](OCOC)[C@@H](C)/C=C(\\C)C(=O)OC[C@H](O)[C@H](O)[C@@H](O)CO']
        list_of_smiles_to_mark = [Chem.CanonSmiles(smi) for smi in list_of_smiles_to_mark]
        fp_colname = 'fp_ECFP6_bv'

        # # fingerprints from scratch
        # logging.info(f'Calculating fingerprints for {len(df)} molecules')
        # df[fp_colname] = df['smiles'].apply(fingerprint_bitvect)

        # load fingerprints from other file
        df_with_fps = pd.read_pickle('data/DNP_FULL_2016_with_polyketides_fingerprints_fixed2.pickle')
        # assert than numbers of rows match
        assert len(df) == len(df_with_fps)
        # assert that smiles match in the first 10 rows
        assert df['smiles'].head(10).eq(df_with_fps['smiles'].head(10)).all()

        # copy fingerprints from other file
        df[fp_colname] = df_with_fps[fp_colname]

        logging.info(f'Calculating similarities for {len(list_of_smiles_to_mark)} molecules')
        for i, smi in enumerate(list_of_smiles_to_mark):
            fp_here = fingerprint_bitvect(smi)
            similarities = df[fp_colname].apply(lambda x: tanisim(x, fp_here))
            # find the row of df with highest similarity
            max_sim_row = similarities.idxmax()
            # print similarity in that row
            print(f'i={i}, {mark_names[i]}: highest sim is {similarities.loc[max_sim_row]}')
            print(f'Smiles with max sim is: {df.loc[max_sim_row, "smiles"]}')
            x = df.loc[max_sim_row, 'x']
            y = df.loc[max_sim_row, 'y']
            logging.info(f'{mark_names[i]}: x={x}, y={y}')
            plt.scatter(x, y, c='black', alpha=1, s=20, zorder=100, linewidth=0)
            # make horiz shift for text, text on right of the point
            if marklabels:
                plt.text(x+0.2, y, mark_names[i], fontsize=8, zorder=101, ha='left', va='center')


    plt.axis('equal')
    plt.axis('off')

    print(f'ymean {df["y"].mean()}')
    # find rad_lim radius limit such that 99% of points are within the circle
    rad_lim = np.percentile(np.sqrt(df['x'] ** 2 + df['y'] ** 2), radlim_percentile)
    plt.xlim(-rad_lim, rad_lim)
    plt.ylim(-rad_lim, rad_lim)
    if custom_x_limits is not None:
        plt.xlim(custom_x_limits[0], custom_x_limits[1])

    colors = ['C2', 'C4', 'C3', 'C1', 'C0', 'black']
    labels = ['polyketides', 'steroids', 'flavonoids', 'alkaloids', 'terpenes', 'other']
    # plot large scatter dot for the legend
    for i in range(len(colors)):
        plt.scatter([], [], c=colors[i], alpha=0.8, s=50, label=labels[i])
    if legend:
        plt.legend(loc='upper left')
    ## remove legend
    # add margin on the top for the title
    plt.subplots_adjust(top=0.95)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(f'figures/embeddings/panelA3{suffix}.png', dpi=300)
    if do_show:
        plt.show()
    else:
        # delete figure fig
        plt.close(fig)
        plt.clf()
        plt.cla()

def faerun_plot_panel_A_from_hdf(hdf_filepath, title='DNP_TSNE_ECFP6', limit=False, save_annotated=False):
    filepath_without_ext = os.path.splitext(hdf_filepath)[0]
    annotated_filepath = f'{filepath_without_ext}_annotated.hdf'
    # if it exists, load it
    if os.path.exists(annotated_filepath):
        if limit:
            df = pd.read_hdf(annotated_filepath).head(limit)
        else:
            df = pd.read_hdf(annotated_filepath)
    else:
        if limit:
            df = pd.read_hdf(hdf_filepath).head(limit)
        else:
            df = pd.read_hdf(hdf_filepath)

        # annotate_labels
        for i, row in tqdm(df.iterrows(), desc='Annotating labels', total=len(df)):
            mol_here = Chem.MolFromSmiles(row['smiles'])
            canon_smiles = Chem.MolToSmiles(mol_here)
            df.loc[i, 'smiles'] = canon_smiles
            chemical_formula = Chem.rdMolDescriptors.CalcMolFormula(mol_here)
            label = f"{canon_smiles}__{chemical_formula}"
            df.loc[i, 'label'] = label


        substance_classes = ['is_pk', 'is_steroid',
           'is_flavonoid', 'is_alkaloid', 'is_terpene']
        df['bb'] = 5
        for i, row in tqdm(df.iterrows(), desc='Assigning classes', total=len(df)):
            for j, c in enumerate(substance_classes):
                if row[c] == 1:
                    df.loc[i, 'bb'] = j
                    break

        if save_annotated:
            df.to_hdf(f'{filepath_without_ext}_annotated.hdf', key='df', mode='w')

    bb_ids = df['bb'].values
    # labels_groups, groups = Faerun.create_categories(bb_ids)
    labels_groups = [(i, x) for i, x in enumerate(['polyketides', 'steroids', 'flavonoids', 'alkaloids', 'terpenes', 'other'])]

    # select_BBs = ['BB5', 'BB1', 'BB3', 'BB9']
    # colors = ['C4', 'C2', 'C1', 'C3']
    # colors = [f'C{i}' for i in range(0, 10)]
    colors = ['C2', 'C4', 'C3', 'C1', 'C0', 'grey']
    custom_cmap = ListedColormap(colors, name="custom2")
    faerun = Faerun(view="front", coords=False, title=title)#, clear_color="#ffffff", )
    faerun.add_scatter(title, {"x": df['x'], "y": df['y'],
                                    "c": [bb_ids],
                                    "labels": df['label']}, has_legend=True, \
                       colormap=[custom_cmap], \
                       point_scale=0.3,
                       categorical=[True], \
                       series_title=["class"], \
                       max_legend_label=[None], \
                       min_legend_label=[None],
                       legend_labels=[labels_groups])
    faerun.plot(title, template='smiles')




def plot_panelB_from_hdf(hdf_filepath, title='', do_show=True, alpha_for_pk=0.05):
    df = pd.read_hdf(hdf_filepath)
    #filename without extension
    db_filename = hdf_filepath.split('.')[0].split('/')[-1]

    # defining the colors into the 'color' column. If synthesizable column is 1, then color is red else blue
    # df['color'] = df['synthesizable'].apply(lambda x: 'PK' if x == 1 else 'Other')

    fig = plt.figure(figsize=(5, 5))
    ax = sns.scatterplot(x="x", y="y", s=7, data=df[(df['is_pk']==1) & (df['a-methyl-b-hydroxyl']==0)],
                         color='C0', edgecolor=None, alpha=alpha_for_pk,
                         linewidth=0)
    # sns.scatterplot(ax=ax, x="x", y="y", s=7, data=df[df['polyketideome'].notna()], color='C1', edgecolor=None, alpha=0.2,
    #                 label='$\\alpha$-methyl-$\\beta$-hydroxy', zorder=9, linewidth=0)
    sns.scatterplot(ax=ax, x="x", y="y", s=7, data=df[(df['is_pk']==1) & (df['a-methyl-b-hydroxyl']==1)], color='C1', edgecolor=None, alpha=0.2,
                    label='a-methyl-b-hydroxyl', zorder=10, linewidth=0)
    # sns.scatterplot(ax=ax, x="x", y="y", s=20, data=df[df['BB3'].notna()], color='C2', edgecolor=None, alpha=1,
    #                 label='BB3', zorder=20, linewidth=0)
    # sns.scatterplot(ax=ax, x="x", y="y", s=20, data=df[df['BB6'].notna()], color='C2', edgecolor=None, alpha=1,
    #                 label='BB6', zorder=30, linewidth=0)

    plt.axis('equal')
    plt.axis('off')
    # find rad_lim radius limit such that 99% of points are within the circle
    rad_lim = np.percentile(np.sqrt(df['x'] ** 2 + df['y'] ** 2), 99)
    plt.xlim(-rad_lim, rad_lim)
    plt.ylim(-rad_lim, rad_lim)
    # plt.legend()
    ## remove legend
    ax.get_legend().remove()
    # add margin on the top for the title
    plt.subplots_adjust(top=0.95)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(f'figures/embeddings/panelB.png', dpi=300)
    if do_show:
        plt.show()
    else:
        # delete figure fig
        plt.close(fig)
        plt.clf()
        plt.cla()


def plot_panelC_from_hdf(hdf_filepath, title='', do_show=True, size_for_points=10,
                         alpha=0.4, alpha_parent=0.1, suffix='', colors = ['C4', 'C3', 'C2', 'C1'],
                         option='4A'):
    df = pd.read_hdf(hdf_filepath)
    #filename without extension
    db_filename = hdf_filepath.split('.')[0].split('/')[-1]

    # print sums in each column. Column names are from 'BB1' to 'BB16'
    columns_list = ['BB1', 'BB2', 'BB3', 'BB4', 'BB5', 'BB6', 'BB7', 'BB8', 'BB9', 'BB10', 'BB11', 'BB12', 'BB13', 'BB14', 'BB15', 'BB16']
    sums_here = df[columns_list].sum().to_numpy()
    for i, x in enumerate(sums_here):
        print(f'{columns_list[i]}\t{x}')


    # defining the colors into the 'color' column. If synthesizable column is 1, then color is red else blue
    # df['color'] = df['synthesizable'].apply(lambda x: 'PK' if x == 1 else 'Other')

    fig = plt.figure(figsize=(5, 5))
    # ax = sns.scatterplot(x="x", y="y", s=7, data=df[(df['is_pk']==1) & (df['synthesizable']==0)],
    #                      color='C0', edgecolor=None, alpha=0.05,
    #                      linewidth=0)
    # sns.scatterplot(ax=ax, x="x", y="y", s=7, data=df[df['polyketideome'].notna()], color='C1', edgecolor=None, alpha=0.2,
    #                 label='$\\alpha$-methyl-$\\beta$-hydroxy', zorder=9, linewidth=0)
    # select_BBs = ['BB1', 'BB3', 'BB4', 'BB9']
    select_BBs = ['BB11', 'BB1', 'BB3', 'BB9']
    # a mask where all of the select BBs columns of df are zero
    mask = df[select_BBs].eq(0).all(axis=1)
    # ax = sns.scatterplot(x="x", y="y", s=7, data=df[(df['is_pk']==1) & (df['a-methyl-b-hydroxyl']==1) & mask], color='C0',
    #                      edgecolor=None, alpha=alpha_parent,
    #                 label='a-methyl-b-hydroxyl', zorder=10, linewidth=0)

    # ax = sns.scatterplot(x="x", y="y", s=size_for_points, data=df[(df['BB_count'] > 0) & mask], color='black',
    #                      edgecolor=None, alpha=alpha,
    #                      label='polyketides', zorder=-10, linewidth=0)

    if option=='2B':
        ax = sns.scatterplot(x="x", y="y", s=7, data=df[(df['is_pk']==1) & (df['BB_count'] == 0)], color='C2',
                             edgecolor=None, alpha=alpha_parent,
                        label='polyketides', zorder=-10, linewidth=0)
    elif option=='4A':
        ax = sns.scatterplot(x="x", y="y", s=7, data=df[mask], color='#000F87',
                             edgecolor=None, alpha=alpha_parent,
                        label='other_alpha_methyl_beta_hydroxyl', zorder=-10, linewidth=0)


    # ic(df[(df['is_pk']==1) & (df['a-methyl-b-hydroxyl']==1)].count())
    # sns.scatterplot(ax=ax, x="x", y="y", s=20, data=df[df['BB3']>0], color='C3', edgecolor=None, alpha=1,
    #                 label='BB3', zorder=20, linewidth=0)
    # sns.scatterplot(ax=ax, x="x", y="y", s=20, data=df[df['BB6']>0], color='C4', edgecolor=None, alpha=1,
    #                 label='BB6', zorder=30, linewidth=0)
    for i, BB in enumerate(select_BBs):
        sns.scatterplot(ax=ax, x="x", y="y", s=size_for_points, data=df[df[BB] > 0], color=colors[i], edgecolor=None, alpha=alpha,
                        label=BB, zorder=20+i, linewidth=0)

    plt.axis('equal')
    plt.axis('off')
    # find rad_lim radius limit such that 99% of points are within the circle
    rad_lim = np.percentile(np.sqrt(df['x'] ** 2 + df['y'] ** 2), 99)
    plt.xlim(-rad_lim, rad_lim)
    plt.ylim(-rad_lim, rad_lim)
    # plt.legend()
    ## remove legend
    ax.get_legend().remove()
    # add margin on the top for the title
    plt.subplots_adjust(top=0.95)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(f'figures/embeddings/panelC{suffix}.png', dpi=300)
    fig.savefig(f'figures/embeddings/panelC{suffix}.eps', dpi=300)

    if do_show:
        plt.show()
    else:
        # delete figure fig
        plt.close(fig)
        plt.clf()
        plt.cla()


def faerun_plot_panel_C(hdf_filepath, title='Polyketides_TSNE_ECFP6'):
    df = pd.read_hdf(hdf_filepath)

    df = df[(df['is_pk']==1) | (df['BB_count'] > 0)]

    # annotate_labels
    for i, row in tqdm(df.iterrows(), desc='Annotating labels', total=len(df)):
        mol_here = Chem.MolFromSmiles(row['smiles'])
        canon_smiles = Chem.MolToSmiles(mol_here)
        df.loc[i, 'smiles'] = canon_smiles
        chemical_formula = Chem.rdMolDescriptors.CalcMolFormula(mol_here)
        label = f"{canon_smiles}__{chemical_formula}"
        df.loc[i, 'label'] = label

    df['bb'] = 'BB0'
    for i, row in df.iterrows():
        for k in range(1, 17):
            if row[f'BB{k}'] > 0:
                df.loc[i, 'bb'] = f'BB{k}'
                break


    bb_ids = [int(x[2:]) for x in df['bb'].to_list()]
    labels_groups, groups = Faerun.create_categories(bb_ids)
    # select_BBs = ['BB5', 'BB1', 'BB3', 'BB9']
    # colors = ['C4', 'C2', 'C1', 'C3']
    colors = ['grey', 'C2', 'C0', 'C1', 'C3', 'C4'] + [f'C{i}' for i in range(5, 10)]
    colors = colors + ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    custom_cmap = ListedColormap(colors, name="custom")
    faerun = Faerun(view="front", coords=False, title=title)#, clear_color="#ffffff", )
    faerun.add_scatter(title, {"x": df['x'], "y": df['y'],
                                    "c": [bb_ids],
                                    "labels": df['label']}, has_legend=True, \
                       colormap=[custom_cmap], \
                       point_scale=2,
                       categorical=[True], \
                       series_title=["group"], \
                       max_legend_label=[None], \
                       min_legend_label=[None])
    faerun.plot(title, template='smiles')


def tsne_for_halides(fp_type='MAP4'):
    df_filename = 'data/unique_halides_reclassed.pickle'
    df = pd.read_pickle(df_filename)
    df = fix_halide_smiles(df, 'smiles')

    distance_matrix(df, cache_filename=f'data/unique_halides_distance_matrix.npy',
                    force_recalculate=True, algo='loop', fp_colname='fp_ECFP6_bv')

    tsne_params = {'n_components': 2, 'verbose': 2, 'random_state': 137,
                   'metric': 'precomputed',
                   'early_exaggeration': 1, 'perplexity': 40, 'init': 'random',
                   'n_iter': 50000, 'learning_rate': 70,
                   'n_iter_without_progress': 20000, 'method': 'barnes_hut',
                   'n_jobs': 1}

    for perplexity in [3000]:  # , [1000, 3000, 10000, 30000, 100000, 300]:
        for learning_rate in [70000]:
            logging.info(f'learning_rate={learning_rate}, perplexity={perplexity}')
            db_filepath = df_filename
            filepath_without_extension = db_filepath.split('.')[0]
            distmatrix_cache_filename = f'{filepath_without_extension}_distance_matrix.npy'
            compute_tsne_with_distmatrix_and_save(db_filepath=db_filepath,
                                                  distmatrix_cache_filename=distmatrix_cache_filename,
                                                  output_filename=f'{filepath_without_extension}_tsne_px{tsne_params["perplexity"]}_lr{tsne_params["learning_rate"]}_'
                                                                  f'{int(round(tsne_params["n_iter"] / 1000))}kiter.hdf',
                                                  memmap_mode='r', tsne_params=tsne_params,
                                                  change_dtypes_to_str=False)


def tsne_for_halides_with_hybrid_metric(fp_type='MAP4', hybrid_weights=(0.5, 2), df_filename = 'data/unique_halides_reclassed.pickle'):
    hybrid_weights_str = [f'{x:.2f}'.replace('.', 'p') for x in hybrid_weights]
    db_filepath = df_filename
    filepath_without_extension = db_filepath.split('.')[0]
    distmatrix_cache_filename = f'{filepath_without_extension}_distance_matrix_hybridW{hybrid_weights_str[0]}-{hybrid_weights_str[1]}.npy'
    df = pd.read_pickle(df_filename)
    df = fix_halide_smiles(df, 'smiles')
    df_npas = pd.read_csv('D:/Docs/Dropbox/Lab/catalyst-coevolution/unique_halides_and_bb_vacuum.csv')
    logging.info(f'NaNs in the NPA column: {df_npas["npa"].isna().sum()}')

    for i, row in df.iterrows():
        # find row in df_npas with same value in 'smiles' colyumn and copy npa value to 'npa' column in df
        df.loc[i, 'npa'] = df_npas[df_npas['smiles'] == row['smiles']]['npa'].values[0]
        # if same smiles is not found in df_npas, set npa to 0 and notify with logging.info
        if len(df_npas[df_npas['smiles'] == row['smiles']]['npa'].values) == 0:
            df.loc[i, 'npa'] = 0
            logging.info(f'No NPA value for {row["smiles"]}, row {i}')
    # replace nan npas with 0
    df['npa'] = df['npa'].fillna(0)

    # iterate over df rows and make a distance matrix
    fp_colname = 'fp_ECFP6_bv'
    dist_matrix = np.zeros((len(df), len(df)), dtype=np.float16)
    tuple_of_fingerprints = tuple(df[fp_colname].to_list())
    for i in tqdm(range(len(df)), total=len(df), desc='Computing distance matrix'):
        for j in range(len(df)):
            tanimoto_distance = 1 - tanisim(tuple_of_fingerprints[i], tuple_of_fingerprints[j])
            npa_distance = abs(df.loc[i, 'npa'] - df.loc[j, 'npa'])
            dist_matrix[i, j] = tanimoto_distance * hybrid_weights[0] + npa_distance * hybrid_weights[1]

    # save distance matrix
    np.save(distmatrix_cache_filename, dist_matrix)

    tsne_params = {'n_components': 2, 'verbose': 2, 'random_state': 137,
                   'metric': 'precomputed',
                   'early_exaggeration': 1, 'perplexity': 40, 'init': 'random',
                   'n_iter': 50000, 'learning_rate': 70,
                   'n_iter_without_progress': 20000, 'method': 'barnes_hut',
                   'n_jobs': 1}

    perplexity = 3000
    learning_rate = 70000
    logging.info(f'learning_rate={learning_rate}, perplexity={perplexity}')
    # make strings from hybrid weights, replacing dot with p

    compute_tsne_with_distmatrix_and_save(db_filepath=db_filepath,
                                          distmatrix_cache_filename=distmatrix_cache_filename,
                                          output_filename=f'{filepath_without_extension}_hybridW{hybrid_weights_str[0]}-{hybrid_weights_str[1]}_tsne_px{tsne_params["perplexity"]}_lr{tsne_params["learning_rate"]}_'
                                                          f'{int(round(tsne_params["n_iter"] / 1000))}kiter.hdf',
                                          memmap_mode='r', tsne_params=tsne_params,
                                          change_dtypes_to_str=False)


def tsne_for_halides_with_local_metric(df_filename = 'data/unique_halides_reclassed.pickle', decay_radius=2,
                                       perplexity=40):
    db_filepath = df_filename
    filepath_without_extension = db_filepath.split('.')[0]
    string_of_decay_radius = f'{decay_radius:.1f}'.replace('.', 'p')
    distmatrix_cache_filename = f'{filepath_without_extension}_distance_matrix_localmetric_decayrad{string_of_decay_radius}.npy'
    df = pd.read_pickle(df_filename)
    df = fix_halide_smiles(df, 'smiles')

    # if distmatrix_cache_filename exists, load it and return
    if os.path.exists(distmatrix_cache_filename):
        logging.info(f'Loading distance matrix from {distmatrix_cache_filename}')
        dist_matrix = np.load(distmatrix_cache_filename)
    else:
        # iterate over df rows and make a distance matrix
        dist_matrix = np.zeros((len(df), len(df)), dtype=np.float16)
        for i in tqdm(range(len(df)), total=len(df), desc='Computing distance matrix'):
            for j in range(len(df)):
                molecule1 = Chem.MolFromSmiles(df.loc[i, 'smiles'])
                molecule2 = Chem.MolFromSmiles(df.loc[j, 'smiles'])
                local_exponential_distance = exp_similarity_from_bromine(molecule1, molecule2, decay_radius=decay_radius)
                dist_matrix[i, j] = 1 - local_exponential_distance
        # save distance matrix
        np.save(distmatrix_cache_filename, dist_matrix)

    tsne_params = {'n_components': 2, 'verbose': 2, 'random_state': 137,
                   'metric': 'precomputed',
                   'early_exaggeration': 1, 'perplexity': 40, 'init': 'random',
                   'n_iter': 50000, 'learning_rate': 70,
                   'n_iter_without_progress': 20000, 'method': 'barnes_hut',
                   'n_jobs': 1}

    tsne_params['perplexity'] = perplexity
    # make strings from hybrid weights, replacing dot with p

    compute_tsne_with_distmatrix_and_save(db_filepath=db_filepath,
                                          distmatrix_cache_filename=distmatrix_cache_filename,
                                          output_filename=f'{filepath_without_extension}_localmetric_decayrad{string_of_decay_radius}_tsne_px{tsne_params["perplexity"]}_lr{tsne_params["learning_rate"]}_'
                                                          f'{int(round(tsne_params["n_iter"] / 1000))}kiter.hdf',
                                          memmap_mode='r', tsne_params=tsne_params,
                                          change_dtypes_to_str=False)


def tsne_for_halides_map4():
    # compute_fingerprints_for_df(db_filepath='data/unique_halides.pickle',
    #                             force_recalculate=True, fp_colname='fp_MAP4', N_PROC=1)
    #
    # df = pd.read_pickle('data/unique_halides_map4fingerprints.pickle')
    # distmat = distance_matrix(df, cache_filename=f'data/unique_halides_map4fingerprints_distance_matrix.npy',
    #                 force_recalculate=True, algo='loop', fp_colname='fp_MAP4_bv',
    #                           chunksize=1, max_workers=1)

    df_filename = 'data/unique_halides_map4fingerprints.pickle'
    tsne_params = {'n_components': 2, 'verbose': 2, 'random_state': 137,
                   'metric': 'precomputed',
                   'early_exaggeration': 1, 'perplexity': 20, 'init': 'random',
                   'n_iter': 50000, 'learning_rate': 1,
                   'n_iter_without_progress': 50000, 'method': 'barnes_hut',
                   'n_jobs': 1}

    for perplexity in [3000]:  # , [1000, 3000, 10000, 30000, 100000, 300]:
        for learning_rate in [70000]:
            logging.info(f'learning_rate={learning_rate}, perplexity={perplexity}')
            db_filepath = df_filename
            filepath_without_extension = db_filepath.split('.')[0]
            distmatrix_cache_filename = f'{filepath_without_extension}_distance_matrix.npy'
            compute_tsne_with_distmatrix_and_save(db_filepath=db_filepath,
                                                  distmatrix_cache_filename=distmatrix_cache_filename,
                                                  output_filename=f'{filepath_without_extension}_tsne_px{tsne_params["perplexity"]}_lr{tsne_params["learning_rate"]}_'
                                                                  f'{int(round(tsne_params["n_iter"] / 1000))}kiter.hdf',
                                                  memmap_mode='r', tsne_params=tsne_params,
                                                  change_dtypes_to_str=False)


def plot_panel_halides_from_hdf(hdf_filepath, title='', do_show=True, size_for_points=10, alpha=0.5, suffix='', percentile_for_rad=99,
                                rad_lim_factor=1, distmatrix_cache_filename=None, plot_clusters=False,
                                number_of_medoids = 4, colors = ('C4', 'C2', 'C1', 'C3')):
    df = pd.read_hdf(hdf_filepath)
    #filename without extension
    db_filename = hdf_filepath.split('.')[0].split('/')[-1]
    df['count'] = df['count'].astype(int)
    # print first 10 sorted by count, high first
    df_sorted = df.sort_values(by='count', ascending=False)
    # in df_sorted set 'smiles' column as index column
    df_sorted.set_index('smiles', inplace=True)
    # drop columns x, y, smiles
    df_sorted.drop(columns=['x', 'y'], inplace=True)
    print(df_sorted)

    # # print a sum of "count" column for rows where 'bb' column is 'BB3'
    # print(df_sorted[df_sorted['bb'] == 'BB3']['count'].sum())

    # subtract mean x from x and mean y from y
    df['x'] = df['x'] - df['x'].mean()
    df['y'] = df['y'] - df['y'].mean()
    # set count column type to int

    # group by 'bb' column and sort by count of rows in each group
    df_grouped = df.groupby('bb').count().sort_values(by='bb', ascending=True)
    print(df_grouped)

    # defining the colors into the 'color' column. If synthesizable column is 1, then color is red else blue
    # df['color'] = df['synthesizable'].apply(lambda x: 'PK' if x == 1 else 'Other')

    fig = plt.figure(figsize=(5, 5))

    # ax = sns.scatterplot(x="x", y="y", s=7, data=df[(df['is_pk']==1) & (df['synthesizable']==0)],
    #                      color='C0', edgecolor=None, alpha=0.05,
    #                      linewidth=0)
    # sns.scatterplot(ax=ax, x="x", y="y", s=7, data=df[df['polyketideome'].notna()], color='C1', edgecolor=None, alpha=0.2,
    #                 label='$\\alpha$-methyl-$\\beta$-hydroxy', zorder=9, linewidth=0)
    # print sums of "count' grouped by 'bb' category
    print(df.groupby('bb')['count'].sum())
    select_BBs = ['BB11', 'BB1', 'BB3', 'BB9']
    rmin = 10
    if not plot_clusters:
        for i, BB in enumerate(select_BBs):
            df_here = df[df['bb'] == BB]
            if size_for_points == 0:
                sizes = rmin + df_here['count'].to_numpy() * 10
            else:
                sizes = size_for_points
            if i == 0:
                ax = sns.scatterplot(x="x", y="y", s=sizes, data=df[df['bb'] == BB], color=colors[i],
                                     edgecolor=None, alpha=alpha, label=BB, zorder=30, linewidth=0)
            else:
                sns.scatterplot(ax = ax, x="x", y="y", s=sizes, data=df[df['bb'] == BB], color=colors[i],
                                     edgecolor=None, alpha=alpha, label=f'{BB}', zorder=30, linewidth=0)

        # plot the ones that are not in select BBs
        df_not_in_select_BBs = df[~df['bb'].isin(select_BBs)]
        if size_for_points == 0:
            sizes = rmin + df_not_in_select_BBs['count'].to_numpy() * 10
        else:
            sizes = size_for_points
        sns.scatterplot(ax=ax, x="x", y="y", s=sizes, data=df_not_in_select_BBs, color='grey',
                        edgecolor=None, alpha=alpha, label='Other', zorder=20, linewidth=0)

    fp_colname = 'fp_ECFP4'
    df[fp_colname] = df['smiles'].apply(fingerprint)
    def mark_one_halide(smiles, color='black', label='L'):
        fp = fingerprint(smiles)
        # iterate over polyketides df and find the most similar molecule in the full df
        similarities = df[fp_colname].apply(lambda x: tanisim(x, fp))
        # find the index of the most similar molecule
        idx = similarities.idxmax()
        print(f'{smiles} is most similar to {df.loc[idx, "smiles"]}, similarity {max(similarities)}')
        # get the x and y of the most similar molecule
        x = df.loc[idx, 'x']
        y = df.loc[idx, 'y']
        size = rmin + df.loc[idx, 'count'] * 10
        # plot a red dot
        plt.scatter(x, y, s=size, marker='o', linewidth=2, zorder=100,
                    facecolors='none', edgecolors=color)
        # annotate the molecule with offset equal to the size of the dot
        plt.annotate(label, (x, y), fontsize=18, color=color, zorder=100, xytext=(-5, 10), textcoords='offset points',)

    # mark_one_halide('BrC1=CC=CC=C1', color='black', label='A')
    # mark_one_halide('Br/C=C(C(OC(C)(C)C)=O)\\C', color='black', label='B')
    # mark_one_halide('Br/C=C(C(ON(C)C)=O)\\C', color='black', label='C')
    # mark_one_halide('C/C(C)=C/Br', color='C5', label='D')
    #
    #
    # # make and plot medoids
    # if distmatrix_cache_filename is not None:
    #     distmatrix = np.load(distmatrix_cache_filename)
    #     assert distmatrix.shape[0] == len(df)
    #     # Initialize medoids randomly
    #     medoids, clusters = get_spectral_clusters(distmatrix, n_clusters=number_of_medoids)
    #
    #     cluster_dict = {medoids[i]: clusters[i] for i in range(len(medoids))}
    #     medoids.sort()
    #     clusters = [cluster_dict[medoid] for medoid in medoids]
    #
    #     for medoid in medoids:
    #         # get x,y from df
    #         x = df.loc[medoid, 'x']
    #         y = df.loc[medoid, 'y']
    #         plt.scatter(x, y, s=100, marker='x', linewidth=3, zorder=100, color='black')
    #
    #     if plot_clusters:
    #         ax = None
    #         for i, cluster in enumerate(clusters):
    #             print(f'medoid {medoids[i]}: {len(cluster)} members')
    #             df_here = df[df.index.isin(cluster)]
    #             if size_for_points == 0:
    #                 sizes = rmin + df_here['count'] * 10
    #             else:
    #                 sizes = size_for_points
    #             if ax is None:
    #                 ax = sns.scatterplot(x="x", y="y", s=sizes, data=df_here, color=colors[i],
    #                                      edgecolor=None, alpha=alpha, label=f'{i}', zorder=30, linewidth=0)
    #             else:
    #                 sns.scatterplot(ax=ax, x="x", y="y", s=sizes, data=df_here, color=colors[i],
    #                                 edgecolor=None, alpha=alpha, label=f'{i}', zorder=30, linewidth=0)

    plt.axis('equal')
    plt.axis('off')
    # find rad_lim radius limit such that 99% of points are within the circle
    rad_lim = np.percentile(np.sqrt(df['x'] ** 2 + df['y'] ** 2), percentile_for_rad)
    rad_lim *= rad_lim_factor
    plt.xlim(-rad_lim, rad_lim)
    plt.ylim(-rad_lim, rad_lim)
    plt.legend()
    ## remove legend
    # ax.get_legend().remove()
    # add margin on the top for the title
    plt.subplots_adjust(top=0.95)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(f'figures/embeddings/panelH_{suffix}.png', dpi=300)
    # fig.savefig(f'figures/embeddings/panelH_{suffix}.eps', dpi=300)
    if do_show:
        plt.show()
    else:
        # delete figure fig
        plt.close(fig)
        plt.clf()
        plt.cla()


def add_BBs():
    # hybrid_weights_str = [f'{x:.2f}'.replace('.', 'p') for x in hybrid_weights]
    df_filename = 'data/unique_halides_reclassed.pickle'
    db_filepath = df_filename
    filepath_without_extension = db_filepath.split('.')[0]
    df = pd.read_pickle(df_filename)
    df = fix_halide_smiles(df, 'smiles')
    df_npas = pd.read_csv('data/qm/unique_halides_and_bb_vacuum.csv')
    logging.info(f'NaNs in the NPA column: {df_npas["npa"].isna().sum()}')
    # for rows in df_npas with indices starting from 230, add the rows into df
    for index in range(230, len(df_npas)):
        smiles_here = df_npas.loc[index, 'smiles']
        # npa_here = df_npas.loc[index, 'npa']
        BB_here = df_npas.loc[index, 'bb']
        fingerprint_here = chiral_fingerprint(smiles_here)
        fingerprint_bv_here = chiral_fingerprint_bitvect(smiles_here)
        # add a row into df with the same smiles, npa, and BB
        df = df.append({'smiles': smiles_here, 'bb': BB_here, 'count':0, 'fp_ECFP6':fingerprint_here,
                        'fp_ECFP6_bv':fingerprint_bv_here},
                       ignore_index=True)
    df.to_pickle('data/unique_halides_reclassed_plus_bbs.pickle')
    # sum of counts where bb columns is equal to BB1
    sum_bb1 = df[df['bb'] == 'BB1']['count'].sum()

def huh():
    df = pd.read_pickle('data/polyketides_bitvect_fp.pickle')
    # drop 'smiles' column
    df.drop(columns=['smiles'], inplace=True)
    # rename column 'Halide SMILES' to 'smiles'
    df.rename(columns={'Halide SMILES': 'smiles'}, inplace=True)
    df = df[df['synthesizable'] == 1]

    # use only rows where smiles column is not None and not empty string
    df = df[df['smiles'].notna()]
    df = df[df['smiles'] != '']
    # reset index
    df.reset_index(drop=True, inplace=True)
    fp_colname = 'fp_ECFP4'
    df[fp_colname] = df['smiles'].apply(fingerprint)
    def mark_one_halide(smiles, color='black', label='L'):
        fp = fingerprint(smiles)
        # iterate over polyketides df and find the most similar molecule in the full df
        similarities = df[fp_colname].apply(lambda x: tanisim(x, fp))
        # find the index of the most similar molecule
        idx = similarities.idxmax()
        print(f'{smiles} is most similar to {df.loc[idx, "smiles"]}, similarity {max(similarities)}')
    mark_one_halide('BrC1=CC=CC=C1', color='black', label='A')
    mark_one_halide('Br/C=C(C(OC(C)(C)C)=O)\\C', color='black', label='B')
    mark_one_halide('Br/C=C(C(ON(C)C)=O)\\C', color='black', label='C')
    mark_one_halide('C/C(C)=C/Br', color='C5', label='D')
    mark_one_halide('O=C(C)C(C)(/C=C/Br)C', color='C5', label='E')


def propagate_reclassing(hdf_filepath='data/only_polyketides_fingerprints_len100k_tsne_px80_lr1855_50kiter_with_ab.hdf',
                         output_filepath='data/only_polyketides_fingerprints_len100k_tsne_px80_lr1855_50kiter_with_ab_rec.hdf'):
    df = pd.read_hdf(hdf_filepath)
    df_reclassed = pd.read_pickle('data/unique_halides_reclassed.pickle')
    list_of_bb_cols = [f'BB{i}' for i in range(1, 17)]
    for i, row in df.iterrows():
        if row['BB4'] != 1:
            continue
        smiles = Chem.CanonSmiles(row['Halide SMILES'])
        smiles2 = Chem.CanonSmiles(row['smiles'])
        # find row with the same smiles in df_reclassed
        df_reclassed_row = df_reclassed[df_reclassed['smiles'] == smiles]
        if len(df_reclassed_row) == 0:
            print(f'{smiles} not found in df_reclassed')
            continue
        assert len(df_reclassed_row) == 1
        # get the reclassed value of bb class
        bb_class = df_reclassed_row['bb'].values[0]
        # set the value of bb class in df
        df.loc[i, 'BB4'] = 0
        df.loc[i, bb_class] = 1

    df.to_hdf(output_filepath, key='df', mode='w')


def faerun_plot_halides(hdf_filepath):
    df = pd.read_hdf(hdf_filepath)
    bb_ids = [int(x[2:]) for x in df['bb'].to_list()]
    labels_groups, groups = Faerun.create_categories(bb_ids)
    # select_BBs = ['BB5', 'BB1', 'BB3', 'BB9']
    # colors = ['C4', 'C2', 'C1', 'C3']
    colors = ['k', 'C2', 'C0', 'C1', 'C3', 'C4'] + [f'C{i}' for i in range(5, 10)]
    colors = colors + ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    custom_cmap = ListedColormap(colors, name="custom")
    faerun = Faerun(view="front", coords=False, title='Halides_embedding_TSNE_MAP4', clear_color="#ffffff",)
    faerun.add_scatter("Halides_embedding_TSNE_MAP4", {"x": df['x'], "y": df['y'],
                                    "c": [bb_ids, df['count']],
                                    "labels": df['smiles']}, has_legend=True, \
                       colormap=[custom_cmap, "rainbow"], \
                       point_scale=20,
                       categorical=[True, False], \
                       series_title=["group", "count"], \
                       max_legend_label=[None, "maxcount"], \
                       min_legend_label=[None, 'mincount'])
    faerun.plot('Halides_embedding_TSNE_MAP4', template='smiles')

if __name__ == '__main__':
    add_BBs()
    # filename = 'only_ab_polyketides_fingerprints_tsne_px30_lr707_50kiter_reclassed.hdf'
    # plot_panelC_from_hdf(hdf_filepath=f'data/{filename}',
    #                      alpha_parent=0.6, suffix='_ab_only')

    # plot_panelB_from_hdf(hdf_filepath=f'data/{filename}', alpha_for_pk=0.2)

    # propagate_reclassing()

    # faerun_plot_halides('data/unique_halides_reclassed_tsne_px40_lr70_50kiter.hdf')

    # faerun_plot_panel_C(hdf_filepath=f'data/only_polyketides_fingerprints_len100k_tsne_px80_lr1855_50kiter_with_ab_rec.hdf')
    # plot_panelC_from_hdf(f'data/only_polyketides_fingerprints_len100k_tsne_px80_lr1855_50kiter_with_ab_rec.hdf')

    # # huh()
    # # mark_ab('DNP_FULL_2016_with_polyketides_fingerprints_len100k_tsne_px1060_lr24735_50kiter.hdf')
    # # mark_ab('only_polyketides_fingerprints_len100k_tsne_px80_lr1855_50kiter.hdf')
    #
    # version = 3

    # T-SNE ECFP4 for entire DNP
    # filename = 'DNP_FULL_2016_with_polyketides_fingerprints_len100k_tsne_px1060_lr24735_5kiter.hdf'
    filename = 'DNP_FULL_2016_with_polyketides_fingerprints_len100k_tsne_px1060_lr24735_50kiter.hdf' ## this one is used in the article
    # filename = 'DNP_FULL_2016_with_polyketides_fingerprints_len100k_tsne_px353_lr24735_50kiter.hdf'
    plot_panelA_from_hdf(hdf_filepath=f'data/{filename}', custom_x_limits=(-273, 287), figsize=(9, 6),
                         suffix='_article')

    # filename = 'DNP_FULL_2016_with_polyketides_fingerprints_len100k_tsne_px3534_lr24735_5kiter_fixed.hdf'
    # plot_panelA_from_hdf(hdf_filepath=f'data/{filename}', custom_x_limits=(-273, 287), figsize=(9,6),
    #                      suffix='px3534')
    # filename = 'DNP_FULL_2016_with_polyketides_fingerprints_len100k_tsne_px106_lr24735_50kiter_fixed.hdf'
    # plot_panelA_from_hdf(hdf_filepath=f'data/{filename}', custom_x_limits=(-273, 287), figsize=(9,6),
    #                      suffix='px106')

    # filename = 'DNP_FULL_2016_with_polyketides_fingerprints_len100k_tsne_px35_lr24735_50kiter_fixed.hdf'
    # plot_panelA_from_hdf(hdf_filepath=f'data/{filename}', custom_x_limits=(-273, 287), figsize=(9,6),
    #                      suffix='px35')

    filename = 'DNP_FULL_2016_with_polyketides_fingerprints_len100k_tsne_px1060_lr24735_50kiter.hdf'
    faerun_plot_panel_A_from_hdf(hdf_filepath=f'data/{filename}')

    # UMAP
    # filename = 'DNP_FULL_2016_with_polyketides_fingerprints_fixed2_map4fingerprints_umap_nn1000.hdf'
    # filename = 'DNP_FULL_2016_with_polyketides_fingerprints_fixed2_map4fingerprints_umap_nn300.hdf'
    # filename = 'DNP_FULL_2016_with_polyketides_fingerprints_fixed2_map4fingerprints_umap_nn1000_md0.1.hdf'
    # for mindist in reversed([0.05, 0.1, 0.2, 0.5, 1]):
    #     filename = f'DNP_FULL_2016_with_polyketides_fingerprints_fixed2_map4fingerprints_umap_nn1000_md{mindist}.hdf'
    #     # plot_panelA_from_hdf(hdf_filepath=f'data/{filename}', radlim_percentile=85, suffix=f'_md{mindist}',
    #     #                      title=f'UMAP (min_dist={mindist})', figfactor=1.5)
    #     faerun_plot_panel_A_from_hdf(hdf_filepath=f'data/{filename}',
    #                                  title=f'DNP_UMAP_min_dist_{str(mindist).replace(".","p")}', save_annotated=True)

    # ### UMAP for main figure 1
    # mindist = 0.05
    # filename = f'DNP_FULL_2016_with_polyketides_fingerprints_fixed2_map4fingerprints_umap_nn1000_md{mindist}.hdf'
    # # plot_panelA_from_hdf(hdf_filepath=f'data/{filename}', radlim_percentile=85, suffix=f'_md{mindist}',
    # #                      title=f'UMAP (min_dist={mindist})', figfactor=1.5)
    # # plot_panelA_from_hdf(hdf_filepath=f'data/{filename}', radlim_percentile=85, suffix=f'_md{mindist}_nolegend',
    # #                      title='', figfactor=1.5, legend=False)
    # plot_panelA_from_hdf(hdf_filepath=f'data/{filename}', radlim_percentile=85, suffix=f'_md{mindist}_nolegend_nolabels_1',
    #                      title='', figfactor=1.5, legend=False, marklabels=False)
    # #
    # # if version > 1:
    # #     # T-SNE ECFP4 for only polyketides
    # #     filename = 'only_polyketides_fingerprints_len100k_tsne_px80_lr1855_50kiter_with_ab.hdf'
    # #     plot_panelB_from_hdf(hdf_filepath=f'data/{filename}', alpha_for_pk=0.2)
    # # else:
    # #     plot_panelB_from_hdf(hdf_filepath=f'data/{filename}')
    # #
    # # if version > 2:
    # #     # T-SNE ECFP4 for only a-methyl-b-hydroxyl polyketides
    # #     filename = 'only_ab_polyketides_fingerprints_tsne_px30_lr707_50kiter_reclassed.hdf'
    # # plot_panelC_from_hdf(hdf_filepath=f'data/{filename}', alpha_parent=0.4)
    #
    # plot_panelC_from_hdf(hdf_filepath=f'data/only_polyketides_fingerprints_len100k_tsne_px80_lr1855_50kiter_with_ab_rec.hdf',
    #                      alpha_parent=0.25, alpha=0.25)
    #
    #
    # ########### Halides
    #
    # tsne_for_halides()
    # plot_panel_halides_from_hdf('data/unique_halides_reclassed_tsne_px40_lr70_50kiter.hdf')
    # # plot_panel_halides_from_hdf('data/unique_halides_tsne_px100_lr1_50kiter.hdf')
    # # plot_panel_halides_from_hdf('data/unique_halides_tsne_px100_lr200_50kiter.hdf')
    # # plot_panel_halides_from_hdf('data/unique_halides_tsne_px40_lr70_50kiter.hdf')
    #

    ####### Hybrid metric Halides
    # for hyw in [1]:
    #     hyw_str = f'{hyw:.2f}'.replace('.', 'p')
    #     tsne_for_halides_with_hybrid_metric(hybrid_weights=[1, hyw])
    #     plot_panel_halides_from_hdf(f'data/unique_halides_reclassed_withBBs_hybridW1p00-{hyw_str}_tsne_px40_lr70_50kiter.hdf',
    #                                 title = f'Tanimoto_distance + NPA_distance*{hyw:.2f}',
    #                                 suffix=f'_hybridW1p00-{hyw_str}',
    #                                 rad_lim_factor=1.4)

    ## Version with representative halides included
    # for hyw in [1]:
    #     hyw_str = f'{hyw:.2f}'.replace('.', 'p')
    #     tsne_for_halides_with_hybrid_metric(hybrid_weights=[1, hyw],
    #                                         df_filename='data/unique_halides_reclassed_plus_bbs.pickle')
    #     plot_panel_halides_from_hdf(f'data/unique_halides_reclassed_plus_bbs_hybridW1p00-{hyw_str}_tsne_px40_lr70_50kiter.hdf',
    #                                 title = f'Tanimoto_distance + NPA_distance*{hyw:.2f}',
    #                                 suffix=f'_hybridW1p00-{hyw_str}',
    #                                 rad_lim_factor=1.4)

    # # with local metric
    # decay_radius = 4
    # perplexity = 60
    # string_of_decay_radius = f'{decay_radius:.1f}'.replace('.', 'p')
    # tsne_for_halides_with_local_metric(decay_radius=decay_radius,
    #                                     df_filename='data/unique_halides_reclassed_plus_bbs.pickle', perplexity=perplexity)
    # plot_panel_halides_from_hdf(f'data/unique_halides_reclassed_plus_bbs_localmetric_decayrad{string_of_decay_radius}_tsne_px{perplexity}_lr70_50kiter.hdf',
    #                             title = f'Bromine-centered metric, decay radius {decay_radius}',
    #                             suffix=f'_decayrad_{string_of_decay_radius}_px{perplexity}',
    #                             rad_lim_factor=1.4,
    #                             size_for_points=10)


# # with K-Medoids
#     hyw_str = '1p00'
#     db_filepath = f'data/unique_halides_reclassed_hybridW1p00-{hyw_str}_tsne_px40_lr70_50kiter.hdf'
#     plot_panel_halides_from_hdf(db_filepath,
#                                 title=f'Hybrid metric, BB groups colored',
#                                 suffix=f'_hybridW1p00-{hyw_str}_spectral_bbs',
#                                 rad_lim_factor=1.4,
#                                 distmatrix_cache_filename='data/unique_halides_reclassed_distance_matrix_hybridW1p00-1p00.npy',
#                                 plot_clusters=True,
#                                 number_of_medoids=4,
#                                 colors = ('C4', 'C2', 'C1', 'C3', 'C0'),
#                                 size_for_points=0)
    #
    # plot_panel_halides_from_hdf(db_filepath,
    #                             title=f'Hybrid metric, clusters colored',
    #                             suffix=f'_hybridW1p00-{hyw_str}_spectral_clusters',
    #                             rad_lim_factor=1.4,
    #                             distmatrix_cache_filename='data/unique_halides_reclassed_distance_matrix_hybridW1p00-1p00.npy',
    #                             plot_clusters=True,
    #                             number_of_medoids=4,
    #                             colors = ('grey', 'C3', 'C2', 'C1', 'C5'),
    #                             size_for_points=0)

    # ####### MAP$ Halides
    # # tsne_for_halides_map4()
    # plot_panel_halides_from_hdf('data/unique_halides_map4fingerprints_tsne_px20_lr1_50kiter.hdf')
    # plot_panel_halides_from_hdf('data/unique_halides_reclassed_plus_bbs_localmetric_decayrad4p0_tsne_px60_lr70_50kiter.hdf')


