# checking query interface
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdCoordGen
from icecream import ic
import plotly.graph_objects as go
from sklearn_extra.cluster import KMedoids


def sketch1():
    smiles1 = 'CCC/C=C/C(C(C1OC(CC(C(CC/C=C/C(C(/C=C/1)C)O)=C)C)=O)O)C'
    m1 = Chem.MolFromSmiles(smiles1)

    smiles_p = '(*)CC(O)([H])C(C([H])([H])[H])([H])C=C* |$;;;;;_R1;;;;;_R2$|'

    smi = 'c1nc(*)ccc1* |$;;;R1;;;;R2$|'
    mol = Chem.MolFromSmiles(smi)
    mol.GetAtomWithIdx(3).GetProp("atomLabel")
    mol.GetAtomWithIdx(7).GetProp("atomLabel")
    d = rdMolDraw2D.MolDraw2DCairo(250, 250)
    rdMolDraw2D.PrepareAndDrawMolecule(d,mol)
    d.WriteDrawingText("figures/atom_labels_1.png")

def sketch2():
    # example_smiles = 'CCC/C=C/C(C(C(/C=C/C(C)C)OC(CC(C(CCC=C)=C)C)=O)O)C'
    # example_smiles = 'CCC/C=C/C(C(C(/C=C/C(C)C)OC(CC(C(CCC=C)=C)C)=O)OC)C'
    # example_smiles = 'CCC/C=C/C(C(C(/C=C/C(C)C)OC(CC(C(CCC=C)=C)C)=O)O)CC'
    example_smiles = 'COC1CC(C)Cc2c(OC)c(O)cc3NC(=O)/C(C)=C/CCC(OC)C(O)/C(C)=C/C(C)C1Oc23'
    example_mol = Chem.MolFromSmiles(example_smiles)

    mol_minimum = Chem.MolFromSmarts('[CH3]-[CH](-[CH]=C)-[CH](-C)-[OH1]')

    img = Draw.MolToImage(mol)
    plt.imshow(img)
    plt.show()
    # ps = Chem.AdjustQueryParameters()
    # ps.makeDummiesQueries=True
    # mol = Chem.AdjustQueryProperties(mol, ps)

    # check that mol has substruct
    ic(example_mol.HasSubstructMatch(mol))


def sketch3():
    ###################### Figuring out the entry 7
    df = pd.read_pickle('data/unique_halides.pickle')
    # in row where 'smiles' is 'C/C(Br)=C/C(C)C(C)O', change column 'bb' to 'BB4'
    df = df[df['bb'] == 'BB4']

    patterns = dict()
    # patterns['BB1'] = Chem.MolFromSmarts('BrC(-[H])=C(-[H])[C,c]')
    patterns['BB1'] = Chem.MolFromSmarts('Br/[CH]=[CH]/C')
    patterns['BB2'] = Chem.MolFromSmarts('Br/[CH]=[CH]\C')
    patterns['BB3'] = Chem.MolFromSmarts('Br/[CH]=C(C)C')

    mol = Chem.MolFromSmarts('Br/C(-C)=[CH]/C')
    img = Draw.MolToImage(mol)
    plt.imshow(img)
    plt.show()

    for example_smiles in df['smiles']:
        # example_smiles = 'COC1CC(Cc2c3c(NC(/C(C)=C/CCC(C(/C(C)=C/C(C1O3)C)O)OC)=O)cc(O)c2OC)C'
        # example_smiles = 'CCC/C=C/C(C(C(/C=C/C(C)C)OC(CC(C(CCC=C)=C)C)=O)OC)C'
        # example_smiles = 'CCC/C=C/C(C(C(/C=C/C(C)C)OC(CC(C(CCC=C)=C)C)=O)O)CC'
        # example_smiles = 'COC1CC(C)Cc2c(OC)c(O)cc3NC(=O)/C(C)=C/CCC(OC)C(O)/C(C)=C/C(C)C1Oc23'
        example_mol = Chem.MolFromSmiles(example_smiles)

        print(example_smiles)
        hasmatch = example_mol.HasSubstructMatch(mol, useChirality=True)


        # if hasmatch:
        #     hit_ats = list(example_mol.GetSubstructMatches(mol, useChirality=True))
        #     hit_bonds = []
        #     for bond in mol.GetBonds():
        #         aid1 = hit_ats[bond.GetBeginAtomIdx()]
        #         aid2 = hit_ats[bond.GetEndAtomIdx()]
        #         hit_bonds.append(mol.GetBondBetweenAtoms(aid1, aid2).GetIdx())
        #     d = rdMolDraw2D.MolDraw2DSVG(500, 500)  # or MolDraw2DCairo to get PNGs
        #     rdMolDraw2D.PrepareAndDrawMolecule(d, example_mol, highlightAtoms=hit_ats,
        #                                        highlightBonds=hit_bonds)

        img = Draw.MolToImage(example_mol)
        plt.imshow(img)
        plt.show()
    # ps = Chem.AdjustQueryParameters()
    # ps.makeDummiesQueries=True
    # mol = Chem.AdjustQueryProperties(mol, ps)

    df.loc[df['smiles'] == 'C/C(Br)=C/C(C)C(C)O', 'bb'] = 'BB4'


def sketch_inspect_selection():
    specific_molecular_pattern_SMARTS = '[CH3]-[CH](-C=[C,c])-[CH](-[C,c])-O'
    p = Chem.MolFromSmiles('C([H])([H])([H])C([H])(C=C)C([H])(C)O')

    df = pd.read_hdf(
        'data/DNP_FULL_2016_with_polyketides_fingerprints_len100k_tsne_px1060_lr24735_pkmerged_selected.hdf')
    subms = [Chem.MolFromSmiles(smiles) for smiles in df.index.values]
    names = [str(x) for x in df['name'].to_list()]
    for n in names:
        print(n)

    ic(len(subms))
    subms = [x for x in subms if x.HasSubstructMatch(p)]
    ic(len(subms))

    rdCoordGen.AddCoords(p)
    for m in subms:
        _ = AllChem.GenerateDepictionMatching2DStructure(m, p)
    pattern = Chem.MolFromSmarts(specific_molecular_pattern_SMARTS)

    # Highlighting the part that matches the pattern
    hlist_atoms = []
    hlist_bonds = []
    for molecule in subms:
        hit_ats = list(molecule.GetSubstructMatch(p))
        hit_bonds = []
        for bond in pattern.GetBonds():
            aid1 = hit_ats[bond.GetBeginAtomIdx()]
            aid2 = hit_ats[bond.GetEndAtomIdx()]
            hit_bonds.append(molecule.GetBondBetweenAtoms(aid1, aid2).GetIdx())
        hlist_atoms.append(hit_ats)
        hlist_bonds.append(hit_bonds)
    img = Draw.MolsToGridImage(subms, molsPerRow=4, subImgSize=(600, 600), highlightAtomLists=hlist_atoms,
                               highlightBondLists=hlist_bonds)
    img.save('figures/cdk2_molgrid.aligned.o.png')


def sketch4():
    from rdkit import Chem

    def neutralize_atoms(mol):
        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        return mol
    # from IPython.display import HTML, display
    # mol = Chem.MolFromSmarts('C[Mo+]')
    smarts = '[H][C@@]12CCCC[C@@]1([H])[S](C(C)(C)C)[Ir+]134([CH]5=[CH]1CC[CH]3=[CH]4CC5)[P]1(O2)OC2=C(C=C(C=C2C2=CC(=CC(=C2O1)C(C)(C)C)C(C)(C)C)C(C)(C)C)C(C)(C)C.FC(F)(F)C1=CC(=CC(=C1)[B](C1=CC(=CC(=C1)C(F)(F)F)C(F)(F)F)(C1=CC(=CC(=C1)C(F)(F)F)C(F)(F)F)C1=CC(=CC(=C1)C(F)(F)F)C(F)(F)F)C(F)(F)F'
    # smarts = '[Ag+].[O-]N(=O)=O'
    # mol = Chem.MolFromSmarts(smarts)
    # mol = Chem.MolFromSmiles(smarts)
    smiles = smarts
    # smiles = 'CC[N]1(CC2=CC=CC3=[N]2[Cu+2]1([Cl-])([N](C4=CC=CC=C4)=N3)[Cl-])CC'
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol,
                     Chem.SanitizeFlags.SANITIZE_FINDRADICALS | Chem.SanitizeFlags.SANITIZE_KEKULIZE | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                     catchErrors=True)

    # mol = neutralize_atoms(mol)
    img = Draw.MolToImage(mol)
    plt.imshow(img)
    plt.show()


    #
    # mol2 = mol
    # neutralize_atoms(mol2)
    # # from rdkit.Chem.MolStandardize import rdMolStandardize
    # # un = rdMolStandardize.Uncharger()
    # # mol2 = un.uncharge(mol)
    #     # print(Chem.MolToSmiles(mol2))
    #
    # # smiles = Chem.MolFromSmiles(mol)
    # # mol2 = Chem.MolFromSmiles(smiles)
    # img = Draw.MolToImage(mol2)
    # plt.imshow(img)
    # plt.show()


def sketch5():
    # make pandas dataframe from one column, which is integers from 0 to 10
    df = pd.DataFrame({'a': np.arange(10)})
    # iterate over rows
    for i, row in df.iterrows():
        print(row)
        if row['a'] < 5:
            print('< 5')

        # change the value of next row
        df.loc[i + 1, 'a'] = 100


def sketch6():
    suzuki = 'B(O)(O)([#6:1]).Cl[#6:2]>>[#6:1]-[#6:2]'
    suzuki_back = '[#6:1]-[#6:2]>>B(O)(O)([#6:1]).Cl[#6:2]'
    rea1 = '[CH1:1][OH:2].[OH][C:3]=[O:4]>>[C:1][O:2][C:3]=[O:4]'
    alcohol1 = Chem.MolFromSmiles('CC(CCN)O')
    alcohol2 = Chem.MolFromSmiles('C[C@H](CCN)O')
    alcohol3 = Chem.MolFromSmiles('C[C@@H](CCN)O')
    acid = Chem.MolFromSmiles('CC(=O)O')

    a = Chem.MolFromSmarts('B(O)(O)C-C=C')
    b = Chem.MolFromSmarts('C#C-Cl')
    rxn = AllChem.ReactionFromSmarts(suzuki)
    ps = rxn.RunReactants((a, b))

    # rxn = AllChem.ReactionFromSmarts(suzuki_back)
    # # prod = Chem.MolFromSmiles('C#CCC=C')
    # prod = Chem.MolFromSmiles('C1=C\CCCCCCCCCCCC/1')
    # ps = rxn.RunReactants((prod,))
    # ps = rxn.RunReactants((alcohol1, acid))
    for p in ps:
        print([Chem.MolToSmiles(x, True) for x in p])

    #
    # a, b = ps[0]
    # rxn = AllChem.ReactionFromSmarts(suzuki)
    # ps = rxn.RunReactants((a, b))
    # for p in ps:
    #     print([Chem.MolToSmiles(x, True) for x in p])



def sketch7():
    df = pd.read_csv('D:/Docs/Dropbox/Lab/catalyst-coevolution/unique_halides_and_bb_vacuum.csv')
    df = df[df['npa'].notna()]
    # sort df by npa
    df = df.sort_values(by=['npa'], ascending=False)
    legends = ([f'Charge {x:.4f}' for x in (df['npa'].to_list())])
    opts = Draw.MolDrawOptions()
    opts.legendFraction = 0.25
    opts.legendFontSize = 25
    img = Draw.MolsToGridImage([Chem.MolFromSmiles(smiles) for smiles in df['smiles'].to_list()],
                         subImgSize=(450, 450),
                         legends=legends,
                         molsPerRow=4, drawOptions=opts)
    # png = img.GetDrawingText()
    # with open('figures/NPA_charges_in_DMSO_cc-pVTZbasis.png', 'wb') as f:
    #     f.write(png)
    img.save('figures/NPA_charges_in_vacuum_cc-pVTZbasis.png', "PNG")


def sketch8():
    df = pd.read_csv('D:/Docs/Dropbox/Lab/catalyst-coevolution/unique_halides_and_bb_npa.csv')
    from halides_selection import fix_halide_smiles
    df1 = fix_halide_smiles(df, colname='smiles')
    # sort by length of 'smiles' string
    df1['smiles_len'] = df1['smiles'].apply(lambda x: len(x))
    df1 = df1.sort_values(by=['smiles_len'], ascending=False)
    df1 = df1.drop(columns=['smiles_len'])

    return df, df1


def sketch9():
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Cone(x=[1], y=[1], z=[1], u=[1], v=[1], w=[0]))
    fig.add_trace(go.Cone(x=[2, ] * 3, opacity=0.3, name="opacity:0.3"))
    fig.update_layout(scene_camera_eye=dict(x=-0.76, y=1.8, z=0.92))

    fig.show()


def add_arrow(fig, xyz_from, xyz_to, color='orange', colorscale='Oranges'):
    fig.add_trace(go.Cone(x=[xyz_to[0]], y=[xyz_to[1]], z=[xyz_to[2]],
                          u=[xyz_to[0] - xyz_from[0]], v=[xyz_to[1] - xyz_from[1]], w=[xyz_to[2] - xyz_from[2]],
                          showscale=False,
                          anchor='tip',
                          colorscale=colorscale,
                          sizemode="absolute",
                          sizeref=0.5,
                          opacity=0.8,
                          name="",
                          ))
    # add line
    fig.add_trace(go.Scatter3d(x=[xyz_from[0], xyz_to[0]], y=[xyz_from[1], xyz_to[1]], z=[xyz_from[2], xyz_to[2]],
                                 mode='lines',
                                    line=dict(color=color, width=5),
                                    name="",
                                    ))


def sketch10():
    fig = go.Figure(data=go.Cone(x=[1], y=[1], z=[1], u=[1], v=[1], w=[0]))
    # fig.add_trace(go.Cone(x=[1], y=[1], z=[1], u=[0], v=[1], w=[0]))

    add_arrow(fig, [1, 1, 1], [5, 5, 5])
    fig.update_layout(scene_camera_eye=dict(x=-0.76, y=1.8, z=0.92))

    fig.show()


def sketch11():
    import skunk
    from matplotlib.offsetbox import AnnotationBbox
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    x = np.linspace(0, 2 * np.pi)
    ax.plot(x, np.sin(x))

    # # new code: using skunk box with id sk1
    # box = skunk.Box(50, 50, 'sk1')
    # ab = AnnotationBbox(box, (np.pi / 2, 1),
    #                     xybox=(-5, -100),
    #                     xycoords='data',
    #                     boxcoords='offset points',
    #                     arrowprops=dict(arrowstyle="->"))
    # ax.add_artist(ab)

    # sknunk box with id sk2
    box = skunk.Box(50, 50, 'sk2')
    ab = AnnotationBbox(box, (3 * np.pi / 2, -1),
                        xybox=(-5, 100),
                        xycoords='data',
                        boxcoords='offset points',
                        arrowprops=dict(arrowstyle="->"))

    ax.add_artist(ab)

    # insert current figure into itself at sk1
    # insert svg file in sk2
    svg = skunk.insert(
        {
            'sk1': skunk.pltsvg(),
            'sk2': 'skunk.svg'
        })

    # write to file
    with open('replaced2.svg', 'w') as f:
        f.write(svg)
    # or in jupyter notebook
    skunk.display(svg)



if __name__ == '__main__':
    # sketch11()

    # k = 4
    #
    # # Initialize medoids randomly
    # initial_medoids = np.random.choice(len(fingerprints), k, replace=False).tolist()
    # print('initial randomized %i medoids: ' % k)
    # print(initial_medoids)
    #
    # # Apply k-medoid clustering
    # kmedoids_instance = KMedoids(jaccard_distances, initial_medoids, data_type='distance_matrix')
    # kmedoids_instance.process()
    # clusters = kmedoids_instance.get_clusters()
    # medoids = kmedoids_instance.get_medoids()

