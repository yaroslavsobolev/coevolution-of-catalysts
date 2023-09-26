import pandas as pd
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from icecream import ic

def make_sure_that_all_smiles_contain_intended_pattern(parquet_file='data/DNP_SMILES.parquet',
                                                       minimal_molecular_pattern='[CH3]-[CH](-C=[C,c])-[CH](-[C,c])-O',
                                                       backup_general_pattern='C-C(-C=C)-C(-*)-O',
                                                       folder_for_output_images="figures/molecule_images/"):
    df = pd.read_parquet(parquet_file)
    df = df[df['synthesizable'] == 1]

    mol_minimum = Chem.MolFromSmarts(minimal_molecular_pattern)
    img = Draw.MolToImage(mol_minimum)
    plt.imshow(img)
    plt.show()

    for i, row in df.iterrows():
        smi = row['SMILES']
        ic(i)
        ic(smi)
        mol = Chem.MolFromSmiles(smi)
        if mol.HasSubstructMatch(mol_minimum):
            # patt = mol_minimum
            # ic(len(mol.GetSubstructMatches(patt)))
            #
            # hit_ats = list(mol.GetSubstructMatch(patt))
            # hit_bonds = []
            # for bond in patt.GetBonds():
            #     aid1 = hit_ats[bond.GetBeginAtomIdx()]
            #     aid2 = hit_ats[bond.GetEndAtomIdx()]
            #     hit_bonds.append(mol.GetBondBetweenAtoms(aid1, aid2).GetIdx())
            # d = rdMolDraw2D.MolDraw2DCairo(500, 500)  # or MolDraw2DCairo to get PNGs
            #
            # from rdkit.Chem import rdCoordGen
            #
            # rdCoordGen.AddCoords(mol)
            #
            # rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=hit_ats,
            #                                    highlightBonds=hit_bonds)
            # png = d.GetDrawingText()
            # with open(f'figures/molecule_images/{i:04d}.png', 'wb') as f:
            #     f.write(png)
            continue

        patt = Chem.MolFromSmarts(backup_general_pattern)
        ic(len(mol.GetSubstructMatches(patt)))

        hit_ats = list(mol.GetSubstructMatch(patt))
        hit_bonds = []
        for bond in patt.GetBonds():
           aid1 = hit_ats[bond.GetBeginAtomIdx()]
           aid2 = hit_ats[bond.GetEndAtomIdx()]
           hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
        d = rdMolDraw2D.MolDraw2DCairo(500, 500)

        from rdkit.Chem import rdCoordGen
        rdCoordGen.AddCoords(mol)

        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=hit_ats,
                                           highlightBonds=hit_bonds)
        png = d.GetDrawingText()
        with open(f'{folder_for_output_images}{i:04d}.png', 'wb') as f:
            f.write(png)

        print('not OK')
        break

if __name__ == '__main__':
    make_sure_that_all_smiles_contain_intended_pattern()