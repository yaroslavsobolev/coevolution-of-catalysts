# checking query interface
import pandas as pd
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from icecream import ic

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

# # example_smiles = 'CCC/C=C/C(C(C(/C=C/C(C)C)OC(CC(C(CCC=C)=C)C)=O)O)C'
# # example_smiles = 'CCC/C=C/C(C(C(/C=C/C(C)C)OC(CC(C(CCC=C)=C)C)=O)OC)C'
# # example_smiles = 'CCC/C=C/C(C(C(/C=C/C(C)C)OC(CC(C(CCC=C)=C)C)=O)O)CC'
# example_smiles = 'COC1CC(C)Cc2c(OC)c(O)cc3NC(=O)\C(C)=C\CCC(OC)C(O)\C(C)=C\C(C)C1Oc23'
# example_mol = Chem.MolFromSmiles(example_smiles)
#
# mol_minimum = Chem.MolFromSmarts('[CH3]-[CH](-[CH]=C)-[CH](-C)-[OH1]')
#
#
# img = Draw.MolToImage(mol)
# plt.imshow(img)
# plt.show()
# # ps = Chem.AdjustQueryParameters()
# # ps.makeDummiesQueries=True
# # mol = Chem.AdjustQueryProperties(mol, ps)
#
# # check that mol has substruct
# ic(example_mol.HasSubstructMatch(mol))



####################### Figuring out the entry 7
# example_smiles = 'COC1CC(Cc2c3c(NC(/C(C)=C/CCC(C(/C(C)=C/C(C1O3)C)O)OC)=O)cc(O)c2OC)C'
# # example_smiles = 'CCC/C=C/C(C(C(/C=C/C(C)C)OC(CC(C(CCC=C)=C)C)=O)OC)C'
# # example_smiles = 'CCC/C=C/C(C(C(/C=C/C(C)C)OC(CC(C(CCC=C)=C)C)=O)O)CC'
# # example_smiles = 'COC1CC(C)Cc2c(OC)c(O)cc3NC(=O)\C(C)=C\CCC(OC)C(O)\C(C)=C\C(C)C1Oc23'
# example_mol = Chem.MolFromSmiles(example_smiles)
# # mol = Chem.MolFromSmarts('C-C(-C=C)-C(-*)-O')
#
# ic(example_mol.HasSubstructMatch(mol))
# ic(len(example_mol.GetSubstructMatches(mol)))
#
# img = Draw.MolToImage(mol)
# plt.imshow(img)
# plt.show()
# # ps = Chem.AdjustQueryParameters()
# # ps.makeDummiesQueries=True
# # mol = Chem.AdjustQueryProperties(mol, ps)

