import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

setA = {
    'C/C=C(P(C1=CC=CC=C1)C2=CC=CC=C2)\C=C' : 2.5,
    'CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C' : 65,
    '[2H]C(C1=CC=CC=C1P(C2=CC=CC=C2C([2H])([2H])[2H])C3=CC=CC=C3C([2H])([2H])[2H])([2H])[2H]': 60,
    'CCC1=CC=CC=C1P(C2=CC=CC=C2CC)C3=CC=CC=C3CC': 51,
    'CCCC1=CC=CC=C1P(C2=CC=CC=C2CCC)C3=CC=CC=C3CCC': 17,
    'CC(C)C1=CC=CC=C1P(C2=CC=CC=C2C(C)C)C3=CC=CC=C3C(C)C': 22,
    'C1(C2CCCC2)=CC=CC=C1P(C3=CC=CC=C3C4CCCC4)C5=CC=CC=C5C6CCCC6': 46,
    'CC1=CC(C(F)(F)F)=CC=C1P(C2=CC=C(C(F)(F)F)C=C2C)C3=CC=C(C(F)(F)F)C=C3C': 41,
    'CC1=CC(OC)=CC=C1P(C2=CC=C(OC)C=C2C)C3=CC=C(OC)C=C3C': 83,
    'CC1=CC(N(C)C)=CC=C1P(C2=CC=C(N(C)C)C=C2C)C3=CC=C(N(C)C)C=C3C': 60,
    'C1(CC2=CC=CC=C2)=CC=CC=C1P(C3=CC=CC=C3CC4=CC=CC=C4)C5=CC=CC=C5CC6=CC=CC=C6': 55,
    'C1([As](C2=CC=CC=C2)C3=CC=CC=C3)=CC=CC=C1': 31,
    'C1(P(C2CCCCC2)C3CCCCC3)CCCCC1': 31,
    'C1(C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)=CC=CC=C1': 2,
    'CC(C1=CC(C(C)C)=CC(C(C)C)=C1C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C': 3,
    'COC1=CC=CC(OC)=C1C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4': 10.5,
    'CC(C1=CC(C(C)C)=CC(C(C)C)=C1C2=C(OC)C=CC(OC)=C2P(C3=CC(C(F)(F)F)=CC(C(F)(F)F)=C3)C4=CC(C(F)(F)F)=CC(C(F)(F)F)=C4)C': 1,
    'CC(P(C(C)(C)C)C1=C(OC)C=CC(OC)=C1C2=C(C(C)C)C=C(C(C)C)C=C2C(C)C)(C)C': 2,
    'CC(OC1=CC=CC(OC(C)C)=C1C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C': 11.5,
    'COC1=CC=CC(OC)=C1C2=CC=CC=C2P(C3=CC=CC=C3)C4=CC=CC=C4': 79
}

setB = {'CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C': 23.6,
        'CC1=CC(N(C)C)=CC=C1P(C2=CC=C(N(C)C)C=C2C)C3=CC=C(N(C)C)C=C3C': 16.6,
        'CC1=CC(OC)=CC=C1P(C2=CC=C(OC)C=C2C)C3=CC=C(OC)C=C3C': 18.4,
        'CC(C)OC1=CC=C(P(C2=CC=C(OC(C)C)C=C2C)C3=CC=C(OC(C)C)C=C3C)C(C)=C1': 28.8,
        'CC1=CC(C(F)(F)F)=CC=C1P(C2=CC=C(C(F)(F)F)C=C2C)C3=CC=C(C(F)(F)F)C=C3C': 9.1,
        'FC(C=C1C)=CC=C1P(C2=CC=C(F)C=C2C)C3=CC=C(F)C=C3C': 27.9,
        'FC1=CC=C(C)C(P(C2=CC(F)=CC=C2C)C3=CC(F)=CC=C3C)=C1': 6.4,
        'C1(CC2=CC=CC=C2)=CC=CC=C1P(C3=CC=CC=C3CC4=CC=CC=C4)C5=CC=CC=C5CC6=CC=CC=C6': 9.3,
        'CCC1=CC=CC=C1P(C2=CC=CC=C2CC)C3=CC=CC=C3CC': 18.1,
        'FCC1=CC=CC=C1P(C2=CC=CC=C2CF)C3=CC=CC=C3CF': 21.3,
        'C1(P(C2=CC=CC=C2C3=CC=CC=C3)C4=CC=CC=C4)=CC=CC=C1': 13.7,
        'CC1=CC=CC=C1P(C2=CC=CC=C2C3=CC=CC=C3)C4=CC=CC=C4C': 5.8,
        'CC1=CC=CC=C1P(C2=CC=CC=C2CC3=CC=CC=C3)C4=CC=CC=C4C': 34.3,
        'COC(C=CC=C1)=C1C2=CC=CC=C2P(C3=CC=CC=C3C4=C(OC)C=CC=C4)C5=CC=CC=C5C6=C(OC)C=CC=C6': 22.3,
        'C1(C2=C(C=CC=C3)C3=CC=C2)=CC=CC=C1P(C4=CC=CC=C4C5=C(C=CC=C6)C6=CC=C5)C7=CC=CC=C7C8=C(C=CC=C9)C9=CC=C8': 2.4,
        'CC(OC1=CC=CC(OC(C)C)=C1C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C': 16.1,
        'COC1=CC=CC(OC)=C1C2=CC=CC=C2P(C3=CC=CC=C3)C4=CC=CC=C4': 16.8,
        'COC1=CC=CC(OC)=C1C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4': 24.5,
        'COC1=CC(OC)=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3': 29,
        'COC1=CC(OC(C)C)=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3': 40.8
        }

setC = {'COC1=CC(OC)=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3': 2.5,
        'COC1=CC(OC(C)C)=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3': 24.4,
        'COC1=CC(OC2CCCC2)=C(C=C1)P3[C@@]4(C)O[C@]5(C)O[C@](O[C@@]3(C)C5)(C)C4': 29.1,
        'COC1=CC(OCC2=CC=CC=C2)=C(C=C1)P3[C@@]4(C)O[C@]5(C)O[C@](O[C@@]3(C)C5)(C)C4': 19.9,
        'COC1=CC(C)=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3': 22,
        '[H]C1=CC(C)=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3': 31.2,
        '[H]C1=CC(C2=CC=CC=C2)=C(C=C1)P3[C@@]4(C)O[C@]5(C)O[C@](O[C@@]3(C)C5)(C)C4': 0,
        '[H]C1=CC(CC2=CC=CC=C2)=C(C=C1)P3[C@@]4(C)O[C@]5(C)O[C@](O[C@@]3(C)C5)(C)C4': 34.1,
        'FC(C1=CC(OC(C)C)=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3)(F)F': 24.2,
        'FC(C1=CC=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3)(F)F': 16.6
        }

joined_set = {}
for key in setA:
    joined_set[key] = [setA[key], 0, 0]
for key in setB:
    if key in joined_set:
        print('key repeats')
        joined_set[key][1] = setB[key]
    else:
        joined_set[key] = [0, setB[key], 0]
for key in setC:
    if key in joined_set:
        print('key from C repeats')
        joined_set[key][2] = setC[key]
    else:
        joined_set[key] = [0, 0, setC[key]]

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

# proof and make a list of SMILES
# df_smiles = df_3['smiles']

# for ds in df_smiles:
#     try:
#         cs = Chem.CanonSmiles(ds)
#         c_smiles.append(cs)
#     except:
#         print('Invalid SMILES:', ds)
# print()

c_smiles = list(joined_set.keys())
# make a list of mols
ms = [Chem.MolFromSmiles(x) for x in c_smiles]

# make a list of fingerprints (fingerprint)
fps = [FingerprintMols.FingerprintMol(x, minPath=1, maxPath=9, fpSize=2048, bitsPerHash=2,
                                      useHs=True, tgtDensity=0, minSize=128) for x in ms]

# the list for the dataframe
qu, ta, sim = [], [], []

G = nx.Graph()
indices = list(range(len(fps)))
# compare all fingerprint pairwise without duplicates
tanisims = []
for n, fps1 in enumerate(fps): # -1 so the last fingerprint will not be used
    for m in indices[n+1:]:
        fps2 = fps[m]
        G.add_edges_from([(n, fps.index(fps2), {'myweight': (1/(1.001-DataStructs.TanimotoSimilarity(fps[n], fps2)))**3})])
        tanisims.append(DataStructs.TanimotoSimilarity(fps[n], fps2))
tanisims = np.array(tanisims)


pos = nx.random_layout(G)
for i in range(100):
    pos = nx.spring_layout(G, weight='myweight', pos = pos, iterations=500)

cmaps = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]
for setid in range(3):
    fig1 = plt.figure(1, figsize=(15, 15))
    nx.draw_networkx(G, pos, node_color=[joined_set[c_smiles[node]][setid]/100 for node in G.nodes()], node_size=400, cmap=cmaps[setid],
                     edge_color='white', edgecolors='black')
    plt.axis('equal')
    # sm = plt.cm.ScalarMappable(cmap=cmaps[setid], norm=plt.Normalize(vmin=100*min([joined_set[c_smiles[node]][setid]/100 for node in G.nodes()]),
    #                                                                  vmax=100*max([joined_set[c_smiles[node]][setid]/100 for node in G.nodes()])))
    # sm._A = []
    # plt.colorbar(sm, orientation='horizontal', fraction=0.02, pad=0.1, label='Yield (%)')
    fig1.savefig(f'figures/temp2/set_{setid}.png', dpi=300)
    # plt.show()
    plt.close(fig1)

dim = 15
pos = nx.spring_layout(G, weight='myweight', iterations=500, dim=dim)
for i in range(100):
    pos = nx.spring_layout(G, weight='myweight', pos=pos, iterations=500, dim=dim)

for setid in range(3):
    mean_yield = np.mean([joined_set[c_smiles[node]][setid] for node in G.nodes()])
    meanpos = np.mean(np.array([pos[node] * joined_set[c_smiles[node]][setid] for node in G.nodes()]), axis=0) / mean_yield
    weighted_standard_deviation_of_positions = np.sqrt(
        np.mean(np.array([np.linalg.norm(pos[node] - meanpos) ** 2 * joined_set[c_smiles[node]][setid] for node in G.nodes()]), axis=0) / mean_yield
    )
    print(f'set {setid}: mean yield = {mean_yield}, mean position = {meanpos},'
          f'weighted standard deviation of positions = {weighted_standard_deviation_of_positions}')
    if dim == 2:
        fig1 = plt.figure(1, figsize=(5, 5))
        nx.draw_networkx(G, pos, node_color=[joined_set[c_smiles[node]][setid]/100 for node in G.nodes()], node_size=400, cmap=cmaps[setid],
                         edge_color='white', edgecolors='black')
        plt.scatter([meanpos[0]], [meanpos[1]], s=100, c='black', marker='x')
        plt.show()
        plt.axis('equal')


# plt.savefig('test.png')