import pandas as pd
import logging

# set logging
logging.basicConfig(level=logging.INFO)

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
        '[H]C1=CC(C2=CC=CC=C2)=C(C=C1)P3[C@@]4(C)O[C@]5(C)O[C@](O[C@@]3(C)C5)(C)C4': 0.001,
        '[H]C1=CC(CC2=CC=CC=C2)=C(C=C1)P3[C@@]4(C)O[C@]5(C)O[C@](O[C@@]3(C)C5)(C)C4': 34.1,
        'FC(C1=CC(OC(C)C)=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3)(F)F': 24.2,
        'FC(C1=CC=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3)(F)F': 16.6
        }

smiles_id_A = {
    'C/C=C(P(C1=CC=CC=C1)C2=CC=CC=C2)\C=C' : 2,
    'CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C' : 3,
    '[2H]C(C1=CC=CC=C1P(C2=CC=CC=C2C([2H])([2H])[2H])C3=CC=CC=C3C([2H])([2H])[2H])([2H])[2H]': 4,
    'CCC1=CC=CC=C1P(C2=CC=CC=C2CC)C3=CC=CC=C3CC': 5,
    'CCCC1=CC=CC=C1P(C2=CC=CC=C2CCC)C3=CC=CC=C3CCC': 6,
    'CC(C)C1=CC=CC=C1P(C2=CC=CC=C2C(C)C)C3=CC=CC=C3C(C)C': 7,
    'C1(C2CCCC2)=CC=CC=C1P(C3=CC=CC=C3C4CCCC4)C5=CC=CC=C5C6CCCC6': 8,
    'CC1=CC(C(F)(F)F)=CC=C1P(C2=CC=C(C(F)(F)F)C=C2C)C3=CC=C(C(F)(F)F)C=C3C': 9,
    'CC1=CC(OC)=CC=C1P(C2=CC=C(OC)C=C2C)C3=CC=C(OC)C=C3C': 10,
    'CC1=CC(N(C)C)=CC=C1P(C2=CC=C(N(C)C)C=C2C)C3=CC=C(N(C)C)C=C3C': 11,
    'C1(CC2=CC=CC=C2)=CC=CC=C1P(C3=CC=CC=C3CC4=CC=CC=C4)C5=CC=CC=C5CC6=CC=CC=C6': 12,
    'C1([As](C2=CC=CC=C2)C3=CC=CC=C3)=CC=CC=C1': 13,
    'C1(P(C2CCCCC2)C3CCCCC3)CCCCC1': 14,
    'C1(C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)=CC=CC=C1': 15,
    'CC(C1=CC(C(C)C)=CC(C(C)C)=C1C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C': 16,
    'COC1=CC=CC(OC)=C1C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4': 17,
    'CC(C1=CC(C(C)C)=CC(C(C)C)=C1C2=C(OC)C=CC(OC)=C2P(C3=CC(C(F)(F)F)=CC(C(F)(F)F)=C3)C4=CC(C(F)(F)F)=CC(C(F)(F)F)=C4)C': 18,
    'CC(P(C(C)(C)C)C1=C(OC)C=CC(OC)=C1C2=C(C(C)C)C=C(C(C)C)C=C2C(C)C)(C)C': 19,
    'CC(OC1=CC=CC(OC(C)C)=C1C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C': 1,
    'COC1=CC=CC(OC)=C1C2=CC=CC=C2P(C3=CC=CC=C3)C4=CC=CC=C4': 20
}

smiles_id_B = {'CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C': 3,
        'CC1=CC(N(C)C)=CC=C1P(C2=CC=C(N(C)C)C=C2C)C3=CC=C(N(C)C)C=C3C': 11,
        'CC1=CC(OC)=CC=C1P(C2=CC=C(OC)C=C2C)C3=CC=C(OC)C=C3C': 10,
        'CC(C)OC1=CC=C(P(C2=CC=C(OC(C)C)C=C2C)C3=CC=C(OC(C)C)C=C3C)C(C)=C1': 21,
        'CC1=CC(C(F)(F)F)=CC=C1P(C2=CC=C(C(F)(F)F)C=C2C)C3=CC=C(C(F)(F)F)C=C3C': 9,
        'FC(C=C1C)=CC=C1P(C2=CC=C(F)C=C2C)C3=CC=C(F)C=C3C': 22,
        'FC1=CC=C(C)C(P(C2=CC(F)=CC=C2C)C3=CC(F)=CC=C3C)=C1': 23,
        'C1(CC2=CC=CC=C2)=CC=CC=C1P(C3=CC=CC=C3CC4=CC=CC=C4)C5=CC=CC=C5CC6=CC=CC=C6': 12,
        'CCC1=CC=CC=C1P(C2=CC=CC=C2CC)C3=CC=CC=C3CC': 3,
        'FCC1=CC=CC=C1P(C2=CC=CC=C2CF)C3=CC=CC=C3CF': 24,
        'C1(P(C2=CC=CC=C2C3=CC=CC=C3)C4=CC=CC=C4)=CC=CC=C1': 25,
        'CC1=CC=CC=C1P(C2=CC=CC=C2C3=CC=CC=C3)C4=CC=CC=C4C': 26,
        'CC1=CC=CC=C1P(C2=CC=CC=C2CC3=CC=CC=C3)C4=CC=CC=C4C': 27,
        'COC(C=CC=C1)=C1C2=CC=CC=C2P(C3=CC=CC=C3C4=C(OC)C=CC=C4)C5=CC=CC=C5C6=C(OC)C=CC=C6': 28,
        'C1(C2=C(C=CC=C3)C3=CC=C2)=CC=CC=C1P(C4=CC=CC=C4C5=C(C=CC=C6)C6=CC=C5)C7=CC=CC=C7C8=C(C=CC=C9)C9=CC=C8': 29,
        'CC(OC1=CC=CC(OC(C)C)=C1C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C': 16,
        'COC1=CC=CC(OC)=C1C2=CC=CC=C2P(C3=CC=CC=C3)C4=CC=CC=C4': 20,
        'COC1=CC=CC(OC)=C1C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4': 17,
        'COC1=CC(OC)=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3': 30,
        'COC1=CC(OC(C)C)=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3': 31
        }

smiles_id_C = {'COC1=CC(OC)=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3': 30,
        'COC1=CC(OC(C)C)=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3': 31,
        'COC1=CC(OC2CCCC2)=C(C=C1)P3[C@@]4(C)O[C@]5(C)O[C@](O[C@@]3(C)C5)(C)C4': 32,
        'COC1=CC(OCC2=CC=CC=C2)=C(C=C1)P3[C@@]4(C)O[C@]5(C)O[C@](O[C@@]3(C)C5)(C)C4': 33,
        'COC1=CC(C)=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3': 34,
        '[H]C1=CC(C)=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3': 35,
        '[H]C1=CC(C2=CC=CC=C2)=C(C=C1)P3[C@@]4(C)O[C@]5(C)O[C@](O[C@@]3(C)C5)(C)C4': 36,
        '[H]C1=CC(CC2=CC=CC=C2)=C(C=C1)P3[C@@]4(C)O[C@]5(C)O[C@](O[C@@]3(C)C5)(C)C4': 37,
        'FC(C1=CC(OC(C)C)=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3)(F)F': 38,
        'FC(C1=CC=C(C=C1)P2[C@@]3(C)O[C@]4(C)O[C@](O[C@@]2(C)C4)(C)C3)(F)F': 39
        }

joined_set = {}
for key in setA:
    joined_set[key] = [setA[key], 0, 0]
for key in setB:
    if key in joined_set:
        logging.debug('key repeats')
        joined_set[key][1] = setB[key]
    else:
        joined_set[key] = [0, setB[key], 0]
for key in setC:
    if key in joined_set:
        logging.debug('key from C repeats')
        joined_set[key][2] = setC[key]
    else:
        joined_set[key] = [0, 0, setC[key]]
logging.info(f'The length of the joined set is {len(joined_set)}')

# make dataframe from joined set
df = pd.DataFrame.from_dict(joined_set, orient='index', columns=['A', 'B', 'C'])
# separate index to separate column. Use integer index instead
df.reset_index(level=0, inplace=True)
# rename index column to 'SMILES'
df.rename(columns={'index': 'SMILES'}, inplace=True)

# Search for id number for each smiles successively in smiles_id_A, smiles_id_B, smiles_id_C. Try next if previous fails.
# If all fail, assign id number 0
df['id'] = 0
for i in range(len(df)):
    smiles = df.loc[i, 'SMILES']
    if smiles in smiles_id_A:
        df.loc[i, 'id'] = smiles_id_A[smiles]
        logging.info(f'Found id {smiles_id_A[smiles]} for {smiles} in set A')
    elif smiles in smiles_id_B:
        df.loc[i, 'id'] = smiles_id_B[smiles]
        logging.info(f'Found id {smiles_id_B[smiles]} for {smiles} in set B')
    elif smiles in smiles_id_C:
        df.loc[i, 'id'] = smiles_id_C[smiles]
        logging.info(f'Found id {smiles_id_C[smiles]} for {smiles} in set C')
    else:
        print(f'No id found for {smiles}')

# df.to_parquet('data/ligands.parquet')
#
# df2 = pd.read_parquet('data/ligands.parquet')
df.to_csv('data/ligands.csv', index=False)
