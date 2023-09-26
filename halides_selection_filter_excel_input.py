import pandas as pd

# load XLSX file into Pandas DataFrame
number_of_rows = 2324
df = pd.read_excel('data/DNP_SMILES.xlsx', nrows=number_of_rows)

# make a column with sum over columns like 'BB1' to "BB16'
df['BB_count'] = df.loc[:, 'BB1':'BB16'].sum(axis=1)

# rename columns
df.rename(columns={'Chemical Name': 'name'}, inplace=True)
df.rename(columns={'Molecular Formula': 'formula'}, inplace=True)
df.rename(columns={'Whole molecule SMILES': 'SMILES'}, inplace=True)

# drop columns containing "Unnamed"
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# cleanup some SMILES
df.loc[514, 'SMILES'] = df.loc[514, 'SMILES'].split(' ')[-1]
# add missing '1' to SMILES
df.loc[1073, 'SMILES'] = df.loc[1073, 'SMILES'] + '1'

# save excel cell color in first column into a separate column
from openpyxl import load_workbook
excel_file = 'data/DNP_SMILES.xlsx'
wb = load_workbook(excel_file, data_only = True)
sh = wb['DNP Export']
for row_id in range(2, number_of_rows+2):
    color_in_hex = sh[f'A{row_id}'].fill.start_color.index # this gives you Hexadecimal value of the color
    df.loc[row_id - 2, 'synthesizable'] = 0 if color_in_hex == 'FFC00000' else 1

# drop row with index 880
df = df.drop([880])
df = df.reset_index(drop=True)

# save to parquet
df.to_parquet('data/DNP_SMILES.parquet')