import matplotlib.pyplot as plt
import pandas as pd

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

data = {
"B3LYP\n3-21G" : -0.33835,
"ωB97X-D\ncc-pVDZ" : -0.32900,
"ωB97X-D\naug-cc-pVDZ" : -0.33917,
"ωB97X-D\ncc-pVTZ" : -0.28184,
"ωB97X-D\naug-cc-pVTZ" : -0.28890,
"ωB97X-D\ncc-pVQZ" : -0.28527,
"MP2\naug-cc-pVTZ" : -0.27794,
"CCSD(T)\naug-cc-pVTZ" : -0.27794
}

# make a bar plot with the data, keys are x axis labels
fig1, ax = plt.subplots(figsize=(9, 3.5))
simpleaxis(ax)
plt.bar(range(len(data)), list(data.values()), align='center', color='C1')
plt.xticks(range(len(data)), list(data.keys()))
plt.ylabel('NPA charge on the carbon bonded to bromine, e')

# plt.title('NPA charge on carbon closest to bromine in halide D')
plt.gca().invert_yaxis()
plt.ylim(-0.26, -0.36)
plt.tight_layout()

# annotate last bar with 'ground truth' text.
plt.annotate('Maximum accuracy.\nSlowest calculations.', xy=(7, -0.29486), xytext=(5.8, -0.32), ha='center',
                arrowprops=dict(facecolor='black', shrink=0.02, alpha=0.3),
                )

plt.annotate('Method we use', xy=(3, -0.29486), xytext=(3.2, -0.33), ha='center',
                arrowprops=dict(facecolor='black', shrink=0.02, alpha=0.3),
                )

fig1.savefig('figures/npa/npa_benchmarks_.png', dpi=300)
plt.show()

# make dataframe with four columns: halide, water, DMSO, vacuum
df = pd.DataFrame(columns=['vacuum', 'water', 'DMSO'])
# add data to dataframe
df.loc['Halide A'] = [-0.120, -0.07776, -0.07764]
df.loc['Halide B'] = [-0.259, -0.20867, -0.20827]
df.loc['Halide D'] = [-0.28915, -0.30084, -0.30102]

# make a bar plot with the data, keys are x axis labels, columns are bars of different color
fig2, ax2 = plt.subplots(figsize=(5, 3.5))
simpleaxis(ax2)
colors = ['grey', 'C0', 'C1']
df.plot(kind='bar', ax=plt.gca(), color=colors)
plt.xticks(rotation=0)
plt.ylabel('NPA charge on the carbon bonded to bromine, e')
plt.gca().invert_yaxis()
plt.ylim(0, -0.34)
# remove ticks on x axis
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
plt.tight_layout()
fig2.savefig('figures/npa/npa_solvents.png', dpi=300)
plt.show()

