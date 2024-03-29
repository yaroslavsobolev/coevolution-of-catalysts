from halides_selection_mainfig import *

add_BBs()
filename = 'only_ab_polyketides_fingerprints_tsne_px30_lr707_50kiter_reclassed.hdf'
plot_panelC_from_hdf(hdf_filepath=f'data/{filename}',
                     alpha_parent=0.3, suffix='_ab_only',
                     option='4A')