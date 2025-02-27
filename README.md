# Coevolution of catalysts

Various illustration and computational analogies for a research article about
optimization of organic chemistry catalysts using the coevolution approach:

(article data to be added)

## Requirements

- rdkit 2020.03.1
- matplotlib 3.1.0
- scikit-learn 1.0.2
- cairosvg 2.7.1
- skunk 1.3.0
- icecream 2.1.3
- map4 (MinHashed Atom-Pair)
- tmap 1.0.6
- tqdm 4.66.1

## Reproducing figures from the article

To reproduce the Figure 2B, run the `Figure_2B.py` script.

To reproduce the Figure 4A, run the `Figure_4A.py` script.

To reproduce the Figure 4B, run the `halides_clustering.py` script.

To reproduce the ligands UMAP, run the `ligands_mapping_mainfig.py` script.

For interactive TSNE map, download the `/interactive_HTML_embedding/` folder and open the `DNP_TSNE_ECFP6.html` file in a browser.

Natural charges calculatd with density functional theory (DFT) are provided in the `data/qm/unique_halides_and_bb_vacuum.csv` file.
Run the `qm_nbo_analysis.py` script to reproduce the plot of the natural charges.