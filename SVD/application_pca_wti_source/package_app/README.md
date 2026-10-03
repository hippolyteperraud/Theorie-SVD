# Application SVD/PCA - Courbe a terme WTI (niveau/pente/courbure)

## Contenu
- application_pca_wti.tex : rapport LaTeX (compiler avec pdflatex, 2 passes pour les renvois)
- figures/ : figures PDF utilisees dans le rapport (deja generees)
- data/wti_curve_panel.csv : panel reel EIA (contrats WTI C1-C4, mensuel, 1985-01 a 2024-04)
- data/pca_scores.csv, pca_summary.csv, pca_loadings.csv : resultats de l'analyse SVD
- code/prepare_data.py : fusionne les 4 series brutes EIA en un panel propre
- code/pca_analysis.py : calcule la SVD, exporte les resultats et regenere les figures

## Source des donnees
U.S. Energy Information Administration (EIA), NYMEX Futures Prices, contrats
Cushing OK Crude Oil Future Contract 1-4 (series RCLC1-RCLC4), frequence mensuelle.
https://www.eia.gov/dnav/pet/pet_pri_fut_s1_m.htm
(Donnees publiques, arretees par l'EIA apres le 5 avril 2024.)

## Reproduire l'analyse
    cd code
    python3 prepare_data.py      # regenere data/wti_curve_panel.csv
    python3 pca_analysis.py      # relance la SVD, regenere les figures et CSV
