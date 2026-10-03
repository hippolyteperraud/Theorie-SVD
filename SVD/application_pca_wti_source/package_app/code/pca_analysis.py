import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Chargement du panel reel (EIA, 1985-01 a 2024-04, mensuel)
# ---------------------------------------------------------------
panel = pd.read_csv("/home/claude/svd_report/data/wti_curve_panel.csv", parse_dates=["date"])
maturities = ["C1", "C2", "C3", "C4"]  # contrats a 1,2,3,4 mois
X = panel[maturities].to_numpy()  # (T, 4), niveaux en $/baril
dates = panel["date"].to_numpy()
T = X.shape[0]
print(f"Panel: {T} mois, {X.shape[1]} maturites, de {panel['date'].min().date()} a {panel['date'].max().date()}")

# ---------------------------------------------------------------
# 2. Construction de la matrice des variations mensuelles
#    (pratique standard desk : PCA sur les variations, pas les niveaux)
# ---------------------------------------------------------------
dX = np.diff(X, axis=0)               # (T-1, 4)
dX_mean = dX.mean(axis=0)
dX_centered = dX - dX_mean            # centrage (colonnes de moyenne nulle)

# ---------------------------------------------------------------
# 3. SVD de la matrice centree des variations
#    dX_centered = U * Sigma * V^T
#    Les colonnes de V sont les facteurs de risque de la courbe
# ---------------------------------------------------------------
U, S, Vt = np.linalg.svd(dX_centered, full_matrices=False)
V = Vt.T                              # (4,4) : chaque colonne = un mode

explained_var = (S**2) / np.sum(S**2)
cum_var = np.cumsum(explained_var)

print("\nValeurs singulieres :", np.round(S, 3))
print("Variance expliquee (%) :", np.round(100 * explained_var, 2))
print("Variance cumulee (%)   :", np.round(100 * cum_var, 2))

# ---------------------------------------------------------------
# 4. Normalisation des signes pour l'interpretation
#    (une SVD ne fixe pas le signe des vecteurs -> convention lisible)
# ---------------------------------------------------------------
if V[:, 0].mean() < 0:
    V[:, 0] *= -1; U[:, 0] *= -1
if V[-1, 1] < V[0, 1]:
    V[:, 1] *= -1; U[:, 1] *= -1
if V[1, 2] > 0:
    V[:, 2] *= -1; U[:, 2] *= -1

print("\nChargements (loadings) des 3 premiers facteurs :")
loadings_df = pd.DataFrame(V[:, :3], index=maturities,
                            columns=["Niveau (PC1)", "Pente (PC2)", "Courbure (PC3)"])
print(loadings_df.round(3))

# ---------------------------------------------------------------
# 5. Scores des facteurs dans le temps (projection des variations)
# ---------------------------------------------------------------
scores = dX_centered @ V              # (T-1, 4)
level_score = scores[:, 0]
slope_score = scores[:, 1]
curv_score = scores[:, 2]

results = pd.DataFrame({
    "date": dates[1:],
    "level": level_score,
    "slope": slope_score,
    "curvature": curv_score,
})
results.to_csv("/home/claude/svd_report/data/pca_scores.csv", index=False)

summary = pd.DataFrame({
    "singular_value": S,
    "explained_var_pct": 100 * explained_var,
    "cum_var_pct": 100 * cum_var,
})
summary.to_csv("/home/claude/svd_report/data/pca_summary.csv", index=False)
loadings_df.to_csv("/home/claude/svd_report/data/pca_loadings.csv")

# =================================================================
# FIGURE A : chargements niveau / pente / courbure vs maturite
# =================================================================
fig, ax = plt.subplots(figsize=(5.5, 4))
mats = [1, 2, 3, 4]
colors = ["#1f4e79", "#c0392b", "#277a3c"]
labels = ["Niveau (PC1)", "Pente (PC2)", "Courbure (PC3)"]
for i in range(3):
    ax.plot(mats, V[:, i], marker="o", color=colors[i], label=labels[i])
ax.axhline(0, color="gray", lw=0.8)
ax.set_xticks(mats)
ax.set_xlabel("Contrat (mois de maturite)")
ax.set_ylabel("Chargement (loading)")
ax.set_title("Facteurs de risque de la courbe WTI (via SVD)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("/home/claude/svd_report/figures/pca_loadings.pdf", bbox_inches="tight")
plt.close(fig)

# =================================================================
# FIGURE B : variance expliquee (scree plot)
# =================================================================
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.bar([1, 2, 3, 4], explained_var * 100, color="#1f4e79", alpha=0.8, label="par facteur")
ax.plot([1, 2, 3, 4], cum_var * 100, color="#c0392b", marker="o", label="cumulee")
ax.set_xticks([1, 2, 3, 4])
ax.set_xlabel("Facteur (composante)")
ax.set_ylabel("Variance expliquee (%)")
ax.set_ylim(0, 105)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("/home/claude/svd_report/figures/pca_variance.pdf", bbox_inches="tight")
plt.close(fig)

# =================================================================
# FIGURE C : courbe moyenne + courbe deformee par +/- 1 sigma sur PC1/PC2
# =================================================================
mean_curve = X.mean(axis=0)
sigma1 = S[0] / np.sqrt(T - 1)
sigma2 = S[1] / np.sqrt(T - 1)

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(mats, mean_curve, color="black", lw=2, marker="o", label="Courbe moyenne")
ax.plot(mats, mean_curve + sigma1 * V[:, 0], color=colors[0], ls="--", label="+1$\\sigma$ choc de niveau")
ax.plot(mats, mean_curve - sigma1 * V[:, 0], color=colors[0], ls=":", label="-1$\\sigma$ choc de niveau")
ax.plot(mats, mean_curve + sigma2 * V[:, 1], color=colors[1], ls="--", label="+1$\\sigma$ choc de pente")
ax.set_xticks(mats)
ax.set_xlabel("Contrat (mois de maturite)")
ax.set_ylabel("Prix (\\$/baril)")
ax.set_title("Effet des chocs de niveau et de pente sur la courbe")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("/home/claude/svd_report/figures/pca_curve_shocks.pdf", bbox_inches="tight")
plt.close(fig)

# =================================================================
# FIGURE D : ecart Contrat4 - Contrat1 dans le temps
#   (mesure directe, en $/baril, du contango (>0) / backwardation (<0),
#    a comparer avec le facteur de pente de la figure E)
# =================================================================
spread_41 = X[:, 3] - X[:, 0]
fig, ax1 = plt.subplots(figsize=(7.5, 3.6))
ax1.plot(panel["date"], spread_41, color="#1f4e79", lw=1.0)
ax1.axhline(0, color="gray", lw=0.8)
ax1.fill_between(panel["date"], spread_41, 0, where=(spread_41 >= 0),
                  color="#1f4e79", alpha=0.15, label="Contango (C4>C1)")
ax1.fill_between(panel["date"], spread_41, 0, where=(spread_41 < 0),
                  color="#c0392b", alpha=0.15, label="Backwardation (C4<C1)")
ax1.set_ylabel("Contrat 4 $-$ Contrat 1 (\\$/baril)")
ax1.set_title("Pente de la courbe WTI en niveau : contango vs backwardation")
ax1.legend(fontsize=8, loc="lower left")
ax1.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("/home/claude/svd_report/figures/pca_level_vs_price.pdf", bbox_inches="tight")
plt.close(fig)

# =================================================================
# FIGURE E : facteur de pente dans le temps (contango / backwardation)
# =================================================================
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.plot(results["date"], results["slope"], color="#c0392b", lw=1.0)
ax.axhline(0, color="gray", lw=0.8)
ax.set_ylabel("Score du facteur pente")
ax.set_title("Facteur de pente (chocs mensuels) de la courbe WTI, 1985-2024")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("/home/claude/svd_report/figures/pca_slope_timeseries.pdf", bbox_inches="tight")
plt.close(fig)

print("\nFigures generees.")
