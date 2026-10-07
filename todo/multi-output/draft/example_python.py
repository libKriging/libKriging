# API attendue de MultiOutputKriging côté Python. Seul le mode "pca" est implémenté
# (testé par bindings/Python/pylibkriging/tests/MultiOutputKriging_test.py) ;
# les sections "separable" / "shared" restent prospectives. Cf. todo/multi-output/ANALYSIS.md §5, draft/MultiOutputKriging.hpp.
import numpy as np
import pylibkriging as lk

# --- Données : un code qui renvoie une courbe temporelle ----------------------
# x = (fréquence, amortissement) ∈ [0,1]², sortie = y(x, t) sur 200 pas de temps.
t = np.linspace(0.0, 10.0, 200)                       # q = 200 coordonnées de sortie


def code(x):
    f, a = 0.5 + 1.5 * x[0], 0.1 + 0.5 * x[1]
    return np.exp(-a * t) * np.cos(2 * np.pi * f * t / 5)


rng = np.random.default_rng(0)
X = rng.uniform(size=(40, 2))                         # n × d   (n=40, d=2)
Y = np.array([code(x) for x in X])                    # n × q   (lignes = observations, comme X)
Xnew = rng.uniform(size=(5, 2))                       # m × d

# --- 1. Réduction ACP : le défaut recommandé pour des courbes ----------------
pca = lk.MultiOutputKriging(
    Y, X, "matern5_2",
    output_model="pca(0.99)",     # K choisi pour 99 % de variance expliquée ; ou "pca(5)"
    regmodel="constant",
    normalize=True,               # centrage/réduction par sortie
    optim="BFGS",
    objective="LL",
)
print(pca.nb_components(), pca.pca_explained())       # ex. 6, [0.71, 0.88, …, 0.993]
print(pca.component(0).theta())                       # chaque composante a son propre θ

mean, stdev, cov, deriv = pca.predict(Xnew, return_stdev=True)
# mean, stdev : m × q  (5 × 200) — stdev inclut la variance de troncature

# --- 2. Séparable en temps : covariance temporelle cohérente -----------------
sep = lk.MultiOutputKriging(
    Y, X, "matern5_2",
    output_model="separable(matern5_2)",   # Cov = σ² R_t(t,t';φ) ⊗ r(x,x';θ)
    output_coordinates=t,                  # q (ou q × d_t) — appelle set_output_coordinates avant fit
    regmodel="constant",
    normalize=True,
)
print(sep.theta(), sep.output_theta(), sep.sigma2())  # θ (d), φ (d_t), σ² (1)

# Covariance prédictive sous forme de Kronecker, sans matrice dense mq × mq :
Cx, Sigma = sep.predictCovFactors(Xnew)               # (m × m), (q × q)
# np.kron(Sigma, Cx) = Cov(vec(Y_new)), sortie-majeure (sortie 1 aux m points, puis sortie 2…)

# Trajectoires conjointes → incertitude sur une fonctionnelle de la courbe
sims = sep.simulate(nsim=1000, seed=123, X=Xnew)      # m × q × nsim
trough = sims.min(axis=1)                             # m × nsim : creux de chaque trajectoire
print(np.quantile(trough, [0.05, 0.5, 0.95], axis=1)) # IC à 90 % du creux, par point
# (pas le pic : toutes les courbes valent 1 en t = 0, c'est leur maximum)

# --- 3. θ partagés (PP-GaSP) : rapide, bandes ponctuelles seulement ----------
shared = lk.MultiOutputKriging(Y, X, "matern5_2", output_model="shared")
print(shared.sigma2().shape, shared.beta().shape)     # (q,), (p, q)

# --- Choix du modèle par validation croisée ----------------------------------
for m in (pca, sep, shared):
    print(m.output_model(), m.leaveOneOut())

# --- Mise à jour séquentielle ------------------------------------------------
X_u = rng.uniform(size=(3, 2))
Y_u = np.array([code(x) for x in X_u])                # 3 × q, même q
sep.update(Y_u, X_u, refit=True)

# --- Cohérence : q = 1 redonne Kriging ---------------------------------------
y1 = Y[:, 50]
a = lk.MultiOutputKriging(y1[:, None], X, "matern5_2", output_model="shared")
b = lk.Kriging(y1, X, "matern5_2")
assert np.allclose(a.theta(), b.theta())
