# Krigeage multi-sorties

Dossier de travail pour le support de sorties multiples (`Y` n × q) dans
libKriging, avec un focus sur les sorties temporelles / fonctionnelles.

**État :** `MultiOutputKriging` implémenté en C++ et dans les quatre bindings
pour les modèles `"pca"`, `"shared"`, `"separable"` et `"separable(<kernel>)"`
(`src/lib/MultiOutputKriging.cpp`), avec save/load.

## Contenu

| Fichier | Rôle |
|---|---|
| `ANALYSIS.md` | Revue biblio (§1), logiciels (§2), feuille de route (§3), sorties temporelles (§4), API (§5), questions ouvertes (§6) |
| `draft/MultiOutputKriging.hpp` | Esquisse d'API C++ initiale (la version réelle est `src/lib/include/libKriging/MultiOutputKriging.hpp`) |
| `draft/example_python.py` | Exemple d'usage Python ; sections `pca` et `shared` exécutables |

## Reprise rapide

1. `ANALYSIS.md` §6 : Q1, Q2, Q3, Q5, Q6, Q7 tranchées ; Q4 ouverte.
2. `ANALYSIS.md` §5.4 : fait `"pca"`, généralisation de la factorisation de
   `KrigingImpl`, `"shared"` (objectifs `LL`/`LOO`, `update_simulate`),
   `"separable"` (Σ libre, `predictCovFactors`), bindings Python / R /
   Octave-MATLAB / Julia (même API, sorties `m × q` et `m × q × nsim`), doc
   (`docs/math/MultiOutput.md`, skill, README des bindings) et un notebook par
   binding (`bindings/*/multioutputkriging_*.ipynb`), `"separable(<kernel>)"`
   (Σ = σ² R_t(φ), φ estimé avec θ, vraisemblance vérifiée contre la densité
   gaussienne dense) et save/load (JSON, version 2, état ajusté restitué à
   l'identique). Estimateur scikit-learn
   `MultiOutputKrigingRegressor` fait. Reste : étape 2 (ICM hétérotopique, `q` petit, PR séparée, Q1 ; à concevoir
   avec `MarkovCoKriging`, `ANALYSIS.md` §7). Q7 hors
   périmètre pour l'instant.

Validation de `"shared"` contre `RobustGaSP::ppgasp(method = "mle", nugget.est = FALSE)`
(n = 40, q = 30, matern 5/2, 2026-10-08) : θ identiques à 7 chiffres, LL égales
à 4e-13 près, moyennes à 1e-6 × sd(Y), écarts-types à 0,4 % près.

Voisins : GEK (travail local non publié), multi-fidélité (branche
`feature/multi-fidelity-cokriging`).
