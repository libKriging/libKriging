# Krigeage multi-sorties

Dossier de travail pour le support de sorties multiples (`Y` n × q) dans
libKriging, avec un focus sur les sorties temporelles / fonctionnelles.

**État :** `MultiOutputKriging` implémenté en C++ et Python pour les modèles
`"pca"`, `"shared"` et `"separable"` (`src/lib/MultiOutputKriging.cpp`) ;
`"separable(<kernel>)"` à venir.

## Contenu

| Fichier | Rôle |
|---|---|
| `ANALYSIS.md` | Revue biblio (§1), logiciels (§2), feuille de route (§3), sorties temporelles (§4), API (§5), questions ouvertes (§6) |
| `draft/MultiOutputKriging.hpp` | Esquisse d'API C++ initiale (la version réelle est `src/lib/include/libKriging/MultiOutputKriging.hpp`) |
| `draft/example_python.py` | Exemple d'usage Python ; sections `pca` et `shared` exécutables |

## Reprise rapide

1. `ANALYSIS.md` §6 : Q2, Q3, Q5, Q6 tranchées ; Q1, Q4, Q7 ouvertes.
2. `ANALYSIS.md` §5.4 : fait `"pca"`, généralisation de la factorisation de
   `KrigingImpl`, `"shared"` (objectifs `LL`/`LOO`, `update_simulate`),
   `"separable"` (Σ libre, `predictCovFactors`) ; reste `"separable(<kernel>)"`,
   puis bindings R / Octave / Julia.

Validation de `"shared"` contre `RobustGaSP::ppgasp(method = "mle", nugget.est = FALSE)`
(n = 40, q = 30, matern 5/2, 2026-10-08) : θ identiques à 7 chiffres, LL égales
à 4e-13 près, moyennes à 1e-6 × sd(Y), écarts-types à 0,4 % près.

Voisins : GEK (travail local non publié), multi-fidélité (branche
`feature/multi-fidelity-cokriging`).
