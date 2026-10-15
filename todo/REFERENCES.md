# Références

## Modèle

- **Kennedy, M.C. & O'Hagan, A. (2000).** *Predicting the output from a
  complex computer code when fast approximations are available.*
  Biometrika 87(1), 1–13.
  → le modèle auto-régressif AR(1) originel, formulation jointe.

- **Le Gratiet, L. (2013).** *Multi-fidelity Gaussian process regression for
  computer experiments.* Thèse, Université Paris-Diderot.
  → **la référence à implémenter** : reformulation récursive, factorisation
  de la vraisemblance sous plans emboîtés, formules de prédiction
  récursives, variante bayésienne.

- **Le Gratiet, L. & Garnier, J. (2014).** *Recursive co-kriging model for
  design of computer experiments with multiple levels of fidelity.*
  International Journal for Uncertainty Quantification 4(5), 365–386.
  → version article, plus courte, contient les formules directement
  exploitables.

## Oracle de validation

- **Package R `MuFiCokriging`** (Le Gratiet) — implémentation de référence.
  Fonctions clés : `MuFicokm()` (fit), `predict()`, `NestedDesign()`
  (construction de plans emboîtés).
  ⚠ Archivé sur le CRAN — récupérer depuis les archives CRAN ou une source
  équivalente. Fonctions utiles pour générer les références :
  la sortie de `MuFicokm()` expose `rho`, `Beta`, `SigmA2`, `Theta` par niveau.

- Cohérent avec la démarche de validation déjà en place dans le dépôt
  contre DiceKriging (cf. `tests/references/` et
  `bindings/R/rlibkriging/DiceKriging.md`).

## Cas tests analytiques

- **Forrester, A., Sóbester, A. & Keane, A. (2007).** *Multi-fidelity
  optimization via surrogate modelling.* Proc. R. Soc. A 463, 3251–3269.
  → le cas 1-D à deux fidélités canonique :

      f_haute(x) = (6x − 2)² · sin(12x − 4)
      f_basse(x) = 0.5 · f_haute(x) + 10(x − 0.5) − 5      x ∈ [0,1]

  Idéal comme premier test : `ρ` est exactement 0.5 et le biais est linéaire,
  donc un `ρ` constant avec tendance linéaire sur `δ` doit retrouver la
  vérité terrain presque exactement.

- **Branin 2-D** en versions fine / grossière : cohérent avec les notebooks
  démo existants du dépôt (`*_branin2d_*.ipynb` dans chaque binding).

## Généralisation : co-krigeage collocalisé / Markov-model

- **Journel, A.G. (1999).** *Markov models for cross-covariances.*
  Mathematical Geology 31(8), 955–964.
  → hypothèse de covariance croisée **proportionnelle** à une covariance
  directe (Markov Model 1 : `C_12(h) = (B_12/B_11)·C_1(h)` ; Markov Model 2 :
  symétrique en `C_2`). C'est la même hypothèse structurante que l'AR(1) de
  Le Gratiet, mais **sans exiger d'ordre de fidélité** entre les deux
  variables — seulement un plan `D_2 ⊆ D_1` (variable secondaire connue
  partout où la primaire est échantillonnée).

- **Xu, W., Tran, T.T., Srivastava, R.M. & Journel, A.G. (1992).**
  *Integrating seismic data in reservoir modeling: the collocated
  cokriging alternative.* SPE Annual Technical Conference.
  → co-krigeage collocalisé : la secondaire n'intervient qu'au point cible
  (dérive externe), simplification pratique de MM1/MM2.

→ **Implication d'implémentation** : ces modèles se réduisent, comme
l'AR(1), à un krigeage ordinaire sur un résidu / avec covariable — même
patron de composition, aucune matrice de covariance jointe. Cf.
`DESIGN.md` §1 et §2 (généralisation retenue) et `PLAN.md`.

## Extensions (hors scope v1, pour mémoire)

- **Picheny & Ginsbourger** et la littérature co-EGO / enrichissement
  multi-fidélité : « quel point, à quel niveau ? ». C'est ce que les
  utilisateurs demanderont juste après le modèle lui-même ; l'API `update`
  doit rester compatible avec ce cas d'usage.

- **Perdikaris et al. (2017)**, *Nonlinear information fusion algorithms
  for data-efficient multi-fidelity modelling* (NARGP) : généralisation
  non-linéaire de la relation entre niveaux. Note d'architecture : ce serait
  naturellement exprimable via `WarpKriging` comme sous-modèle, ce qui
  renforce l'argument de faire du type de sous-modèle une **option** et non
  une classe séparée.
