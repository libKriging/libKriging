# Analyse : implémenter le multi-fidélité (co-krigeage récursif AR(1)) dans libKriging

> Analyse produite le 2026-08-04. Reflète l'état du dépôt à cette date.

Motivation : fonctionnalité manquante la plus demandée en simulation
industrielle (coupler un code cher/fin avec un ou plusieurs codes
grossiers/rapides), et cohérente avec l'ADN du projet — l'héritage
DiceKriging, dont l'auteur du modèle récursif (L. Le Gratiet) fait partie de
la même famille d'outils (`MuFiCokriging`).

## 1. La théorie (Le Gratiet 2013) — bonne nouvelle

Modèle auto-régressif de Kennedy & O'Hagan (2000), reformulé récursivement :

    Z_t(x) = ρ_{t-1}(x) · Z_{t-1}(x) + δ_t(x),   t = 2..s
    δ_t ⊥ Z_{t-1},   δ_t ~ GP(f_t(x)ᵀ β_t, σ_t² r_t(·,·;θ_t))

avec `t=1` la fidélité la plus basse (la moins chère) et `t=s` la plus haute.

Avec des **plans emboîtés** `D_s ⊆ D_{s-1} ⊆ … ⊆ D_1`, la vraisemblance
jointe se factorise en `s` vraisemblances indépendantes. Conséquences :

- **pas de matrice de covariance jointe à assembler ni à inverser** :
  `s` fits `Kriging` indépendants sur les résidus `y_t − ρ·y_{t-1}(D_t)`,
  en `O(Σ_t n_t³)` au lieu de `O((Σ_t n_t)³)` ;
- `ρ` s'estime **comme un coefficient de tendance** si on ajoute
  `y_{t-1}(D_t) · g(x)` en colonne(s) de la matrice de régression `F` du
  niveau `t` → réutilise directement la machinerie GLS / `beta` profilé
  existante de `KrigingImpl` ;
- prédiction récursive :

      μ_t(x)  = ρ_{t-1}(x) · μ_{t-1}(x) + μ_{δt}(x)
      s²_t(x) = ρ_{t-1}(x)² · s²_{t-1}(x) + s²_{δt}(x)

C'est donc une **classe de composition**, exactement le patron de
`NestedKriging` (526 l. `.cpp` / 176 l. `.hpp`, vecteur de sous-`Kriging`,
pas de save/load).

## 2. Le vrai point dur côté core

`Trend::RegressionModel` est un **enum fermé** `{None, Constant, Linear,
Interactive, Quadratic}` (`src/lib/include/libKriging/Trend.hpp`) : il
n'existe aucun chemin pour une matrice `F` fournie par l'appelant.
`Trend::regressionModelMatrix(regmodel, newXt)` construit toujours `F` à
partir de l'enum.

Deux options :

- **(a) chemin « F custom » dans `KrigingImpl`** → `ρ` estimé conjointement
  avec `β` par GLS, formulation exacte de Le Gratiet, propre et sans
  optimisation supplémentaire ; **mais** touche le cœur de `Kriging`, donc
  risque de régression sur toutes les classes existantes.
- **(b) profilage externe de `ρ`** : optimisation 1-D (ou `p`-D si
  `ρ(x)=g(x)ᵀρ`) par niveau, autour d'un `Kriging` non modifié → **zéro
  impact** sur le core, mais moins élégant, un peu plus lent, et l'incertitude
  sur `ρ` reste hors du modèle.

Recommandation : prototyper en **(b)** pour valider les formules contre
`MuFiCokriging`, puis migrer vers **(a)** si le surcoût ou l'imprécision
le justifie.

## 3. Rupture d'API : `(y, X)` devient une liste de niveaux

Toute la bibliothèque expose `fit(vec y, mat X, …)`. Le multi-fidélité
impose `fit(vector<vec> y, vector<mat> X, …)` plus des options par niveau.

**C'est ça qui coûte, pas les maths** : le marshalling de listes de matrices
doit être refait dans les **5 bindings**, dont deux pénibles :

- **Julia** : ABI C plate (`bindings/Julia/jlibkriging/csrc/libkriging_c.cpp`)
  — tableaux de pointeurs + tableaux de dimensions à passer à la main ;
- **Octave/MATLAB** : `*_binding.cpp/.hpp` + classe `.m` (cell arrays de
  matrices, et rappel du piège maison : les entiers doivent être `int32(...)`).

**Python** (pybind11 + carma) et **R** (`*_binding.cpp` + `*Class.R` +
`RcppExports` + `.Rd` + `NAMESPACE`) sont plus mécaniques.

Compter ~4 fichiers × 5 bindings + 1 notebook démo chacun (convention du
dépôt : `nestedkriging_branin2d_{py,r,julia}.ipynb`).

## 4. Surface fonctionnelle à décider (scope)

| Fonction | Coût | Recommandation v1 |
|---|---|---|
| `fit` / `predict` | cœur | oui |
| `logLikelihood` (somme des LL par niveau) | faible | oui |
| `simulate` récursif | moyen | oui — simulation niveau par niveau, propagée |
| `update` (ajout d'un point au niveau `t` → refit des niveaux ≥ `t`) | moyen | v1 si possible : c'est *le* cas d'usage industriel (enrichissement adaptatif / co-EGO) |
| `save` / `load` | moyen | différable (`NestedKriging` l'a différé) |
| variante bayésienne (priors sur `β`, `σ²`, `ρ` → variance prédictive corrigée) | élevé | v2 |
| `WarpKriging` comme sous-modèle de niveau | faible si templatisé | option (paramètre), surtout pas une classe séparée |

## 5. Pièges spécifiques à anticiper

- **Plans emboîtés obligatoires** pour la formulation exacte → contrôle
  runtime + message d'erreur explicite, ou mode approché assumé (utiliser
  `μ_{t-1}(D_t)` prédit au lieu de `y_{t-1}(D_t)` observé, ce qui introduit
  une variance non comptabilisée). **En pratique industrielle les plans ne
  sont presque jamais emboîtés** — il faut trancher tôt, c'est structurant.
- **`normalize`** doit être **global**, pas par niveau : une normalisation
  indépendante par niveau casse la relation d'échelle portée par `ρ`.
  (Rappel : `NestedKriging` ne supporte pas encore `normalize` du tout.)
- **Bruit** : `Kriging` gère déjà `noise` (vecteur) et `"nugget"` via
  `NoiseModel {None, Nugget, Heterogeneous}` → la basse-fidélité stochastique
  (Monte-Carlo, CFD instationnaire, éléments discrets) marche « gratuitement ».
  Gros argument de vente, à mettre en avant dans la doc.
- **Sous-estimation de la variance prédictive** si `ρ` et les hyperparamètres
  sont traités comme connus (plug-in). C'est exactement ce que la variante
  bayésienne de Le Gratiet corrige.
- **Composition avec les classes existantes** : ne pas créer
  `MultiFidelityWarpKriging`, `MultiFidelityNestedKriging`, … — le type de
  sous-modèle doit être une **option**, sinon explosion combinatoire.

## 6. Validation et documentation

- `tests/MultiFidelityKrigingTest.cpp` + valeurs de référence dans
  `tests/references/`.
- Oracle naturel : **`MuFiCokriging`**, le package R de Le Gratiet lui-même —
  cohérent avec la validation déjà faite contre DiceKriging
  (cf. `bindings/R/rlibkriging/DiceKriging.md` et `tests/references/`).
- `skills/libkriging/SKILL.md` §1 : nouvelle branche dans l'arbre de décision
  (« avez-vous plusieurs niveaux de fidélité ? ») + les 5
  `skills/libkriging/references/*.md` + `CHANGELOG.md`.

## 7. Estimation

- Cœur C++ + tests unitaires : **~1–2 semaines**.
- Bindings + notebooks + documentation : **~2–3 semaines** (le gros morceau).

**Séquencement proposé**

1. Trancher les décisions ouvertes (`DESIGN.md`) : traitement de `ρ`, support
   des plans non-emboîtés, forme de la signature `fit`.
2. Prototype C++ + test contre `MuFiCokriging`.
3. Bindings dans l'ordre **Python → R → Julia → Octave/MATLAB**
   (du plus simple au plus pénible ; Python sert de terrain d'essai à la
   convention de marshalling des listes).
