# NestedKriging — intégration dans libKriging

## Fichiers

```
src/lib/include/libKriging/NestedKriging.hpp
src/lib/NestedKriging.cpp
tests/NestedKrigingTest.cpp
```

## CMake

`src/lib/CMakeLists.txt` : ajouter `NestedKriging.cpp` à la liste des sources
de la cible `Kriging` (à côté de `Kriging.cpp`, `NuggetKriging.cpp`, ...).

`tests/CMakeLists.txt` : bloc sur le modèle des autres tests :

```cmake
add_executable(NestedKrigingTest NestedKrigingTest.cpp)
target_link_libraries(NestedKrigingTest LINK_PUBLIC Kriging Catch2)
add_dependencies(all_test_binaries NestedKrigingTest)
if(DEFINED ENV{GITHUB_ACTIONS})
    set(AdditionalCatchParameters "~[intensive]")
endif()
catch_discover_tests(NestedKrigingTest EXTRA_ARGS ${AdditionalCatchParameters})
unset(AdditionalCatchParameters)
```

## Points de design / hypothèses à vérifier

1. **`Covariance::CovFunc`** : supposé appelé comme `Cov(dx, theta)` avec
   `dx = x1 - x2` (différence brute), comme dans `_logLikelihood` avec les
   colonnes de `m_dX`. Si la convention diffère, seul `NestedKriging::corrMat`
   est à adapter. Alternative : réutiliser `KrigingImpl::covMat` (public) et
   diviser par sigma2 si elle inclut la variance.
2. **Refit `optim="none"` avec `is_theta_estim=false` / `is_sigma2_estim=false` /
   `is_beta_estim=false`** : suppose que `Kriging::fit` fige bien ces valeurs
   (chemin déjà utilisé par les bindings).
3. **Variance NK vs UK** : l'agrégation NK est en simple kriging (beta0 fixé,
   estimé au préalable par moyenne pondérée des GLS des sous-modèles). La
   variance ne contient donc pas de terme d'incertitude de tendance — cohérent
   puisque beta est figé lors du refit.
4. **Thread-safety** : le fit des sous-modèles est séquentiel (l'état RNG
   d'Armadillo et l'optimiseur lbfgsb doivent être audités avant un
   `omp parallel for`). La prédiction NK est parallélisée OpenMP sur les
   paires de groupes (lecture seule des membres, écritures disjointes).
5. **v1 non couvert** (extensions naturelles, mêmes patrons que le trio
   K/Nugget/Noise) : nugget/noise, normalize, save/load JSON, simulate,
   bindings R/Python/Matlab.

## Coûts

| opération | plein | nested (p groupes) |
|---|---|---|
| fit (1 éval LL) | O(n³) | O(p·(n/p)³) = O(n³/p²) |
| predict (par point) | O(n²) | PoE : O(n²/p) ; NK : O(n²) mais ∥ sur p(p-1)/2 paires |
| mémoire | O(n²) | O(n²/p) (+ n·chunk en NK, réglable via `set_predict_chunk`) |

Ordre de grandeur : n = 10⁵, p = 100 → sous-modèles de 10³ points ; fit ~10⁴×
plus rapide que le kriging plein (inaccessible de toute façon en mémoire).

## Étapes suivantes proposées

1. PR "core + tests" (ces fichiers).
2. Bindings R (`NestedKriging.R` + Rcpp glue) et Python (pybind11), signatures
   alignées sur `Kriging`.
3. `save`/`load` (schéma JSON : hyperparamètres unifiés + partition + refit à
   la volée des sous-modèles au chargement, pour éviter de sérialiser p Cholesky).
4. Benchmark reproductible vs `nestedKriging` (R) et `GpGp` : n ∈ {10⁴, 10⁵},
   d ∈ {2, 8}, critères RMSE / couverture IC 95 % / temps.

## Références

- Rullière, Durrande, Bachoc, Chevalier (2018), *Nested Kriging predictions
  for datasets with a large number of observations*, Statistics & Computing.
- Deisenroth & Ng (2015), *Distributed Gaussian Processes*, ICML (PoE/BCM/rBCM).

## Compatibilité WarpKriging (implémentée)

Un noyau warpé k(φ(x), φ(x′)) reste un prior GP valide : NK et PoE s'appliquent
en théorie. Deux blocages d'implémentation actuels :

- famille PoE : il suffit de templater `NestedKriging` sur le type de
  sous-modèle (signatures `fit`/`predict` identiques), mais l'unification des
  hyperparamètres doit couvrir θ, σ² **et** les paramètres de warp ;
- NK : nécessite d'évaluer le noyau warpé entre points quelconques, or
  `KrigingImpl::covMat` n'applique pas la `FeatureMap` φ et `WarpKriging`
  hérite de `KrigingImpl` en `protected`.

Réalisé : WarpKriging expose désormais covMat(X1, X2) (public, applique φ) et warp_params() ; NestedKriging accepte un argument warping et utilise des sous-modèles WarpKriging avec prior commun (θ, warp) estimé par un unique fit de référence sur un sous-échantillon global (taille min(n, warp_subsample), réglable via set_warp_subsample, défaut 1000), puis sous-modèles en optim="none" (fits fermés : un seul entraînement de warp au total au lieu de p). Historique du design : exposer dans `WarpKriging` un `covMat(X1, X2)` public
appliquant φ, puis brancher `NestedKriging::corrMat` sur
`submodel.covMat(...)/σ²` — ce qui éliminerait aussi l'hypothèse sur la
convention `Cov(dx, θ)`.
