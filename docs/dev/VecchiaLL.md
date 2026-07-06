# Vecchia log-likelihood — objective="VLL(m)"

## Usage

```r
k <- Kriging(y, X, "matern5_2", objective = "VLL(30)")   # or "VLL" (default m=30)
```

Valable dans tous les bindings sans modification : `objective` est une chaîne
passée telle quelle au C++.

## Principe

Vecchia (1988) : log p(y) = Σᵢ log p(yᵢ | y_N(i)) où N(i) contient (au plus)
m plus proches voisins parmi les points *précédents* dans un ordre maxmin
(Guinness 2018). Coût O(n·m³) par évaluation au lieu de O(n³) ; densité
gaussienne valide (Cholesky inverse creuse) ; exacte pour m = n−1.

Profilages identiques à l'objectif "LL" : σ² en forme fermée, β par GLS
par-conditionnelles (tendances constant/linear/quadratic supportées).
Gradient en θ analytique (théorème de l'enveloppe pour β̂).

## Implémentation

- `parse_vll_m` : "VLL" → 30, "VLL(m)" → m ; validation stricte.
- `make_vecchia_sets` : ordre maxmin glouton + m-NN parmi les prédécesseurs,
  O(n²·d) en accès mémoire brut, figés avant l'optimisation (inputs normalisés,
  indépendants de θ).
- `_logLikelihoodVecchia` : n conditionnelles m×m (`safe_chol_lower`).
- Protocole de commit : pendant l'optimisation, `fit_ofn` reçoit
  `grad_out != nullptr` → évaluation Vecchia pure ; l'appel final
  (`grad_out == nullptr`, `km_data != nullptr`) fait **une** factorisation
  exacte O(n³) à θ* → `predict`/`simulate`/`update` inchangés et exacts.
  Le chemin `update(refit)` fait explicitement cet appel final.
- `logLikelihoodVecchiaFun(theta, return_grad)` exposé pour inspection/tests.

## Limites (v1)

- `NoiseModel::None` uniquement (pas de nugget/noise).
- Le commit final reste O(n³) mémoire/temps : praticable jusqu'à n ~ 2·10⁴.
- `predictVecchia(X_n, return_stdev, m=0)` : prédiction locale par
  conditionnement sur les m observations les plus proches (Katzfuss &
  Guinness 2021, response-only) — O(q·m³), parallèle, utilisable après tout
  fit. Moyenne UK avec le β committé, variance SK (pas de covariances
  croisées entre points de prédiction : utiliser `predict` pour le joint).
  À n=1000, d=2 : 3,6 ms vs 33 ms pour les 100 points (BLAS de référence).
  Étape suivante pour n ≥ 10⁵ : sauter le commit exact du fit.
- Effet d'écran faible en grande dimension : recommandé pour d ≤ ~5
  (complémentaire de NestedKriging, robuste en dimension quelconque).
- Ensembles Vecchia non sérialisés (reconstruits au refit).

## Validation

`docs/dev/validate_vll_math.py` (référence numpy exécutée) : VLL(n−1) ≡ LL
exacte à ~2·10⁻⁶, gradient vs différences finies à ~10⁻⁷, convergence en m,
θ̂ VLL(20) ≈ θ̂ MLE exact sur champ Matérn 2D. Tests C++ :
`tests/KrigingVecchiaTest.cpp`.
