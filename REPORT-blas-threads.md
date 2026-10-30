# Threads BLAS et performances des fits libKriging

Étude demandée par `TODO-blas-threads.md`, refaite après rebase sur `master` @ `07fc2a8`
(qui intègre déjà le budget de threads de #415 : garde qui restaure, multistart Kriging
séquentiel sans découpage, `Optim::set_num_threads_percent`, boucles OpenMP dès n = 200 sous
Linux). Données brutes et scripts : `perf-blas/`. Outil de mesure : `bench/bench-blas-threads.cpp`.

## Résumé

- Sur une VM de 4 vCPU sans SMT, avec OpenBLAS 0.3.26 pthreads et le code de `master`,
  **OpenBLAS au budget (4 threads) à côté de l'équipe OpenMP (4 threads) est jusqu'à 3-4×
  plus lent qu'OpenBLAS sur 1 thread pour n < 500**. Les fits à n = 200 gagnent ×1,7 à ×3,6
  avec 1 thread BLAS (toutes classes) ; le fit MLP du TODO (n = 200) passe de 14,4 s à 4,5 s.
- Cause : les threads OpenBLAS attendent activement (~0,1 s) après chaque appel parallèle,
  l'équipe OpenMP aussi (`libgomp`) ; 4 + 4 threads actifs sur 4 CPU logiques s'empêchent
  mutuellement de travailler (`sched_yield`, ~10⁶ changements de contexte par fit).
- À partir de n ≈ 500-700, les threads BLAS redeviennent rentables (×1,2 à ×2,0 à n = 1000-4000).
- Prototype retenu : **règle « petit problème »** dans `BudgetScope` : `fit` / `predict` /
  `update` sur n < 500 observations d'apprentissage passent OpenBLAS à 1 thread, restauré après
  l'appel. **Mode automatique par défaut** : la règle n'agit que si les threads BLAS et l'équipe
  OpenMP dépassent le nombre de CPU logiques (pas de marge SMT : VM, runners CI, SMT désactivé).
  Réglage : `Optim::set_blas_threads_min_n` / `LK_BLAS_THREADS_MIN_N` (-1 automatique, 0 désactivé,
  n > 0 seuil explicite).
- **Contradiction avec `docs/dev/parallel.md` (décision 1)**, qui a abandonné ce passage à 1 thread
  sur un i9-10900K (10 cœurs / 20 threads). Le mode automatique la concilie par hypothèse (voir
  « Limites ») : sur l'i9, 10 + 10 = 20 threads tiennent sur 20 CPU logiques, la règle ne s'y
  active pas. **Cette hypothèse n'a pas été vérifiée sur machine avec SMT** : c'est la principale
  réserve avant fusion.

## 1. Machine (phase 1)

`perf-blas/machine.txt` :

| | |
|---|---|
| CPU | Intel Xeon @ 2,10 GHz (VM), 4 vCPU, 1 thread par cœur (pas de SMT), AVX-512 |
| BLAS | OpenBLAS 0.3.26 pthreads (Ubuntu `openblas-pthread`), `DYNAMIC_ARCH`, noyau `Cooperlake`, 4 threads par défaut |
| OpenMP | libgomp (GCC 13.3) |
| Build | Release, Ninja, sans bindings |
| Base | `master` @ `07fc2a8` + cette branche |

`ctest` (hors Python) : 225/225 verts avec le noyau `Cooperlake` (celui de #429), avant et après
la règle.

Constat du TODO reproduit (`perf-blas/phase1.txt`, `bench-mlpkriging 1 200 4 small_selu`) :

| configuration | fit | user | sys | changements de contexte |
|---|---:|---:|---:|---:|
| défaut (BLAS 4, OpenMP 4) | 14,2-14,7 s | 44-45 s | 17-18 s | 0,9-1,7 M |
| `OPENBLAS_NUM_THREADS=1` | 4,5-4,7 s | 19-20 s | 0,3 s | 800 |
| règle (`LK_BLAS_THREADS_MIN_N=400`) | 4,2-4,6 s | 18-19 s | 0,4-0,7 s | 3 600-5 300 |
| `OMP_WAIT_POLICY=PASSIVE` | 4,2 s | 10 s | 8 s | 38 000 |
| `OMP_NUM_THREADS=1` (BLAS borné au budget = 1) | 4,5 s | 4,7 s | 0,4 s | 31 |
| BLAS 1 + `OMP_NUM_THREADS=1` | 4,1 s | 4,3 s | 0,2 s | 10 |

Contrairement au contexte du TODO (« `OMP_WAIT_POLICY=PASSIVE` ne change rien »), OpenMP compte
désormais : `master` exécute depuis #415 les boucles de covariance et de gradient en OpenMP dès
n = 200 (`kOmpMinRows`). Pour ce fit, ces boucles n'apportent rien en temps réel (BLAS 1 + OMP 1
est le plus rapide) et quadruplent le CPU.

## 2. Micro-benchmarks à paramètres fixés (phase 2)

`bench-blas-threads <op> <n> <d>` : modèle construit à θ fixé (`fit(..., "none")`), puis l'opération
répétée ≥ 1,5 s (3 blocs, médiane), après une attente active de 300 ms (transitoire de démarrage
d'OpenBLAS, `bench/thread-budget/RESULTS.md` section 9). OpenMP au défaut (budget = 4).
`OPENBLAS_NUM_THREADS` ∈ {1, 2, 4} (4 = cœurs physiques = CPU logiques). Médiane de 3 répétitions
(d = 4), 1 répétition pour d = 10 et n = 4000. Ratio = temps(1 thread) / temps(k threads), > 1 si
k threads est plus rapide. Fichiers : `phase2-microbench.csv` (+ `-omp1`, `-timeout4`,
`blas-omp-split`).

| op | n = 50 | 100 | 200 | 400 | 500 | 600 | 700 | 1000 | 2000 | 4000 |
|---|---|---|---|---|---|---|---|---|---|---|
| `kriging_ll` 4 thr | ×0,93 | ×0,90 | **×0,36** | ×0,97 | ×1,22 | ×1,04 | ×1,43 | ×1,62 | ×1,75 | ×1,99 |
| `kriging_predict` 4 thr | ×0,27 | ×0,48 | ×0,67 | ×0,99 | ×1,00 | ×1,11 | ×1,00 | ×1,22 | ×1,53 | ×1,62 |
| `mlp_ll` 4 thr | ×0,95 | ×0,83 | **×0,33** | ×0,89 | ×1,04 | ×0,96 | ×1,02 | ×1,05 | ×1,54 | — |
| `mlp_predict` 4 thr | ×0,25 | ×0,36 | ×0,64 | ×1,08 | ×1,15 | ×1,19 | ×1,29 | ×1,36 | ×1,44 | ×2,09 |
| `warp_ll` 4 thr | ×0,91 | ×0,87 | **×0,37** | ×0,77 | ×0,98 | ×0,87 | ×1,08 | ×1,13 | ×1,44 | — |

(`logLikelihoodFun(θ, grad = true)` pour `*_ll` ; `predict` sur 1000 points avec écart-type ;
MLPKriging `{16, 8}`, 2 sorties, selu, matern5_2 ; WarpKriging un `kumaraswamy` par variable.)
2 threads n'est jamais le meilleur choix (tableau complet : `perf-blas/phase2-summary.md`).

- **Seuil** : 1 thread est le meilleur jusqu'à n = 400 pour toutes les opérations ; la bascule
  est à n ≈ 500 (log-vraisemblance Kriging), 600-700 (`predict`, Warp, MLP). d = 10 : même
  bascule (n ≈ 700 ; `warp_ll` vers 1000).
- **CPU** : avec 4 threads BLAS, le CPU vaut 3,5-4× le temps réel à toutes les tailles.
- **Saut à n = 200** : seuil des boucles OpenMP. Avec `OMP_NUM_THREADS=1`, 4 threads BLAS
  sont au contraire un peu plus rapides dès n = 200 (×1,18 `kriging_ll`, ×1,05 `mlp_ll`) : la
  perte vient de la concurrence BLAS / OpenMP, pas de BLAS seul.
- **Répartition du budget** (`phase2-blas-omp-split.csv`, médiane de 2) :

  | n | BLAS 1 + OMP 4 | BLAS 2 + OMP 2 | BLAS 4 + OMP 4 |
  |---|---:|---:|---:|
  | 200 (`kriging_ll`) | 2,73 ms | 3,05 | 11,7 |
  | 400 | 15,9 | 13,6 | 15,0 |
  | 1000 | 110 | 100 | 72,8 |
  | 2000 | 769 | 561 | 389 |

  Un partage sans sursouscription (2 + 2 = 4 CPU) évite l'effondrement à n = 200 mais perd
  30-50 % aux grands n : un seuil sur n reste la meilleure règle simple.
- **`OPENBLAS_THREAD_TIMEOUT=4`** (attente active minimale) ne corrige pas n = 200
  (×0,37 `kriging_ll`, ×0,33 `mlp_ll` à 4 threads) : ce n'est pas une alternative.

## 3. Décomposition du coût (phase 3)

`perf record -g -F 499`, fit MLP n = 200 (`bench-mlpkriging 1 200 4 small_selu`),
`perf-blas/phase3-profile.txt` :

| | `OPENBLAS_NUM_THREADS=1` | `=4` |
|---|---:|---:|
| libgomp (attente active de l'équipe OpenMP, `0x256c2`) | 72 % | 50 % |
| noyau (`__schedule`, `do_sched_yield`, `do_syscall_64`) | 1,6 % | 29 % |
| libc (`__sched_yield`) | 2,5 % | 8 % |
| OpenBLAS (`dgemm_kernel`, `dtrsm_kernel_*`) | 16 % | 10 % |
| libKriging | 6 % | 2 % |

- Ce qui coûte n'est ni `potrf`, ni `trsm`, ni les `gemm` du réseau (moins de 20 % du temps à
  eux tous), mais l'**attente active** : celle de l'équipe OpenMP entre deux régions, et, à 4
  threads BLAS, les `sched_yield` des threads OpenBLAS inactifs et l'ordonnanceur noyau qu'ils
  sollicitent.
- `openblas_set_num_threads` lui-même coûte ~10 ns (mesure de la première campagne, même
  OpenBLAS) ; le réveil du pool après un passage de 1 à 4 threads coûte ≤ 0,5 ms à n = 400. Le
  réglage se fait donc une fois par appel public (`fit`, `predict`, `update`), pas par
  évaluation : c'est ce que fait `BudgetScope`.
- Un processus OpenBLAS (4 threads) brûle ~0,28 s de CPU dans les 250 ms suivant son chargement
  sans aucun appel BLAS (transitoire déjà décrit par `RESULTS.md` section 9) : toutes les mesures
  ci-dessus attendent 300 ms avant de commencer.

## 4. Prototypes (phase 4)

État de `master` vis-à-vis des prototypes du TODO :

- **A. Garde qui restaure** : déjà dans `master` (#415 : `BlasThreadGuard` restaure, les handlers
  `pthread_atfork` restaurent le parent, `BlasThreadsTest` le vérifie avec et sans multistart et
  après `fork`). Rien à ajouter.
- **B. Plafond selon n** : `master` plafonne déjà au budget (cœurs physiques, `BudgetScope`) et
  répartit le multistart (`TaskPool`). Ajouté ici : 1 thread BLAS si n < seuil.
- **C. Réglage utilisateur** : `Optim::set_blas_threads_min_n` / `LK_BLAS_THREADS_MIN_N` ; le
  budget lui-même reste réglable par `Optim::set_num_threads_percent`.

Implémentation (`src/lib/ThreadBudget.cpp`) : `BudgetScope(n)` aux entrées `fit` / `predict` /
`update` de `Kriging` et `WarpKriging` (donc `MLPKriging`) ; ailleurs (`simulate`, Vecchia,
Nystrom, NestedKriging) `BudgetScope()` inchangé. Sans effet dans un worker du pool (le pool a
déjà fixé sa part). OpenMP n'est pas touché. Un seul binaire sert aux deux variantes :
`LK_BLAS_THREADS_MIN_N=0` reproduit exactement le chemin de `master`.

Fits sur graines (`bench-blas-threads fit <classe> <optim> <n> 4 <graine>`, données et départs
fixés par la graine, attente de 300 ms ; médianes ; « meilleure LL » = nombre de graines où la
variante atteint la meilleure log-vraisemblance des deux). `master` = règle désactivée, `règle` =
seuil 500. Fichiers : `phase4-fits.csv`, `phase4-summary.md`.

| classe | optim | n | graines | temps master | temps règle | ratio | CPU master → règle | meilleure LL master / règle |
|---|---|---:|---:|---:|---:|---:|---|---|
| Kriging | BFGS | 100 | 10 | 0,090 s | 0,092 s | ×0,98 | 0,33 → 0,09 s | 10 / 10 |
| Kriging | BFGS | 200 | 10 | 0,578 | 0,222 | **×2,60** | 2,3 → 0,9 | 10 / 10 |
| Kriging | BFGS | 400 | 10 | 1,04 | 1,05 | ×1,00 | 4,1 → 2,5 | 10 / 10 |
| Kriging | BFGS | 1000 | 10 | 15,8 | 16,7 | ×0,95 (*) | = | 10 / 10 |
| Kriging | BFGS | 2000 | 5 | 158 | 158 | ×1,00 (*) | = | 5 / 5 |
| Kriging | BFGS10 | 100 | 10 | 0,092 | 0,090 | ×1,03 | 0,36 → 0,28 | 10 / 10 |
| Kriging | BFGS10 | 400 | 10 | 1,39 | 1,38 | ×1,01 | = | 10 / 10 |
| Kriging | BFGS10 | 1000 | 5 | 23,5 | 25,3 | ×0,93 (*) | = | 5 / 5 |
| nugget | BFGS | 100 | 10 | 0,074 | 0,077 | ×0,96 | 0,29 → 0,08 | 9 / 8 |
| nugget | BFGS | 200 | 10 | 0,827 | 0,228 | **×3,63** | 3,3 → 0,9 | 8 / 10 |
| nugget | BFGS | 400 | 10 | 1,21 | 0,96 | ×1,26 | 4,7 → 2,4 | 10 / 10 |
| nugget | BFGS | 1000 | 10 | 4,16 | 3,63 | ×1,15 (*) | = | 10 / 10 |
| nugget | BFGS10 | 100 | 10 | 0,210 | 0,162 | ×1,30 | 0,75 → 0,45 | 10 / 10 |
| nugget | BFGS10 | 400 | 10 | 3,07 | 3,08 | ×0,99 | = | 10 / 10 |
| nugget | BFGS10 | 1000 | 3 | 23,1 | 22,8 | ×1,01 (*) | = | 3 / 3 |
| Warp kumaraswamy | BFGS | 100 | 10 | 0,059 | 0,057 | ×1,03 | 0,23 → 0,06 | 10 / 10 |
| Warp kumaraswamy | BFGS | 200 | 10 | 0,508 | 0,263 | **×1,93** | 2,0 → 0,9 | 9 / 10 |
| Warp kumaraswamy | BFGS | 400 | 10 | 1,25 | 1,34 | ×0,93 | 4,9 → 2,5 | 8 / 10 |
| Warp kumaraswamy | BFGS | 1000 | 3 | 10,1 | 10,4 | ×0,97 (*) | = | 3 / 3 |
| Warp kumaraswamy | BFGS10 | 100 | 10 | 0,366 | 0,352 | ×1,04 | = | 10 / 10 |
| Warp kumaraswamy | BFGS10 | 400 | 10 | 5,40 | 5,51 | ×0,98 | = | 10 / 10 |
| Warp mlp | BFGS | 100 | 10 | 2,82 | 2,08 | ×1,35 | 11,2 → 2,1 | 7 / 3 |
| Warp mlp | BFGS | 200 | 10 | 2,44 | 1,43 | **×1,71** | 9,6 → 4,8 | 6 / 4 |
| Warp mlp | BFGS | 400 | 5 | 16,0 | 10,8 | ×1,48 | 63 → 21 | 3 / 2 |
| MLPKriging | BFGS+Adam | 100 | 10 | 0,488 | 0,523 | ×0,93 | 1,9 → 0,5 | 5 / 8 |
| MLPKriging | BFGS+Adam | 200 | 10 | 3,90 | 1,50 | **×2,60** | 15,3 → 5,9 | 6 / 4 |
| MLPKriging | BFGS+Adam | 400 | 10 | 8,42 | 4,76 | ×1,77 | 33 → 9,5 | 2 / 8 |
| MLPKriging | BFGS+Adam | 1000 | 3 | 198 | 198 | ×1,00 (*) | = | 3 / 3 |
| MLPKriging | BFGS | 100 | 10 | 0,466 | 0,421 | ×1,11 | 1,8 → 0,4 | 5 / 5 |
| MLPKriging | BFGS | 200 | 10 | 2,59 | 1,54 | **×1,68** | 10,2 → 5,6 | 5 / 5 |
| MLPKriging | BFGS | 400 | 5 | 7,51 | 7,61 | ×0,99 | 30 → 13 | 2 / 3 |

(*) n ≥ 500 : la règle ne s'applique pas, les deux variantes sont identiques ; ces lignes donnent
le **niveau de bruit des fits** (×0,93 à ×1,15).

- **Temps réel** : gain de ×1,7 à ×3,6 à n = 200 pour toutes les classes, de ×1,3 à ×1,8 à
  n = 400 pour les classes à réseau (Warp `mlp`, MLPKriging) ; neutre à n = 100 (pas de boucle
  OpenMP sous n = 200) et pour Kriging / Kumaraswamy à n = 400 ; aucune perte au-delà du bruit.
- **CPU** : divisé par 1,5 à 5 sous le seuil.
- **Multistart (`BFGS10`)** : neutre ; le pool de `master` donne déjà 1 thread BLAS par worker.
- **Optimum** : identique pour Kriging / nugget / Kumaraswamy (mêmes chemins ou mêmes optima) ;
  pour les classes à réseau, le nombre de threads change les arrondis donc le chemin
  (FAQ de `master`) : la règle atteint le meilleur optimum plus souvent pour MLPKriging
  `BFGS+Adam` à n = 400 (8/10 contre 2/10) et moins souvent pour Warp `mlp` à n = 100 (3/10 contre
  7/10). Aucun sens systématique (sur 10 graines, 7 contre 3 n'est pas significatif) : les fits
  de ces classes ne sont pas reproductibles d'un nombre de threads à l'autre, avec ou sans règle.

## 5. Non-régression (phase 5)

- `ctest` hors Python : **225/225** avec la règle en mode automatique (active sur cette machine),
  comme sans elle.
- `BlasThreadsTest` : cas ajouté « Below Optim::blas_threads_min_n observations a fit runs
  OpenBLAS on 1 thread » (un thread de surveillance voit OpenBLAS à 1 thread pendant un fit et
  un `predict` à n = 60 avec un seuil de 100, au budget à n = 120, et le compte de l'appelant
  restauré après). Le cas « appelant à 4 threads, retrouvé après `fit` avec et sans multistart »
  du TODO existe déjà dans `master`.
- `bench/bench.sh` réduit (n = 200 et 400, toutes les classes, `fit`, 3 itérations) : ×1,8 à
  ×5,7 à n = 200, neutre à n = 400 (section 6).
- pytest : non exécuté ; aucun binding n'est modifié (le réglage n'est exposé dans aucun binding :
  la CI #419 compare les bindings entre eux, et la variable `LK_BLAS_THREADS_MIN_N` vaut pour
  tous).
- `git clang-format --diff origin/master` : propre.

## 6. `bench.sh` réduit

`bench/bench.sh iterations=3 n=<200|400> d=4 <classe> fit`, `master` = `LK_BLAS_THREADS_MIN_N=0`,
branche = mode automatique (actif ici) ; médiane de 3 fits (même graine pour les deux variantes,
mais chemins d'optimisation différents dès que le nombre de threads change). Logs :
`perf-blas/benchsh/`.

| class | config | n | master median (ms) | branch median (ms) | master / branch |
|---|---|---:|---:|---:|---:|
| kriging | Kriging | 200 | 722 | 126 | x5.71 |
| kriging | Kriging | 400 | 435 | 337 | x1.29 |
| nuggetkriging | nugget | 200 | 592 | 208 | x2.84 |
| nuggetkriging | nugget | 400 | 600 | 627 | x0.96 |
| warpkriging | none | 200 | 104 | 57 | x1.83 |
| warpkriging | affine | 200 | 3865 | 1517 | x2.55 |
| warpkriging | boxcox | 200 | 8216 | 2914 | x2.82 |
| warpkriging | kumaraswamy | 200 | 9554 | 3105 | x3.08 |
| warpkriging | knots | 200 | 8383 | 3178 | x2.64 |
| warpkriging | mlp | 200 | 20798 | 8916 | x2.33 |
| warpkriging | neural_mono | 200 | 13123 | 4844 | x2.71 |
| warpkriging | categorical | 200 | 17249 | 7273 | x2.37 |
| warpkriging | ordinal | 200 | 8869 | 3199 | x2.77 |
| warpkriging | none | 400 | 542 | 363 | x1.49 |
| warpkriging | affine | 400 | 8300 | 8238 | x1.01 |
| warpkriging | boxcox | 400 | 13840 | 14394 | x0.96 |
| warpkriging | kumaraswamy | 400 | 14552 | 13710 | x1.06 |
| warpkriging | knots | 400 | 20696 | 18851 | x1.10 |
| warpkriging | mlp | 400 | 54325 | 53761 | x1.01 |
| warpkriging | neural_mono | 400 | 29465 | 28475 | x1.03 |
| warpkriging | categorical | 400 | 36523 | 38281 | x0.95 |
| warpkriging | ordinal | 400 | 14781 | 15161 | x0.97 |
| mlpkriging | small_selu | 200 | 13292 | 4552 | x2.92 |
| mlpkriging | medium_selu | 200 | 5708 | 2872 | x1.99 |
| mlpkriging | large_selu | 200 | 14232 | 5877 | x2.42 |
| mlpkriging | small_tanh | 200 | 6211 | 1863 | x3.33 |
| mlpkriging | adam_only | 200 | 14962 | 4231 | x3.54 |
| mlpkriging | small_selu | 400 | 24885 | 23273 | x1.07 |
| mlpkriging | medium_selu | 400 | 30495 | 31179 | x0.98 |
| mlpkriging | large_selu | 400 | 38077 | 36202 | x1.05 |
| mlpkriging | small_tanh | 400 | 34768 | 33313 | x1.04 |
| mlpkriging | adam_only | 400 | 24617 | 23137 | x1.06 |

- n = 200 : ×1,8 à ×5,7 pour toutes les configurations (WarpKriging, toutes déformations : ×1,8
  à ×3,1 ; MLPKriging : ×2,0 à ×3,5).
- n = 400 : ×0,95 à ×1,10 sauf Kriging (×1,29) et Warp `none` (×1,49) ; aucune perte au-delà du
  bruit (3 itérations seulement ; écart-type jusqu'à 35 % de la moyenne pour Kriging à n = 200).
- Écart avec la phase 4 à n = 400 pour MLPKriging (×1,77 en phase 4, ×1,0 ici) : configurations
  différentes (ici `{16, 8}` etc., `BFGS+Adam` sans multistart, sur `sum sin(2 pi x)`, données de
  `bench-mlpkriging`) et 3 itérations contre 10 graines ; la phase 4 fait foi.

## 7. Seuil et politique recommandés

1. **Garder la règle, seuil 500**, en **mode automatique par défaut** : active seulement quand
   threads BLAS + budget OpenMP > CPU logiques. Sur cette VM (4 + 4 > 4) elle agit ; sur l'i9 de
   `RESULTS.md` (10 + 10 = 20) elle n'agit pas, conformément à la décision 1 de `parallel.md`.
2. **Avant fusion**, mesurer sur une machine avec SMT (l'i9 de #415) :
   `perf-blas/run-phase2.sh` et `perf-blas/run-phase4-all.sh` avec `LK_BLAS_THREADS_MIN_N=0`
   contre `500`. Si la règle y est aussi favorable, supprimer la condition SMT ; si elle y est
   défavorable comme l'annonce `RESULTS.md`, la condition est justifiée ; si les deux machines
   divergent pour une autre raison (VM, ordonnanceur), revoir le critère.
3. **À étudier séparément** : les boucles OpenMP sous n ≈ 500. Avec BLAS sur 1 thread, elles
   accélèrent la log-vraisemblance Kriging (×1,8 à n = 200) mais pas le fit MLP du TODO
   (BLAS 1 + OMP 1 y est le plus rapide, pour 4× moins de CPU) ; `kOmpMinRows = 200` a été
   choisi sur l'i9.
4. Alternative sans code pour les utilisateurs sur VM : `LK_NUM_THREADS_PERCENT=50` (budget 2 sur
   4 vCPU, 2 + 2 ≤ 4) donne ici des fits aussi rapides que la règle à n = 200 (nugget : 0,17 s
   contre 0,47 s sans règle, 0,15-0,21 s avec), au prix de 30-50 % aux grands n.

## 8. Limites

- **Une seule machine**, une VM à 4 vCPU sans SMT : elle ressemble aux runners CI, pas aux postes
  de travail. Le plafond « cœurs physiques » ne peut pas y être distingué des CPU logiques.
  La condition SMT du mode automatique est une **hypothèse** tirée de la comparaison avec
  `RESULTS.md` (autre machine, autre OpenBLAS de build, mesures de `master`), non une mesure.
- **Runners CI à 4 vCPU** : même configuration que cette machine ; la règle y sera active et
  change le nombre de threads BLAS (donc les arrondis) des tests à n < 500. `ctest` complet est vert
  ici ; d'autres runners (noyau OpenBLAS `Cooperlake` / Xeon 6973P, #429) peuvent réagir autrement.
- **macOS avec Accelerate** : pas d'`openblas_set_num_threads`, la règle est sans effet
  (`get_openblas_threads()` = 0). macOS Intel avec OpenBLAS de brew : non mesuré ;
  `kOmpMinRows` = 800 hors Linux, donc pas d'équipe OpenMP sous n = 800 et a priori pas de
  concurrence à n < 500.
- **Windows** : `openblas_set_num_threads` trouvé par `GetProcAddress` seulement pour un
  `openblas.dll` / `libopenblas.dll` déjà chargé ; non mesuré (#417).
- **OpenBLAS variante OpenMP** : threads BLAS et équipe OpenMP partagent alors le runtime OpenMP ;
  non mesuré ici (`RESULTS.md` section 2 l'a fait sur l'i9).
- **Mesures de fits** : 3 à 10 graines ; au-delà de n = 1000 les fits prennent des minutes
  (Kriging `BFGS` n = 2000 : 158 s), d'où des grilles réduites (pas de Warp `mlp` ni de MLPKriging
  `BFGS` à n ≥ 1000, pas de `BFGS10` à n = 2000). `mlp_ll` et `warp_ll` ne sont pas mesurés à
  n = 4000.
- `WarpKriging::logLikelihoodFun` ne différencie que par rapport à θ (warp fixé) : les
  micro-benchmarks `mlp_ll` / `warp_ll` ne couvrent pas la rétropropagation dans le réseau, que
  les fits de la phase 4 couvrent.
- Première campagne (avant rebase, ancienne machine à 2,8 GHz, ancien `master`) abandonnée : code
  différent (pas de boucles OpenMP, multistart limité à 1 thread) et mesures prises pendant le
  transitoire de démarrage d'OpenBLAS. Elle reste sur la branche locale `backup/blas-threads-v1`
  (non poussée).
