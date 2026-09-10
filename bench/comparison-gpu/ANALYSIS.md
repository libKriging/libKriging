# Pourquoi libKriging est lent dans ce benchmark GPU

Analyse du chemin `objective="LLIterative(m)"` / `predictIterative` (branche
`feature/cg-predict`), à partir du code (`src/lib/Kriging.cpp`,
`src/lib/KrigingImpl.cpp`, `src/lib/LinearAlgebra.cpp`,
`src/lib/cuda/`) et de mesures sur H100 (`bench/comparison-gpu`).

## Quick wins appliqués (2026-09-09)

1. **`build_dX` n'est plus forcé pour les objectifs light** (`Kriging.cpp`) :
   `LLIterative` / `LLNystrom` / Vecchia-light en `optim="none"` ne
   construisent plus le tenseur `m_dX` (`d×n²`, jusqu'à 4 Go à n=8000)
   qu'ils ne lisent jamais. Aligné sur le chemin `optim!="none"` qui le
   faisait déjà. Tests `KrigingIterativeTest` / `Nystrom` / `Vecchia` : OK.
2. **Plafonds de threads levés** sur les matvecs matrix-free
   (`_logLikelihoodIterative` Rmul/dRmul_all, `predictIterative` Rmul) :
   `get_optimal_threads(8|2)` → `get_optimal_threads(omp_get_max_threads())`,
   donc pilotés par `OMP_NUM_THREADS`.

Effet mesuré (constructeur `LLIterative(20)`, n=2000, θ=0.3, `OMP_NUM_THREADS=48`) :

| cas | avant | après |
|---|---|---|
| d=4 (`sine_sum`) | 28.7 s | **5.5 s** |
| d=8 | 29.2 s | **9.0 s** |

SLQ : ~3–5× (levée du plafond 8). `m_dX` : supprimé du chemin.

## Corrections de fond appliquées (2026-09-09)

3. **Matvec batché device exposé** (`LinearAlgebraCuda::rmulBatched`) et
   **`stochasticLogDet` batché** (`LinearAlgebra::stochasticLogDetBatched`) :
   les `nprobe` récurrences de Lanczos du log-déterminant SLQ avancent
   maintenant **en lockstep**, un seul lancement `R·[v_1|…|v_20]` par pas
   de Lanczos sur le GPU au lieu de 20 matvecs CPU séquentiels. Le SLQ
   passe de **~9 s à ~0,1 s** (d=8, n=2000) — il n'est plus un goulot.
   Le même `rmulBatched` est prêt pour la trace de Hutchinson du gradient
   (à câbler ; le kernel `dR/dθ·v` reste à écrire).
4. **CG CUDA : scalaires α/β/convergence gardés sur le device**
   (`CudaLinearAlgebra.cpp` + kernels `cg_alpha`/`cg_beta`/`cg_restart`/
   `cg_any_active`). Supprime ~4 `cudaMemcpy` bloquants par itération ; le
   host ne rapatrie plus qu'un `int` « colonnes encore actives ? » toutes
   les 10 itérations. Gain surtout sensible **sous contention GPU** (chaque
   synchro attendait derrière les kernels des autres locataires — c'est ce
   qui faisait passer le CG de ~2,5 s à ~17 s au premier run).

Effet cumulé (constructeur `LLIterative(20)`, n=2000, θ=0.3) :

| | avant | quick wins | + fond |
|---|---|---|---|
| d=8 fit | 29 s | 9 s | **0,3 s** |
| d=4 fit | 29 s | 5,5 s | **3 s** (CG-bound, cf. reste) |

Tests C++ : `KrigingIterativeTest`, `KrigingPredictIterativeTest`,
`Nystrom`, `Vecchia`, `LinearAlgebra` (93 tests) — tous OK.

## Corrections de fond — 2e lot (2026-09-09)

5. **Kernel `dR/dθ·v` device** (`drmul_batched_kernel` + `lk_dlncov_pair`,
   formes closes identiques à `Covariance::DlnCovDtheta_*`) et
   `LinearAlgebraCuda::dRmulBatched`. La trace de Hutchinson du gradient
   (`_logLikelihoodIterative`) route un seul appel batché sur
   `[x | probe_0 … probe_{m-1}]` → **le gradient itératif n'est plus
   CPU-bound**. `iter_ll+grad` d=8 : ~18 s → **~1,8 s** (n=2000).
6. **Préconditionneur Nyström/Woodbury sur le chemin CG CUDA**
   (`conjugateGradient` params `precU/precDinv/precMcholLower` +
   `precond_apply`/`gemm_Ut`/`trisolve_MMt`/`cg_beta_precond` kernels).
   `WoodburyFactorization` expose `U()`/`Dinv()`/`McholLower()` → l'apply
   GPU est **bit-identique** au CPU. `objective="LLIterative(m,precond_rank)"`
   et `predictIterative(use_nystrom_precond=True)` **tournent maintenant
   sur GPU** (avant : fallback CPU forcé). Coupe les 2n itérations du
   régime mal conditionné.
7. **`predictIterative` stdev** : les solves de la *variance* utilisent
   `sqrt(tol)` (l'erreur sur l'écart-type ~ √résidu) — jamais plus serré
   que `tol`. Le solve de la *moyenne* garde `tol`. ~2× sur le nombre
   d'itérations CG du chemin `return_stdev=True` (48 s → ~9–24 s selon n).
   `predIter.stdev(32)` d=4 n=2000 : 48 s → **~9 s**.
8. **Binding** : `logLikelihoodIterativeFun` exposé dans `pylibkriging`.

Tests C++ : 45 iterative/predictIterative/LinearAlgebra + sweep large — OK
(dont #36/#37/#50 qui valident le CG préconditionné GPU contre l'exact).

## TL;DR

Sur `borehole` (d=8), `libkriging-gpu` vs `gpytorch` :

| n | phase | GPyTorch (H100) | libKriging (CUDA) | rapport |
|---|---|---|---|---|
| 2000 | fit | 0.38 s | **17 s** | ×45 |
| 2000 | update | 0.06 s | **26 s** | ×430 |
| 4000 | fit | 0.44 s | **68 s** | ×155 |
| 4000 | update | 0.10 s | **103 s** | ×1000 |
| 8000 | fit | 0.6 s | **273 s** | ×450 |
| 8000 | update | 0.23 s | **412 s** | ×1800 |
| 8000 | predict mean (2000 pts) | 0.19 s | 2.1 s | ×11 |
| 8000 | predict stdev (32 pts) | 0.002 s | 25 s | ×12000 |

## Décomposition mesurée (n=2000, θ=0.3, CUDA actif)

`fit` du benchmark = constructeur `Kriging(optim="none", "LLIterative(20)")`
+ `logLikelihoodFun`. Le constructeur seul, par différence
(`LLIter(1) − LL` ≈ solve CG ; `LLIter(20) − LLIter(1)` ≈ SLQ) :

| terme | d=4 (`sine_sum`) | d=8 | backend |
|---|---|---|---|
| `LL` exact (m_dX + Cholesky dense) — référence | 0.7 s | 1.1 s | CPU (BLAS) |
| `cgSolve([F|y])` — solve CG 2 colonnes (β/σ²) | **~17 s** | ~1 s | **GPU** |
| `stochasticLogDet` (SLQ, 20 probes × 20 Lanczos) | ~12 s | **~29 s** | **CPU**, ≤8 threads, séquentiel/probe |
| `m.logLikelihoodFun(θ, grad=True)` → `_logLikelihood` **exact** | ~0.3 s | ~0.5 s | CPU (BLAS) |

Selon le conditionnement de `R` (fixé par θ et d) **l'un ou l'autre**
domine, jamais le GPU utilement :

* **θ « long » / R mal conditionné** (d=4 ici) → le **solve CG GPU** ne
  converge pas (`tol=1e-8` jamais atteint) et tourne ses `max_iter = 2n` ≈
  4000 itérations, chacune avec ~4 `cudaMemcpy` bloquants → ~17 s pour
  2 colonnes sur H100.
* **θ « court » / R proche de l'identité** (d=8, ou `borehole` en unités
  physiques) → CG converge vite, mais le **SLQ log-dét reste sur CPU** :
  400 matvecs O(n²) fixes, ≤ 8 threads, séquentiels sur les 20 probes →
  ~29 s.

`fit`/`update` croissent en O(n²) (×4 quand n double). À n=8000 le H100 est
à **0 % pendant des minutes** (phase SLQ) puis sous-exploité (phase CG).
`m_dX` (tenseur de distances `d×n²`, **4 Go** à n=8000, cause n°2) s'ajoute
et est reconstruit à chaque `fit` **et** `update`.

## Cause n°1 : le GPU n'accélère que les *solves* CG — et mal

`LinearAlgebraCuda` n'expose qu'une seule primitive : `conjugateGradient`
(`src/lib/cuda/CudaLinearAlgebra.cuh`). Or une évaluation de
`_logLikelihoodIterative` (vraisemblance + gradient) fait bien plus que des
solves CG, et **tout le reste tourne sur CPU** :

| sous-étape de `_logLikelihoodIterative` | backend | coût |
|---|---|---|
| `cgSolve([F | y])` (β/σ²) — 2 colonnes | **GPU** si CUDA | O(n²·iters) |
| `stochasticLogDet(Rmul, …)` (SLQ, log‖R‖) | **CPU seulement** | `nprobe`(20) × `lanczos_steps`(20) = **400 matvecs**, *séquentiels sur les probes* |
| `cgSolve(probes)` (Hutchinson, w = R⁻¹z) — 20 colonnes *(gradient uniquement)* | **GPU** si CUDA | O(n²·iters) |
| `dRmul_all(x)` + boucle `dRmul_all(probe_p)` (trace de Hutchinson) *(gradient uniquement)* | **CPU seulement** | **1 + nprobe = 21 passes** O(n²·d), chacune évaluant `cov()` **et** `_DlnCovDtheta()` par paire |

`stochasticLogDet` (`src/lib/LinearAlgebra.cpp:512`) reçoit le lambda
`Rmul` **CPU** (`Kriging.cpp:1716`), pas `cgSolve` : Lanczos n'est pas un
solve CG, et il n'existe aucun matvec device réutilisable. Idem pour
`dRmul_all` (`Kriging.cpp:1751`), défini comme une boucle OpenMP CPU.

**Et le solve CG que le GPU fait quand même est lent** : `cgSolve([F|y])`
= 17 s pour 2 colonnes à n=2000 sur H100. Aucun préconditionneur, `tol =
1e-8` jamais atteint sur un `R` d'interpolation mal conditionné → CG tourne
ses `max_iter = 2n` ≈ 4000 itérations, chacune avec ~4 `cudaMemcpy`
bloquants (α/β et test de convergence calculés côté hôte — voir cause
n°5). GPyTorch, lui, préconditionne (Cholesky pivoté) et garde tout sur le
device.

### Ce que mesure vraiment la phase `fit` du benchmark

`fit` = `Kriging(..., optim="none", "LLIterative(20)")` **+** un
`m.logLikelihoodFun(theta, grad=True)`. Or :

* **le constructeur** (`optim="none"` → branche `Kriging.cpp:2247`) appelle
  `_logLikelihoodIterative` **sans gradient** → 1× `cgSolve(FY)` [GPU] + 1×
  SLQ (400 matvecs CPU) + construit `m_dX` (cause n°2) ;
* **`m.logLikelihoodFun(theta, True)`** (`Kriging.cpp:391`) appelle
  **`_logLikelihood` — la vraisemblance EXACTE dense-Cholesky O(n³)**, PAS
  `_logLikelihoodIterative`. Il existe bien `logLikelihoodIterativeFun`
  (`Kriging.cpp:1822`) mais **il n'est pas exposé dans `pylibkriging`** :
  depuis Python on ne peut évaluer QUE l'objectif exact. Heureusement il
  est rapide (BLAS/LAPACK : ~0,2–0,5 s à n=2000), donc il ne domine pas —
  mais **le coût itératif de la phase `fit` est entièrement dans le
  constructeur** (SLQ + `m_dX`), pas dans l'appel `logLikelihoodFun`.

Donc `fit ≈ constructeur ≈ SLQ (CPU) + m_dX (CPU) + cgSolve(FY) (GPU)`.
La phase `update` reconstruit un modèle → repaie SLQ + `m_dX`.

Le gradient itératif complet (SLQ + 21× `dRmul_all`) n'est déclenché que
par `logLikelihoodIterativeFun` en C++ (bench C++ `bench-iterative-cuda`,
fits BFGS) — pas par ce benchmark Python.

## Cause n°2 : un tenseur de distances O(n²·d) construit pour rien

`Kriging::fit` (`Kriging.cpp:2143`) :

```cpp
const bool build_dX = (optim == "none")
                      || !((objective.rfind("LLNystrom", 0) == 0) || ...);
```

`build_dX` est **inconditionnellement vrai** dès `optim="none"`, y compris
pour `LLIterative`. `fit_setup_impl` appelle alors
`LinearAlgebra::compute_dX(m_X)` qui **alloue `arma::mat(d, n*n)`** puis la
remplit en O(n²·d), suivi de `max(abs(m_dX), 1)` (encore O(n²·d)).
`compute_dX` est plafonné à **2 threads** (`get_optimal_threads(2)`).

| n (d=8) | taille de `m_dX` |
|---|---|
| 2000 | 256 Mo |
| 4000 | 1 Go |
| 8000 | **4 Go** |

Or la branche `LLIterative`-light (`Kriging.cpp:2247`) fait `return` sans
jamais construire de factorisation exacte : **`m_dX` n'est jamais lu**. Le
commentaire à `Kriging.cpp:2137` le dit lui-même (« Nystrom/Iterative never
touch m_dX ») — la condition `(optim == "none")` est le bug. C'est aussi ce
qui invalide la mesure d'empreinte mémoire O(n) annoncée pour `LLIterative`.

## Cause n°3 : plafonds de threads sur un nœud à 192 cœurs

`get_optimal_threads(max_default)` (`Kriging.cpp:40`) renvoie
`min(omp_max_threads, max_default)`. Les matvecs matrix-free sont bridés :

| matvec | plafond |
|---|---|
| `Rmul` / `dRmul_all` dans `_logLikelihoodIterative` | **8** (`Kriging.cpp:1597`, `:1759`) |
| `Rmul` dans `predictIterative_impl` | **2** (`KrigingImpl.cpp`) |
| `compute_dX` | **2** (`LinearAlgebra.cpp:638`) |

Sur cette machine (192 cœurs) `OMP_NUM_THREADS=48` ne change rien : ~184
cœurs restent inutilisés pendant toute la phase CPU.

## Cause n°4 : `stochasticLogDet` séquentiel sur les probes

`LinearAlgebra.cpp:518` : `for (p = 0; p < nprobe; ++p)` — les 20 probes
Rademacher sont traités l'un après l'autre, chacun faisant 20 pas de
Lanczos. Les probes sont pourtant indépendants (contrairement à
`cgSolve(probes)` qui, lui, parallélise sur les 20 colonnes via
`LinearAlgebra::conjugateGradient`). SLQ ne bénéficie donc **ni** du GPU
**ni** du parallélisme inter-probe, seulement des ≤ 8 threads intra-matvec.

## Cause n°5 : `predictIterative`

* **mean** : le fallback CPU `Rmul` (`KrigingImpl.cpp`) est plafonné à
  **2 threads** *et* utilise `Xt.col(i) - Xt.col(j)` → allocation d'un
  temporaire armadillo par paire (exactement ce que le `Rmul` de
  `_logLikelihoodIterative` évite via `memptr()` brut). Le chemin CUDA est
  pris quand CUDA est actif, mais (`CudaLinearAlgebra.cpp`) :
  - réalloue tous les buffers device (`cudaMalloc`×11) et re-transfère
    `Xt`/`theta`/`B` **à chaque appel** de `cgSolve` ;
  - fait **~4 `cudaMemcpy` bloquants par itération CG** (`d_scratch` D2H,
    `d_alpha` H2D, puis `d_scratch` D2H, `d_beta` H2D) — la vérification de
    convergence et le calcul de α/β sont faits côté **hôte**. Chaque
    `cudaMemcpy` = une synchro device complète.
  - **ne s'arrête pas avant `max_iter = 2n`** si la tolérance `1e-8` n'est
    pas atteinte. Sur un noyau d'interpolation mal conditionné (ex.
    `sine_sum` θ=0.3), CG tourne ses 2n itérations entières → 4000
    itérations × (~4 synchros hôte + lancements de kernels) domine, pas les
    FLOPs du matvec. Aucun préconditionneur n'est activé par défaut.
* **stdev** : `cgSolve(R_on)` = **un solve CG par point de prédiction**
  (`KrigingImpl.cpp`, `R_on` est n×n_n), plus `cgSolve(m_F)` pour la
  correction GLS. Coût O(n²·iters·n_n). D'où 25 s pour 32 points à n=8000.

## Cause n°6 : matrix-free strict, sans BLAS ni cache

Chaque `Rmul` recalcule tous les `R_ij` par un appel `cov()`
(transcendantes : `sqrt`, `exp`, polynôme pour Matérn 5/2) — à **chaque**
itération CG **et** à chaque pas de Lanczos, jamais mis en cache
(volontaire : `Kriging.cpp:1556`). `max_iter = 2n`, `tol = 1e-8`. Sur
`sine_sum` θ=0.3 le noyau est mal conditionné → CG proche de son plafond
`2n` d'itérations, chacune un matvec O(n²) plein.

Face à ça, GPyTorch (BBMM) exécute **toute** la chaîne — solves, log-dét
stochastique (Lanczos), trace stochastique — en **tenseurs GPU batchés**
avec préconditionneur Cholesky pivoté.

## Pistes de correction (par ratio impact/effort)

**Quick wins (peu de code, gros gain sur ce benchmark) :**

1. **Ne pas construire `m_dX` pour les objectifs light en `optim="none"`.**
   `Kriging.cpp:2143` : exclure `LLNystrom` / `LLIterative` / Vecchia-light
   de la condition `(optim == "none") || …` (leur branche `optim="none"`
   fait `return` sans `make_Model` exact et ne lit jamais `m_dX`). Gain :
   −4 Go et −O(n²·d) par `fit` **et** par `update`. Restaure aussi
   l'empreinte O(n) annoncée. Risque ~nul.
2. **Relever/supprimer les plafonds `get_optimal_threads(8|2)`** des matvecs
   itératifs (`Kriging.cpp:1597/1759`, `KrigingImpl.cpp`, `LinearAlgebra.cpp:638`)
   — ou les indexer sur `OMP_NUM_THREADS`. Sur un nœud 192 cœurs : ~×10–20
   sur les phases CPU-bound (SLQ, `m_dX`).
3. **Paralléliser `stochasticLogDet` sur les probes** (`LinearAlgebra.cpp:518`,
   boucle `for p`) — 20 tâches indépendantes, aujourd'hui séquentielles.
   ~×20 potentiel sur la phase SLQ (combiné avec #2).
4. **Préconditionner le `cgSolve` du fit par défaut.** Le
   préconditionneur Nyström/Woodbury existe déjà
   (`objective="LLIterative(m,precond_rank)"`) mais `precond_rank=0` par
   défaut → CG non préconditionné → 2n itérations sur `R` mal conditionné.
   L'activer par défaut (rang ~50–100) ou l'exposer proprement côté Python.

**Corrections de fond :**

5. **Exposer un matvec batché device** dans `LinearAlgebraCuda` et y router
   la boucle de Lanczos de `stochasticLogDet` et `dRmul_all` — même matvec
   O(n²) que le CG tourne déjà sur GPU, juste pas exposé hors CG. Rendrait
   `fit`/`update` réellement GPU-bound.
6. **CG CUDA** (`CudaLinearAlgebra.cpp`) : (a) garder `Xt`/`theta` sur le
   device entre les `cgSolve` successifs d'une même évaluation (pas de
   re-upload) ; (b) faire le test de convergence + calcul α/β sur device et
   ne synchroniser que tous les k pas → supprime ~4 `cudaMemcpy` bloquants
   par itération ; (c) implémenter le chemin préconditionné côté CUDA
   (aujourd'hui `use_nystrom_precond` force le fallback CPU).
7. **`predictIterative` mean** : réutiliser le `Rmul` rapide de
   `_logLikelihoodIterative` (memptr brut, symétrique, 8 threads) plutôt que
   la version 2-threads à temporaires armadillo.
8. **`predictIterative` stdev** : préconditionneur Nyström par défaut, et/ou
   `max_iter` réduit + `tol` relâché pour la variance (bien moins
   d'exactitude requise que pour la moyenne) ; sinon le coût est
   intrinsèquement O(n²·iters·n_points).
9. **Binding** : exposer `logLikelihoodIterativeFun` dans `pylibkriging`
   (aujourd'hui `logLikelihoodFun` renvoie silencieusement l'objectif exact
   O(n³)).

## Reproduire

Décomposition par sous-phase (CUDA actif), à θ=0.3 :

```python
import time, numpy as np, pylibkriging as lk
def T(f):
    a=time.perf_counter(); r=f(); return time.perf_counter()-a, r
n,d = 2000,4
X = np.random.default_rng(0).random((n,d)); y = np.sin(2*np.pi*X).sum(1)
th = np.full(d,0.3); pr = {"theta":th.reshape(1,-1),"sigma2":float(np.var(y))}
mk = lambda obj: lk.Kriging(y,X,"matern5_2",regmodel="constant",normalize=False,
                            optim="none",objective=obj,parameters=pr)
lk.set_cuda_iterative_enabled(True)
print("LL exact   ", T(lambda: mk("LL"))[0])            # m_dX + Cholesky dense
print("LLIter(1)  ", T(lambda: mk("LLIterative(1)"))[0]) # + cgSolve(FY) GPU + SLQ(1)
print("LLIter(20) ", T(lambda: mk("LLIterative(20)"))[0])# + SLQ(20) CPU
m = mk("LLIterative(20)")
Xt = np.random.default_rng(1).random((300,d))
print("predIter mean ", T(lambda: m.predictIterative(Xt, False, 0, 1e-6))[0])
print("predIter stdev", T(lambda: m.predictIterative(Xt[:8], True, 0, 1e-6))[0])
```

Observation directe la plus parlante : lancer le benchmark et regarder
`nvidia-smi` pendant une phase `fit`/`update` de `libkriging-gpu` — le H100
reste à 0 % pendant des minutes (n=8000), tandis que 8 threads CPU tournent
à fond.
