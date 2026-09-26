# Convergence des méthodes itératives (CG) — rapport de travail

Statut : **changements CPU validés (165/165 tests), changements GPU NON compilés ni testés**.
Rédigé pour permettre la reprise par une autre session (humaine ou IA), en particulier
dans Claude Code sur des machines disposant de GPU CUDA / HIP / SYCL / Metal.

Environnement de mesure de ce rapport : conteneur Linux, 1 vCPU, 3 Go RAM, gcc 13,
OpenBLAS/LAPACK système, cmake 4.4 (`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` requis),
build `Release`, bindings désactivés. Aucune mesure GPU.

---

## 0. TL;DR

1. Le CG de libKriging (CPU et les 4 backends GPU) faisait un **redémarrage complet
   `p = r` (ou `p = z`) toutes les 50 itérations**. Cela détruit la conjugaison et
   transforme le CG en quasi plus-forte-pente sur les matrices de covariance mal
   conditionnées : c'était la **cause réelle** des deux tests en échec
   (`... dense fast path matches the matrix-free path`), pas l'ordre de sommation
   BLAS ni le warm-start.
2. CPU : redémarrage supprimé, remplacé par confirmation sur le vrai résidu +
   remplacement de résidu en conservant la direction + arrêt sur plancher d'arrondi.
   GPU : le recalcul périodique du vrai résidu est conservé mais la direction n'est
   plus réinitialisée (mise à jour beta standard). **À tester sur matériel.**
3. `predictIterative` : préconditionnement de Nyström **automatique**, conservé
   seulement si le facteur capte ≥ 50 % de trace(R).
4. Optimiseur LLIterative : **warm-start** des résolutions CG d'une évaluation à la
   suivante (par thread). Cache R⁻¹[F|y] estampillé par θ.
5. Rejeté après mesure : détection de stagnation par fenêtre, pas de Gauss-Seidel
   dans `updateIterative`, préconditionnement auto de LLIterative.
6. Problèmes **antérieurs** mis en évidence, non corrigés : le fit LLIterative non
   borné dérive vers la borne haute de θ ; biais SLQ important ; imprécision de
   `update` ; la résolution de la moyenne atteint presque toujours `max_iter`.

---

## 1. Historique git et état des branches

| Élément | Valeur |
|---|---|
| `origin/master` | `6f50133a` = `v1.2.2` + 3 commits (fix R `utils` Imports) |
| `origin/feature/cg-predict` (avant rebase) | `c222a114`, 115 commits, base `d41c8134` |
| Branche rebasée (locale) | `rebase/cg-predict-on-v1.2.2`, HEAD `94600659`, 116 commits sur `origin/master` |
| Branche de ce travail | `feature/iterative-convergence`, 1 commit au-dessus de `94600659` |

**Rien n'a été poussé depuis la session d'origine (pas d'identifiants GitHub).**
Ordre de publication recommandé :

```sh
# 1) publier la branche rebasée (préférer une nouvelle branche à un force-push
#    si d'autres personnes travaillent sur feature/cg-predict)
git push origin rebase/cg-predict-on-v1.2.2
#    ou : git push --force-with-lease origin rebase/cg-predict-on-v1.2.2:feature/cg-predict
# 2) publier ce travail et ouvrir une PR dont la BASE est la branche rebasée
git push -u origin feature/iterative-convergence
gh pr create --base rebase/cg-predict-on-v1.2.2 --head feature/iterative-convergence \
  --title "Iterative CG: remove periodic restart, auto Nystrom precond, optimizer warm start" \
  --body-file docs/dev/iterative-convergence/REPORT.md
```

Une PR basée sur `master` contiendrait aussi les 116 commits du rebase.

### 1.1 Rebase `feature/cg-predict` → `master` (fait)

- Conflit `skills/libkriging/references/cpp.md` : deux sections conservées.
- Conflits notebooks (`docs/comparisons/libKriging_vs_GPyTorch.ipynb`,
  `docs/math/{llnystrom,llvecchia,subsetofdata}_vs_cholesky.ipynb`) : version de la
  branche retenue. Justification : côté master, seules les sorties avaient changé
  (sources vérifiées cellule par cellule), sauf deux correctifs GPyTorch
  (`noise_constraint`, `.double()`) que la branche couvre autrement
  (`torch.set_default_dtype(torch.float64)` + contrainte de bruit propre).
- `CHANGELOG.md` : conflits résolus par union, puis commit correctif `94600659` :
  la branche écrivait ses ~445 lignes dans la section **déjà publiée** `[1.2.0]` ;
  elles ont été déplacées dans `[Unreleased]`. L'entrée `subsetOfData` (#358) publiée
  en 1.2.0, que la branche avait fusionnée dans l'entrée `predictIterative`, a été
  restaurée à l'identique. Sections ≥ 1.2.2 identiques à master.
- `src/`, `tests/`, sous-modules : identiques octet pour octet à la branche d'origine
  (seul `cmake/version.cmake` passe de 1.2.1 à 1.2.2).
- Le passage en version 2.0.0 n'a PAS été fait (aucune rupture d'API identifiée
  justifiant un majeur ; décision à prendre par les mainteneurs).

---

## 2. Diagnostic

### 2.1 Hypothèse « warm-start » réfutée

Hypothèse initiale : le warm-start du cache `m_iterative_RinvFY_cache` (commit
`ae450e3f`) expliquerait les écarts. Réfutée :

- test `predictIterative ... dense fast path` : modèle ajusté avec `"LL"` → cache vide
  → démarrage à froid des deux côtés (résultats WARM et COLD identiques bit à bit) ;
- test `LLIterative ... dense fast path` : `logLikelihoodIterativeFun` appelle
  `_logLikelihoodIterative` sans x0.

Effet réel du warm-start quand il est actif (après fit `LLIterative(30,0,24,2,1e-10)`,
référence Cholesky aux mêmes θ/β/σ², écart max rapporté à sd(y)) :

| Cas | cond(R) | err. rel. du cache | moyenne COLD | moyenne WARM |
|---|---|---|---|---|
| n=60 θ=0,15 sans matrice | 4,5e3 | 2e-3 | 8,6e-5 | 3,3e-7 |
| n=60 θ=0,15 dense | 4,5e3 | 2e-3 | 6,0e-5 | 3,6e-7 |
| n=160 θ=0,3 sans matrice | 5,2e8 | 0,91 | 5,3e-3 | 2,4e-3 |
| n=160 θ=0,3 dense | 5,2e8 | 0,90 | 2,6e-3 | 9,8e-4 |

Conclusion : le warm-start **améliore** (facteur 2 à 250 à budget égal). Dégradation
marginale de l'écart-type (1–3 %) car `W_F` est accepté en 0 itération dès que le
cache satisfait `sqrt(tol)`.

### 2.2 Cause réelle : redémarrage `p = r`

Banc `harness/cg.cpp` sur la matrice R exacte, trois variantes (CG standard ;
redémarrage `p = r` toutes les 50 itérations = comportement de la bibliothèque ;
remplacement de résidu en gardant la direction) :

- n=60, cond 4,5e3, 2n itérations : CG standard résidu 1,7e-9 / erreur 4e-9 ;
  bibliothèque 4,8e-5 / 2,3e-3.
- n=160, cond 1,5e8, 5000 itérations : CG standard erreur 3e-10 ; bibliothèque
  stagne à 0,67.
- Remplacement de résidu avec direction conservée : proche du CG standard
  (n=60 : 2,3e-8 à 2n).

Validation dans la bibliothèque (diagnostic temporaire `LK_CG_RESTART_EVERY`,
retiré depuis ; banc `harness/ll.cpp`, n=160, θ={0,25 ; 0,3}) :

| Budget CG | Redémarrage | Écart ll (sans matrice/dense) | Écart max gradient |
|---|---|---|---|
| mult=2 | 50 | 0,136 | 17 |
| mult=2 | aucun | 0,039 | 59 |
| mult=40 | 50 | 3,7e-3 | 22 |
| mult=40 | aucun | 1,1e-8 | 9,4e-8 |

Le commentaire historique du code (« croissance instable après ~2n itérations »)
n'est reproduit dans aucun test : cette croissance n'apparaît qu'après convergence,
et le critère d'arrêt la neutralise.

---

## 3. Changements implémentés (branche `feature/iterative-convergence`)

### 3.1 CG CPU — `src/lib/LinearAlgebra.cpp`

`conjugateGradient` (une colonne à la fois) et `conjugateGradientBatched` :

- plus de redémarrage périodique ;
- quand le résidu récursif passe sous `tol`, recalcul du vrai résidu `b - A*x`
  (une multiplication, **partagée** entre les colonnes concernées dans la version
  batchée : `AmulBatched(Xc.cols(idx))`) ;
  - vrai résidu < tol → colonne convergée ;
  - sinon, si le vrai résidu n'est pas au moins 10 % meilleur que la confirmation
    précédente (`cg_stall_factor = 0.9`) → arrêt, colonne **non convergée**
    (plancher d'arrondi) ;
  - sinon remplacement de résidu (r ← vrai résidu) et **direction conservée** ;
- gardes : `pAp <= 0`, `rz_new <= 0`, résidu non fini ;
- `n_unconverged` compte désormais toutes les colonnes n'ayant pas atteint `tol`
  sur leur vrai résidu (max_iter, plancher, ou rupture) ;
- `LK_DEBUG_CG_ITERS=1` affiche aussi `confirm=` (nombre de confirmations).

**Essai rejeté** : détection de stagnation par fenêtre (arrêt si pas de gain de
10 % sur max(200, n) itérations). Sur n=160, cond ~1e8, elle arrêtait les
résolutions vers 1000–1500 itérations (écart ll 1,5e-3, gradient 76) : le CG mal
conditionné a des plateaux de plusieurs centaines d'itérations avant de converger.

Conséquence non résolue : quand `tol` est inatteignable, une résolution consomme tout
`max_iter` (la résolution de la moyenne dans `predictIterative` le fait dans presque
tous les cas mesurés, cf. §4.1). Une détection de stagnation *fiable* reste à trouver.

### 3.2 CG GPU — CUDA / HIP / SYCL / Metal (NON TESTÉ)

Fichiers : `src/lib/{cuda/CudaLinearAlgebra.cpp, hip/HipLinearAlgebra.cpp,
sycl/SyclLinearAlgebra.cpp, metal/MetalLinearAlgebra.cpp}`.

Dans la branche `(it + 1) % restart_every == 0` (recalcul exact du résidu, toujours
en fp64 pour le chemin CUDA mixed-precision) :

- avant : `cg_restart[_precond]_launch` (rz_old ← r·r ou r·z, test de tolérance)
  puis copie `p ← r` (ou `p ← z`) ;
- après : `cg_beta[_precond]_launch` (test de tolérance, beta = rz_new/rz_old,
  rz_old ← rz_new) puis `batched_update_p_launch` (`p ← r + beta p` ou `z + beta p`),
  c'est-à-dire exactement les appels de la branche non périodique, mais avec le vrai
  résidu.

Les noyaux `cg_restart*` ne sont plus appelés par ces boucles (laissés en place).
Différences restantes avec le CPU : pas de confirmation au moment de la convergence
(le test porte sur le résidu récursif entre deux recalculs), pas d'arrêt sur plancher.
Metal travaille en **fp32** : le comportement sur matrices mal conditionnées est à
vérifier en priorité.

### 3.3 Préconditionnement auto — `src/lib/KrigingImpl.cpp` (`predictIterative_impl`)

- Si `use_nystrom_precond == false`, `LinearAlgebra::cg_auto_precond == true`,
  `precond_rank > 0` et `n >= 2*precond_rank` : construction du facteur de Nyström
  de rang `min(precond_rank, n)` ; conservé si `1 - sum(diag_resid)/n >= 0.5`
  (fraction de trace(R) = n captée), sinon abandonné (CG simple).
- API : `LinearAlgebra::cg_auto_precond`, `cg_auto_precond_min_captured`,
  `set_cg_auto_precond(bool)` (`src/lib/include/libKriging/LinearAlgebra.hpp`) ;
  variables d'environnement `LK_CG_AUTO_PRECOND=0`, `LK_CG_AUTO_PRECOND_MIN_CAPTURED`.
- `LK_DEBUG_CG_ITERS=1` affiche `nystrom precond rank=.. captured=.. auto:kept|auto:dropped|requested`.
- S'applique à Kriging, WarpKriging, MLPKriging (implémentation partagée). Les
  bindings héritent du comportement sans changement d'API.
- Le préconditionneur est aussi transmis aux backends GPU (déjà supporté par
  `LK_PRED_GPU_BIND`), **non testé** en mode auto.

Justification de l'absence d'estimation a priori du conditionnement : un Lanczos de
~30 pas coûte ~30 multiplications O(n²), plus que la construction du facteur
(O(n·k²)).

### 3.4 Warm-start — `src/lib/Kriging.cpp`, `Kriging.hpp`

- `_logLikelihoodIterative` : nouveaux paramètres `W_x0` / `W_out` (point de départ
  et résultat de R⁻¹·sondes). x0 vérifiés en dimension avant usage (sinon ignorés).
- Objectif d'optimisation LLIterative (`make_fit_objective`) : état partagé
  `WarmState` (map `thread::id → (RinvFY, W)` protégée par mutex, détenue par
  `shared_ptr` capturé dans la lambda) ; chaque évaluation part des solutions de
  l'évaluation précédente du même thread. `LK_CG_WARM_OPTIM=0` désactive.
  Limite : les chemins GPU fusionnés `[F|y|sondes]` (CUDA/HIP/Metal avec gradient)
  n'utilisent pas x0.
- Cache R⁻¹[F|y] : nouveau membre `m_iterative_RinvFY_cache_theta`, positionné aux
  3 points d'écriture du cache ; `iterative_cache_valid()` exige n, nombre de
  colonnes et θ identiques. Utilisé par `updateIterative` et `predictIterative`.
  Le cache n'est pas sérialisé (inchangé) : un modèle rechargé n'a pas de warm-start.

**Essai rejeté** : pas de Gauss-Seidel par blocs sur les nouvelles lignes dans
`updateIterative` (x_u = R_uu⁻¹(b_u − R_uo x_old) au lieu de 0). Gain nul sur 16
configurations (n=300, θ∈{0,1 ; 0,3}, n_u∈{5 ; 40}, mult∈{1 ; 2}, banc
`harness/up.cpp`) : l'erreur est dominée par l'imprécision de la solution des
anciennes lignes. Retiré.

### 3.5 Tests — `tests/KrigingPredictIterativeTest.cpp`, `tests/KrigingIterativeTest.cpp`

- predictIterative dense vs sans matrice : comparaison à `tol=1e-14`, `max_iter=10n`
  (solutions convergées) ; seuil moyenne 1e-8·sd inchangé ; seuil écart-type
  1e-6·sd (résolutions de variance à `sqrt(tol)=1e-7`).
- LLIterative dense vs sans matrice : partie non préconditionnée passée à
  `LLIterative(30,0,24,40,1e-10)` (cond ~1e8 → 2n itérations structurellement
  insuffisantes) ; commentaire « KNOWN FAILURE / roundoff plateau » réécrit.
- Durée de la suite : 39,9 s (avant) → 27,2 s (après correction CG seule) → 46 s
  (final ; +21 s dus au test LLIterative à mult=40).

---

## 4. Mesures

### 4.1 Préconditionnement auto (`harness/pc.cpp`, 50 points, tol 1e-8, max_iter 2n)

| Cas | Trace captée | Décision | Temps OFF → AUTO | Erreur moyenne OFF → AUTO |
|---|---|---|---|---|
| n=200 θ=0,02 | 0,27 | écarté | = | = |
| n=200 θ=0,05 | 0,43 | écarté | = | = |
| n=200 θ=0,15 | 0,93 | conservé | 0,044 → 0,049 s | 3,6e-4 → 1,4e-4 |
| n=200 θ=0,30 | 0,997 | conservé | 0,032 → 0,025 s | 4,3e-3 → 4,0e-5 |
| n=200 θ=0,60 | 1,00 | conservé | 0,011 → 0,015 s | 1,7e-3 → 3,6e-5 |
| n=1000 θ=0,02 | 0,09 | écarté | = | = |
| n=1000 θ=0,05 | 0,29 | écarté | = | = |
| n=1000 θ=0,15 | 0,91 | conservé | 2,35 → 1,35 s | 1,8e-3 → 1,2e-4 |
| n=1000 θ=0,30 | 0,995 | conservé | 1,13 → 0,82 s | 4,0e-3 → 2,3e-4 |
| n=1000 θ=0,60 | 1,00 | conservé | 0,72 → 0,78 s | 2,2e-3 → 1,8e-4 |

Erreurs rapportées à sd(y), référence `predict` (Cholesky). Erreur d'écart-type
inchangée (~1e-3, fixée par `sqrt(tol)`). Dans presque tous les cas la résolution
de la moyenne **atteint max_iter** (tol 1e-8 inatteignable) : c'est le poste de
coût dominant restant.

### 4.2 Préconditionnement de LLIterative (non activé automatiquement ; `harness/llp.cpp`)

Rang 50 vs 0, `LLIterative(30,k,24,2,1e-8)` :

| Cas | Biais ll (k=0 → k=50) | Erreur gradient (k=0 → k=50) | Temps |
|---|---|---|---|
| n=160 θ=0,3 | 14 % → 0,8 % | 0,16 → 0,13 | ×2,5 |
| n=160 θ=0,6 | 16 % → 0,8 % | 1,5 → 2,1 | ×2,5 |
| n=500 θ=0,3 | 29 % → 9 % | 0,16 → 0,38 | ×1,45 |
| n=500 θ=0,6 | 31 % → 9 % | 0,81 → 2,7 | ×1,45 |

Le gradient (estimateur de Hutchinson, 30 sondes) n'est pas amélioré : option
laissée explicite.

### 4.3 Warm-start de l'optimiseur (`harness/wo.cpp`)

Fit BFGS, n=300, `ROUGH=1`, `LLIterative(30,50,24,10,1e-8)`, θ borné
(`Optim::set_theta_upper_factor(0.5)`, `THUP=0.5`) :

| | évaluations | itérations CG | temps | θ final |
|---|---|---|---|---|
| sans warm-start | 48 | 140 326 | 18,95 s | (0,2574 ; 0,2557) |
| avec warm-start | 26 | 57 549 | 9,04 s | (0,2575 ; 0,2557) |

Référence exacte (`LL`) : θ = (0,192 ; 0,203). Sans borne, le fit dérive vers la
borne haute (θ≈7) avec ou sans warm-start ; le warm-start peut alors changer le θ
final (tous deux faux) — cf. §5.

---

## 5. Problèmes antérieurs mis en évidence (non traités)

1. **Dérive du fit LLIterative vers la borne haute de θ** (θ≈7 contre ≈0,19 exact),
   reproduite sur la branche d'origine (build séparé, même banc). Dans ce régime
   toutes les résolutions épuisent `max_iter` et l'objectif n'est plus fiable.
2. **Biais SLQ** : ll itérative 381 contre 441,46 exacte (n=160, θ={0,25;0,3}) ;
   jusqu'à 31 % à n=500. Gradient souvent faux de >100 %.
3. **Précision de `updateIterative`** : jusqu'à 90 % d'erreur relative sur
   R⁻¹[F|y] à θ=0,3 avec le budget par défaut.
4. **Résolution de la moyenne à tol 1e-8 inatteignable** dans la plupart des cas
   réalistes → consomme tout `max_iter`.
5. `tol` relatif au résidu : un faible résidu ne garantit pas une faible erreur
   (cf. §2.1, n=160 : résidu plus faible à froid, erreur plus faible à chaud).

---

## 6. Plan de test GPU (à exécuter dans Claude Code)

Pour chaque backend disponible :

```sh
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA_ITERATIVE=ON      # ou ENABLE_HIP_ITERATIVE / ENABLE_SYCL_ITERATIVE / ENABLE_METAL_ITERATIVE
cmake --build build-gpu -j
ctest --test-dir build-gpu --output-on-failure
```

Vérifications, par ordre de priorité :

1. **Compilation** des 4 fichiers modifiés (appels `cg_beta*_launch` /
   `batched_update_p_launch` dans la branche périodique ; variables `d_beta`/`dBeta`
   déjà déclarées dans ces fonctions).
2. **Non-régression** : `ctest` complet, en particulier `[iterative]`,
   `[predictiterative]`, et les tests GPU spécifiques.
3. **Convergence CPU vs GPU** sur les cas de §2.2 et §4.1 : relancer
   `harness/cg.cpp`, `harness/ll.cpp`, `harness/pc.cpp` avec le backend actif
   (`LK_DEBUG_CG_ITERS=1`) et comparer itérations, erreur, temps.
   Attendu : même amélioration qu'en CPU (plus de plateau à 5e-5).
   Point de vigilance : sur GPU, le test de tolérance porte sur le résidu récursif
   entre deux recalculs — vérifier qu'aucune colonne n'est déclarée convergée à tort.
4. **CUDA mixed precision** (`LK_ITERATIVE_CUDA_MIXED_PRECISION=1`) : la branche
   périodique est l'étape de correction fp64 ; vérifier que garder la direction ne
   dégrade pas (sinon envisager de réinitialiser seulement en mixed precision).
5. **Metal (fp32)** : comportement sur cond(R) ≥ 1e6 ; comparer à l'ancien
   comportement (`git stash` ou build de `rebase/cg-predict-on-v1.2.2`).
6. **Préconditionnement auto + GPU** (predictIterative) : vérifier que le chemin
   `LK_PRED_GPU_BIND` avec `woodbury_pc` fonctionne et comparer temps/erreur avec
   `LK_CG_AUTO_PRECOND=0`.
7. **Warm-start optimiseur + GPU** : mesurer avec `LK_CG_WARM_OPTIM=0/1` ; sur les
   chemins fusionnés `[F|y|sondes]`, le x0 n'est pas passé (gain attendu nul) —
   envisager de l'ajouter (`conjugateGradient(..., x0)` existe déjà côté GPU).
8. **bench/gpu/bench_gpu.py** : relancer ; les résultats publiés dans le README
   (notamment la mise à l'échelle n=4000→8000) sont susceptibles de changer.

Comparaison avant/après : construire `rebase/cg-predict-on-v1.2.2` dans un worktree
séparé (`git worktree add ../orig rebase/cg-predict-on-v1.2.2`) et compiler les
bancs contre chaque arbre.

---

## 7. Bancs d'essai (`harness/`)

Compilation : `LK_BUILD=/chemin/build ./build.sh <nom>` (réutilise les flags de la
cible `KrigingPredictIterativeTest`). `ws.cpp`, `cg.cpp` et `up.cpp` accèdent aux
membres privés via `#define private public` (diagnostic uniquement).

| Fichier | Rôle | Exemple |
|---|---|---|
| `ws.cpp` | warm vs cold dans predictIterative, erreur vs Cholesky | `./ws 160 0.3 "LLIterative(30,0,24,2,1e-10)" mf` (`mf`/`de` = sans matrice / dense) |
| `cg.cpp` | variantes de CG sur R exacte, cold/warm | `./cg 160 0.25` |
| `ll.cpp` | ll et gradient, sans matrice vs dense | `./ll 40` (arg = mult) |
| `llp.cpp` | biais LLIterative avec/sans préconditionneur | `./llp 500 0.3` |
| `pc.cpp` | predictIterative, préconditionneur off/on | `LK_CG_AUTO_PRECOND=0 ./pc 1000 0.3` |
| `wo.cpp` | fit BFGS (warm-start optimiseur), update | `THUP=0.5 ROUGH=1 LK_CG_WARM_OPTIM=1 LK_DEBUG_CG_ITERS=1 ./wo fit 300 "LLIterative(30,50,24,10,1e-8)" BFGS \| awk -f sum.awk` |
| `up.cpp` | précision de R⁻¹[F|y] après update | `./up 300 0.3 40 "LLIterative(30,0,24,2,1e-8)"` |
| `sum.awk` | somme des itérations `LK_DEBUG_CG_ITERS` | |

Variables d'environnement utiles : `LK_ITERATIVE_DENSE_MAX_MB` (0 = force sans
matrice), `LK_DEBUG_CG_ITERS`, `LK_CG_AUTO_PRECOND`, `LK_CG_AUTO_PRECOND_MIN_CAPTURED`,
`LK_CG_WARM_OPTIM`, `LK_ITERATIVE_CUDA_MIXED_PRECISION`.

---

## 8. Suite proposée

1. Tests GPU (§6) ; corriger les backends si nécessaire.
2. Détection de stagnation fiable (pour ne plus consommer `max_iter` quand `tol` est
   inatteignable), par exemple fondée sur l'estimation d'erreur en norme A via les
   coefficients de Lanczos du CG (estimateur de Hestenes–Stiefel / Strakoš–Tichý)
   plutôt que sur le résidu.
3. Biais SLQ / gradient de LLIterative (§5.1, §5.2) : priorité avant toute
   optimisation de performance ; SLQ préconditionné, plus de pas de Lanczos,
   nombre de sondes, et vérification de la cohérence ll/gradient.
4. Warm-start sur les chemins GPU fusionnés.
5. Point non traité de la stratégie initiale : LOVE (variance prédictive par
   décomposition de Lanczos de rang faible) pour `predictIterative(return_stdev=true)`.
6. Point écarté à la demande : bascule automatique Cholesky/CG quand R est dense en
   mémoire.
