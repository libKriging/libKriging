# Points de contact : fichiers à créer / modifier

Relevé sur l'arbre réel du dépôt. `C` = à créer, `M` = à modifier.
Le patron à copier partout est **`NestedKriging`** (classe de composition la
plus récente et la plus proche structurellement).

## 1. Cœur C++

| | Fichier | Note |
|---|---|---|
| C | `src/lib/include/libKriging/MarkovCoKriging.hpp` | esquisse dans `draft/` |
| C | `src/lib/MarkovCoKriging.cpp` | ~600–900 l. attendues (réf. `NestedKriging.cpp` = 526 l.) ; couvre AR(1) **et** collocalisé (même classe, cf. `DESIGN.md` §0/§7bis) |
| M | `src/lib/CMakeLists.txt` | ajouter la paire `.cpp` / `.hpp` + lien vers `lbfgsb_cpp` (déjà vendorisé, cf. §7bis) |
| M | `src/lib/KrigingLoader.cpp` / `.hpp` | seulement si `save`/`load` est dans le scope v1 |

~~`Trend.hpp/.cpp` (chemin `F` custom)~~ et ~~`KrigingImpl.cpp/.hpp`
(accepter un `F` fourni)~~ : **retirés du scope** — D1 verrouillée sur
l'option (b), aucune modification du core requise (`DESIGN.md` §6 D1,
§7bis).

Rien d'autre dans `src/` : `Covariance`, `Optim`, `Random`, `LinearAlgebra`
sont réutilisés tels quels, sans modification.

## 2. Tests C++

| | Fichier |
|---|---|
| C | `tests/MarkovCoKrigingTest.cpp` |
| C | `tests/references/…` (valeurs de référence issues de `MuFiCokriging`) |
| M | `tests/CMakeLists.txt` — copier le bloc `NestedKrigingTest` (lignes ~354-360) : `add_executable` + `target_link_libraries(… Kriging Catch2)` + `add_dependencies(all_test_binaries …)` + `catch_discover_tests` |

## 3. Binding Python (pybind11 + carma) — *à faire en premier*

| | Fichier |
|---|---|
| C | `bindings/Python/pylibkriging/src/_pylibkriging/MarkovCoKriging_binding.cpp` |
| C | `bindings/Python/pylibkriging/src/_pylibkriging/MarkovCoKriging_binding.hpp` |
| M | `bindings/Python/pylibkriging/src/_pylibkriging/pylibkriging.cpp` (enregistrement du module) |
| M | `bindings/Python/pylibkriging/CMakeLists.txt` |
| C | `bindings/Python/pylibkriging/tests/MarkovCoKriging_test.py` |
| M | `bindings/Python/pylibkriging/tests/CMakeLists.txt` |
| C | `bindings/Python/markovcokriging_branin2d_py.ipynb` (notebook démo, convention du dépôt) |

## 4. Binding R (Rcpp)

| | Fichier |
|---|---|
| C | `bindings/R/rlibkriging/src/MarkovCoKriging_binding.cpp` |
| M | `bindings/R/rlibkriging/src/RcppExports.cpp` (régénéré par `Rcpp::compileAttributes()`) |
| C | `bindings/R/rlibkriging/R/MarkovCoKrigingClass.R` |
| M | `bindings/R/rlibkriging/R/RcppExports.R` (régénéré) |
| M | `bindings/R/rlibkriging/R/allGenerics.R` (si nouvelles génériques S3) |
| M | `bindings/R/rlibkriging/NAMESPACE` |
| C | `bindings/R/rlibkriging/man/*.Rd` (compter ~10-15 fichiers, cf. le volume existant pour `WarpKriging`) |
| C | `bindings/R/rlibkriging/tests/…` |
| C | `bindings/R/markovcokriging_branin2d_r.ipynb` |

## 5. Binding Julia (ABI C plate — le plus pénible avec Octave)

| | Fichier |
|---|---|
| M | `bindings/Julia/jlibkriging/csrc/libkriging_c.cpp` — **le point dur** : passer une liste de matrices via l'ABI C (tableau de `double*` + tableaux de `n_t`), ou éviter le problème via la variante `level` (cf. `DESIGN.md` D3) |
| M | `bindings/Julia/jlibkriging/src/jlibkriging.jl` |
| C | `bindings/Julia/jlibkriging/tests/multi_fidelity_kriging_test.jl` |
| M | `bindings/Julia/jlibkriging/CMakeLists.txt` |
| C | `bindings/Julia/markovcokriging_branin2d_julia.ipynb` |

⚠ Rappel `AGENTS.md` : le FFI Julia attend exactement `Matrix{Float64}` /
`Vector{Float64}`, aucune promotion automatique.

## 6. Binding Octave / MATLAB (mex)

| | Fichier |
|---|---|
| C | `bindings/Octave/mlibkriging/MarkovCoKriging_binding.cpp` |
| C | `bindings/Octave/mlibkriging/MarkovCoKriging_binding.hpp` |
| C | `bindings/Octave/mlibkriging/MarkovCoKriging.m` |
| M | `bindings/Octave/mlibkriging/mLibKriging.cpp` (dispatch) |
| M | `bindings/Octave/mlibkriging/CMakeLists.txt` |
| C | `bindings/Octave/mlibkriging/tests/MarkovCoKriging_test.m` |

⚠ Rappel `AGENTS.md` : les arguments entiers (`nsim`, `seed`, et ici
potentiellement `level`) doivent être passés en `int32(...)` ; la couche mex
ne convertit pas.

## 7. Documentation

| | Fichier |
|---|---|
| M | `skills/libkriging/SKILL.md` — nouvelle branche §1 de l'arbre de décision : « plusieurs niveaux de fidélité ? » ; nouvelle section options `rho` / niveaux |
| M | `skills/libkriging/references/cpp.md` |
| M | `skills/libkriging/references/python.md` |
| M | `skills/libkriging/references/r.md` |
| M | `skills/libkriging/references/julia.md` |
| M | `skills/libkriging/references/octave-matlab.md` |
| M | `CHANGELOG.md` |
| M | `AGENTS.md` — ajouter `MarkovCoKriging` à la liste des classes de la section « API usage guidance » |
| M | `bindings/R/rlibkriging/DiceKriging.md` — table de correspondance, si pertinent vis-à-vis de `MuFiCokriging` |

## 8. Récapitulatif de l'effort

| Poste | Fichiers | Estimation |
|---|---|---|
| Cœur C++ + tests | ~4 (`Trend`/`KrigingImpl` retirés) | 1 semaine (option (b) verrouillée, pas de bascule d'option ni de risque de régression core) |
| Python | ~7 | 3–4 j |
| R | ~20 (dont beaucoup de `.Rd`) | 4–5 j |
| Julia | ~5 | 3–4 j (ABI C) |
| Octave/MATLAB | ~6 | 3–4 j (mex) |
| Documentation | ~9 | 2 j |

**Le poste dominant est le marshalling multi-bindings, pas les
mathématiques.** C'est ce qui rend la décision D3 (`DESIGN.md`) — liste de
matrices vs. vecteur `level` — la plus rentable à trancher correctement.
