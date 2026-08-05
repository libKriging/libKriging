# Plan d'implémentation

Chaque phase a un **critère de sortie** vérifiable. Ne pas démarrer la
phase `n+1` avant de l'avoir atteint.

---

## Phase 0 — Arbitrages (aucun code)

- [ ] **D1** : traitement de `ρ` → option (a) `F` custom dans `KrigingImpl`,
      ou (b) profilage externe. *(proposition : (b) puis (a))*
- [ ] **D2** : plans non-emboîtés → refus strict, mode approché optionnel,
      ou co-krigeage complet. *(proposition : refus strict en v1)*
- [ ] **D3** : signature `fit` → `vector<vec>/vector<mat>` ou
      `(y, X, level)`. **Décision la plus rentable** : elle conditionne le
      coût des 5 bindings, poste dominant du projet.
- [ ] **D4** : options par niveau (scalaire diffusé vs. vecteur).
- [ ] **D5** : nom de la classe. *(proposition : `MultiFidelityKriging`)*
- [ ] Scope v1 : `update` dedans ou dehors ? `save`/`load` dedans ou dehors ?

**Sortie** : `DESIGN.md` §6 mis à jour, chaque décision marquée TRANCHÉE
avec sa justification.

---

## Phase 1 — Oracle de validation (avant tout code C++)

- [ ] Installer `MuFiCokriging` (R) — cf. `REFERENCES.md`.
- [ ] Produire un cas test reproductible : fonction analytique
      2-fidélités 1-D de Forrester et al. (2007), puis un cas 2-D.
- [ ] Figer les sorties de référence (`theta`, `sigma2`, `rho`, `beta`, LL,
      moyennes et écarts-types de prédiction sur une grille) dans
      `tests/references/`.

**Sortie** : jeu de références figé, indépendant de toute implémentation
libKriging. *Rationale : sans oracle, impossible de distinguer un bug
d'un désaccord de convention.*

---

## Phase 2 — Prototype C++ hors arbre

- [ ] Implémenter le fit récursif en option (b) dans un fichier isolé
      (peut rester dans `wip2/draft/`).
- [ ] Vérifier la reproduction des références de la phase 1 à la tolérance
      des tests existants du dépôt.
- [ ] Vérifier le cas dégénéré `s = 1` ≡ `Kriging`.
- [ ] Vérifier le cas `ρ = 0` ≡ `Kriging` sur la seule haute-fidélité.

**Sortie** : formules validées numériquement. *C'est le seul jalon qui peut
invalider la conception ; tout ce qui suit est de l'intégration.*

---

## Phase 3 — Intégration au cœur

- [ ] `src/lib/include/libKriging/MultiFidelityKriging.hpp` (depuis `draft/`).
- [ ] `src/lib/MultiFidelityKriging.cpp`.
- [ ] Enregistrement dans `src/lib/CMakeLists.txt`.
- [ ] `tests/MultiFidelityKrigingTest.cpp` + bloc dans `tests/CMakeLists.txt`.
- [ ] `predict`, `logLikelihood`, `summary`.
- [ ] `simulate` récursif (graine propagée trajectoire par trajectoire).
- [ ] `update` (si dans le scope v1).
- [ ] Si option (a) retenue : chemin `F` custom dans `KrigingImpl` — **faire
      passer la totalité de la suite de tests existante** avant et après,
      c'est du code partagé par toutes les classes.

**Sortie** : `ctest` vert, y compris toute la suite préexistante.

---

## Phase 4 — Bindings, dans cet ordre

Python sert de terrain d'essai à la convention de marshalling ; ne pas
commencer par Julia ou Octave.

- [ ] **Python** (pybind11/carma) + test + notebook `…_branin2d_py.ipynb`.
- [ ] **R** (Rcpp) + `.Rd` + `NAMESPACE` + test + notebook.
- [ ] **Julia** (ABI C plate) + test + notebook.
- [ ] **Octave/MATLAB** (mex) + test.
- [ ] Test de cohérence inter-bindings : mêmes entrées → mêmes sorties
      (cf. `bindings/Python/pylibkriging/tests/binding_consistency_test.py`).

**Sortie** : chaque binding reproduit les références de la phase 1.

---

## Phase 5 — Documentation

- [ ] `skills/libkriging/SKILL.md` : branche multi-fidélité dans l'arbre §1.
- [ ] Les 5 `skills/libkriging/references/*.md`.
- [ ] `CHANGELOG.md`, `AGENTS.md`.
- [ ] Documenter explicitement les pièges : ordre des niveaux (`t=1` = plus
      basse fidélité), exigence d'emboîtement, `normalize` global.

---

## Rappels de politique du dépôt

- **Ne jamais faire `git push`** (cf. `AGENTS.md`) — demander à l'utilisateur.
- Sous-modules obligatoires : `git submodule update --init --recursive`.
- `ENABLE_OCTAVE_BINDING` et `ENABLE_MATLAB_BINDING` sont **mutuellement
  exclusifs**.
- `ARMA_32BIT_WORD` doit rester cohérent entre le cœur et le binding R.
