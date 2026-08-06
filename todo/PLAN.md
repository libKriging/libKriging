# Plan d'implémentation

Chaque phase a un **critère de sortie** vérifiable. Ne pas démarrer la
phase `n+1` avant de l'avoir atteint.

---

## Phase 0 — Arbitrages (aucun code)

- [x] **D1** : traitement de `ρ` → **TRANCHÉE : option (b) profilage
      externe, définitivement** (pas de bascule vers (a) prévue). Permet de
      couvrir AR(1) et co-krigeage collocalisé avec le même code, sans
      toucher `KrigingImpl`/`Trend`. Voir `DESIGN.md` §6 D1, §7bis.
- [ ] **D2** : plans non-emboîtés → refus strict, mode approché optionnel,
      ou co-krigeage complet. *(proposition : refus strict en v1 ; le mode
      approché reste gratuit à activer ensuite, cf. §7bis — il réutilise
      `predict` du niveau parent, déjà nécessaire de toute façon)*
- [ ] **D3** : signature `fit` → `vector<vec>/vector<mat>` ou
      `(y, X, level)`. **Décision la plus rentable** : elle conditionne le
      coût des 5 bindings, poste dominant du projet.
- [ ] **D4** : options par niveau (scalaire diffusé vs. vecteur).
- [ ] **D5** : nom de la classe. *(proposition : garder `MarkovCoKriging`,
      documenter le cas `s=2` collocalisé comme usage particulier — voir
      `DESIGN.md` D5)*
- [ ] Scope v1 : `update` dedans ou dehors ? `save`/`load` dedans ou dehors ?

**Sortie** : `DESIGN.md` §6 mis à jour, chaque décision marquée TRANCHÉE
avec sa justification. *(D1 fait ; D2/D3/D4/D5 restent à statuer.)*

---

## Phase 1 — Oracles de validation (avant tout code C++)

- [ ] Installer `MuFiCokriging` (R) — cf. `REFERENCES.md`. Oracle AR(1).
- [ ] Produire un cas test reproductible : fonction analytique
      2-fidélités 1-D de Forrester et al. (2007), puis un cas 2-D.
- [ ] Figer les sorties de référence (`theta`, `sigma2`, `rho`, `beta`, LL,
      moyennes et écarts-types de prédiction sur une grille) dans
      `tests/references/`.
- [ ] Cas test collocalisé (`s=2`, pas de fidélité) : oracle MM1/MM2 via
      `gstat` (R) ou gslib, cf. `REFERENCES.md` §Généralisation. Même
      format de références figées.

**Sortie** : jeux de références figés (AR(1) + collocalisé), indépendants
de toute implémentation libKriging. *Rationale : sans oracle, impossible de
distinguer un bug d'un désaccord de convention.*

---

## Phase 2 — Prototype C++ (définitif, plus « hors arbre » à refaire)

Depuis le verrouillage de D1 sur l'option (b) (`DESIGN.md` §7bis), ce
prototype n'est plus une étape jetable : c'est directement le code visé
pour la Phase 3, juste pas encore rattaché au build.

- [ ] Implémenter le fit récursif — profilage externe de `ρ` via
      `lbfgsb_cpp` (déjà vendorisé/lié, cf. `DESIGN.md` §7bis) — dans un
      fichier isolé (`todo/draft/`).
- [ ] Vérifier la reproduction des références AR(1) de la Phase 1 à la
      tolérance des tests existants du dépôt.
- [ ] Vérifier la reproduction des références collocalisées de la Phase 1.
- [ ] Vérifier le cas dégénéré `s = 1` ≡ `Kriging`.
- [ ] Vérifier le cas `ρ = 0` ≡ `Kriging` sur la seule haute-fidélité /
      primaire.

**Sortie** : formules validées numériquement pour les deux cas d'usage
(AR(1) et collocalisé). *C'est le seul jalon qui peut invalider la
conception ; tout ce qui suit est de l'intégration mécanique.*

---

## Phase 3 — Intégration au cœur (allégée : aucune touche à `KrigingImpl`/`Trend`)

- [ ] `src/lib/include/libKriging/MarkovCoKriging.hpp` (depuis `draft/`).
- [ ] `src/lib/MarkovCoKriging.cpp`.
- [ ] Enregistrement dans `src/lib/CMakeLists.txt` (lien vers `lbfgsb_cpp`,
      déjà une dépendance du projet — pas de nouvelle entrée `find_package`).
- [ ] `tests/MarkovCoKrigingTest.cpp` + bloc dans `tests/CMakeLists.txt`
      (cas AR(1) et collocalisé).
- [ ] `predict`, `logLikelihood`, `summary`.
- [ ] `simulate` récursif (graine propagée trajectoire par trajectoire).
- [ ] `update` (si dans le scope v1).

**Sortie** : `ctest` vert, y compris toute la suite préexistante — *aucune
régression possible sur `Kriging`/`WarpKriging`/`MLPKriging`/`NestedKriging`
puisqu'aucun de leurs fichiers n'est modifié.*

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
