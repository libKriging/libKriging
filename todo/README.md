# todo — Co-krigeage de type Markov : AR(1) multi-fidélité (Le Gratiet) + collocalisé

Dossier de travail pour l'implémentation d'un modèle de co-krigeage de type
Markov dans libKriging (`MarkovCoKriging`), couvrant à la fois le
multi-fidélité récursif AR(1) et le co-krigeage collocalisé.

**État : analyse / conception. Aucun code produit dans l'arbre source.**
Rien n'a encore été modifié dans `src/`, `bindings/`, `tests/` ou `skills/`.

## Contenu

| Fichier | Rôle |
|---|---|
| `ANALYSIS.md` | Analyse initiale : ce que l'implémentation implique, effort, séquencement |
| `DESIGN.md` | Formulation mathématique détaillée + décisions de conception à trancher |
| `TOUCHPOINTS.md` | Liste exhaustive des fichiers à créer / modifier (core, 5 bindings, tests, doc) |
| `PLAN.md` | Checklist séquencée, phase par phase, avec critères de sortie |
| `REFERENCES.md` | Bibliographie et oracles de validation |
| `draft/MarkovCoKriging.hpp` | Esquisse d'API C++ — vérifiée syntaxiquement, non branchée au build |

## Reprise rapide

1. **D1 (traitement de `ρ`) est tranchée : option (b) profilage externe,
   définitivement** — zéro modification de `KrigingImpl`/`Trend`, et le
   même code couvre l'AR(1) **et** le co-krigeage collocalisé (Journel
   MM1/MM2, Xu et al. 1992). Voir `DESIGN.md` §0 et §7bis.
2. Lire `DESIGN.md` §« Décisions ouvertes » restantes (D2 plans
   non-emboîtés, D3 signature `fit`, D4 options par niveau, D5 nom).
3. Puis suivre `PLAN.md` phase 0 → 5 (Phase 1 inclut désormais un oracle
   collocalisé en plus de `MuFiCokriging`).

## Contexte projet au moment de l'analyse

- Aucune trace de multi-fidélité / co-krigeage dans le dépôt
  (`grep -ri "fidelity\|cokriging"` → vide hors dépendances).
- Le patron de référence à copier est `NestedKriging` : classe de composition
  sur un `std::vector<std::unique_ptr<Kriging>>`, 526 l. `.cpp` / 176 l. `.hpp`,
  save/load volontairement différé.
