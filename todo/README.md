# wip2 — Multi-fidélité : co-krigeage récursif AR(1) (Le Gratiet)

Dossier de travail pour l'implémentation d'un modèle multi-fidélité dans
libKriging (`MultiFidelityKriging`).

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
| `draft/MultiFidelityKriging.hpp` | Esquisse d'API C++ — vérifiée syntaxiquement, non branchée au build |

## Reprise rapide

1. Lire `DESIGN.md` §« Décisions ouvertes » — 3 arbitrages bloquent tout le reste
   (traitement de `ρ`, plans non-emboîtés, forme de la signature `fit`).
2. Puis suivre `PLAN.md` phase 0 → 5.

## Contexte projet au moment de l'analyse

- Aucune trace de multi-fidélité / co-krigeage dans le dépôt
  (`grep -ri "fidelity\|cokriging"` → vide hors dépendances).
- Le patron de référence à copier est `NestedKriging` : classe de composition
  sur un `std::vector<std::unique_ptr<Kriging>>`, 526 l. `.cpp` / 176 l. `.hpp`,
  save/load volontairement différé.
- `wip/` (dossier voisin) contient une série de patches GEK (krigeage avec
  gradients) — non lié, mais même convention de dossier de travail.
