# Krigeage multi-sorties

Dossier de travail pour le support de sorties multiples (`Y` n × q) dans
libKriging, avec un focus sur les sorties temporelles / fonctionnelles.

**État : analyse / conception. Aucun code produit dans l'arbre source.**

## Contenu

| Fichier | Rôle |
|---|---|
| `ANALYSIS.md` | Revue biblio (§1), logiciels (§2), feuille de route (§3), sorties temporelles (§4), API (§5), questions ouvertes (§6) |
| `draft/MultiOutputKriging.hpp` | Esquisse d'API C++ — vérifiée par `g++ -fsyntax-only`, non branchée au build |
| `draft/example_python.py` | Exemple d'usage Python de l'API attendue (non exécutable) |

## Reprise rapide

1. Trancher `ANALYSIS.md` §6 (cas d'usage cible, `Σ̂` singulier, format de cov).
2. Suivre `ANALYSIS.md` §5.4 : `"pca"` par composition → généralisation
   `KrigingImpl` à `m_Y` → `"shared"` → `"separable"` → `"separable(<kernel>)"`.

Voisins : GEK (travail local non publié), multi-fidélité (branche
`feature/multi-fidelity-cokriging`).
