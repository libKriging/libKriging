# Conception : `MarkovCoKriging`

## 0. Généralisation retenue : AR(1) et co-krigeage collocalisé, même code

Le modèle ci-dessous n'est pas spécifique à la « fidélité » : c'est un cas
de **co-krigeage de type Markov** (Journel 1999, MM1/MM2 — cf.
`REFERENCES.md`), dont l'hypothèse structurante est que la covariance
croisée entre deux champs est **proportionnelle** à une covariance directe.
L'AR(1) de Le Gratiet en est l'instance hiérarchique (`t` = niveau de
fidélité, plans emboîtés par coût croissant). Le **co-krigeage collocalisé**
(Xu et al. 1992) en est l'instance à `s=2` **sans notion d'ordre de
fidélité** : `f_1` = variable secondaire connue sur un plan large, `f_2` =
variable primaire sous-échantillonnée, avec la même contrainte structurale
`D_2 ⊆ D_1` (renommée, pas de coût/fidélité impliqué).

**Conséquence pour l'implémentation : les deux cas partagent exactement le
même code.** Seule la sémantique du mot « niveau » change (position dans
une chaîne de dépendance de covariance, pas nécessairement un ordre de
coût). Voir §7bis pour le détail de l'implémentation « expositions
uniquement » qui découle de ce constat.

## 1. Modèle

`s` champs reliés en chaîne par une hypothèse de covariance croisée
proportionnelle. Dans le cas AR(1), `t = 1` est le moins cher / le moins
fidèle, `t = s` le code de référence ; dans le cas collocalisé, `s = 2` et
`t = 1` est simplement la variable secondaire.

    Z_1(x) ~ GP( f_1(x)ᵀ β_1 , σ_1² r_1(·,·;θ_1) )
    Z_t(x) = ρ_{t-1}(x) · Z_{t-1}(x) + δ_t(x)          t = 2..s
    δ_t    ~ GP( f_t(x)ᵀ β_t , σ_t² r_t(·,·;θ_t) ),   δ_t ⊥ Z_{t-1}

Formes de `ρ` à supporter :

| Forme | Paramètres | Usage |
|---|---|---|
| `"constant"` | 1 | défaut, cas le plus courant |
| `"linear"` : `ρ(x) = ρ_0 + Σ_k ρ_k x_k` | `d+1` | biais dépendant de x |
| plus généralement `ρ(x) = g(x)ᵀ ρ` | `dim(g)` | même arbre de bases que `Trend::RegressionModel` |

## 2. Estimation — la factorisation

**Hypothèse structurante : plans emboîtés** `D_s ⊆ D_{s-1} ⊆ … ⊆ D_1`.

Sous cette hypothèse, la vraisemblance jointe de `(y_1, …, y_s)` se factorise :

    L(y_1,…,y_s) = L_1(y_1) · Π_{t=2..s} L_t( y_t | y_{t-1}(D_t) )

et chaque facteur `L_t` est **exactement** la vraisemblance d'un krigeage
ordinaire sur `D_t` avec :

- observations  `y_t`
- matrice de régression  `F_t = [ diag(y_{t-1}(D_t)) · G(D_t) | F(D_t) ]`
  où `G` est la base de `ρ` et `F` la base de tendance de `δ_t`
- coefficients  `β̃_t = [ ρ , β_t ]`

Donc `ρ` et `β_t` sortent ensemble du même GLS profilé, et `σ_t²` du même
profilage que d'habitude. **Aucune optimisation supplémentaire** — c'est
l'option (a).

### Option (a) — matrice `F` fournie par l'appelant

Nécessite d'ouvrir un chemin dans `KrigingImpl` acceptant un `F` explicite
au lieu de le dériver de `Trend::RegressionModel`.

- Avantages : formulation exacte, `ρ` gratuit, incertitude sur `ρ` incluse
  dans la variance via le terme GLS habituel.
- Coût : modification du cœur partagé par `Kriging`, `WarpKriging`,
  `MLPKriging`, `NestedKriging` → risque de régression, à couvrir par les
  tests existants.
- Effet de bord positif : un `F` custom est aussi ce dont on aurait besoin
  pour d'autres extensions (tendances utilisateur, bases physiques).

### Option (b) — profilage externe de `ρ`

Boucle externe sur `ρ` ; pour chaque `ρ` candidat, fit d'un `Kriging`
standard sur `z_t = y_t − ρ(D_t) ⊙ y_{t-1}(D_t)` ; on retient le `ρ`
maximisant la LL concentrée.

- Avantages : **zéro modification du core**, prototypage immédiat.
- Coût : une optimisation 1-D (ou `dim(g)`-D) par niveau imbriquée autour
  d'une optimisation d'hyperparamètres ; incertitude sur `ρ` non propagée.

> **Recommandation** : prototyper en (b), valider numériquement contre
> `MuFiCokriging`, puis basculer en (a).

## 3. Prédiction

Récursion, du niveau 1 vers le niveau `s` :

    μ_1(x)  = krigeage ordinaire du sous-modèle 1
    s²_1(x) = idem

    μ_t(x)  = ρ_{t-1}(x) · μ_{t-1}(x) + μ_{δt}(x)
    s²_t(x) = ρ_{t-1}(x)² · s²_{t-1}(x) + s²_{δt}(x)

En version plug-in (`ρ` connu). En version bayésienne (v2), un terme
supplémentaire apparaît, fonction de la variance a posteriori de `ρ`.

`predict` doit rendre `(mean, stdev)` a minima ; la signature de `Kriging`
rend `(mean, stdev, cov, mean_deriv, stdev_deriv)` — décider si on aligne
(les dérivées se propagent aussi par la récursion, mais c'est du travail
supplémentaire).

## 4. Simulation

Récursivement : simuler `nsim` trajectoires de `Z_1` aux points demandés,
puis pour chaque `t`, simuler `δ_t` et composer
`Z_t = ρ_{t-1} ⊙ Z_{t-1} + δ_t`, **avec la même graine propagée**
trajectoire par trajectoire. C'est la seule façon d'obtenir des
trajectoires jointes cohérentes.

## 5. Mise à jour (`update`)

Ajout de `(x_new, y_new)` **au niveau `t`** :

- niveaux `< t` : inchangés ;
- niveau `t` : `update` du sous-modèle ;
- niveaux `> t` : la relation d'emboîtement peut être rompue, et `y_{t}(D_{t+1})`
  a changé si `x_new ∈ D_{t+1}` → refit des niveaux supérieurs.

C'est le cas d'usage industriel n°1 (enrichissement adaptatif, co-EGO :
« quel point, à quel niveau de fidélité ? »). À ne pas sacrifier.

## 6. Décisions ouvertes (BLOQUANTES — à trancher avant tout code)

### D1. Traitement de `ρ` : TRANCHÉE — option (b), **définitivement**
Voir §2 et §7bis. Ce n'est plus « prototype (b) puis migration vers (a) » :
**(b) est retenue en permanence**, pas seulement comme étape transitoire.

Justification du changement de position :
- (a) n'apporte de gain que sur la propagation de l'incertitude de `ρ`
  dans la variance prédictive — un raffinement, pas une correction d'erreur.
- (a) coûte une modification du chemin `F` partagé par `Kriging`,
  `WarpKriging`, `MLPKriging`, `NestedKriging` → risque de régression sur
  tout le core pour un gain marginal.
- (b) permet de traiter l'AR(1) **et** le co-krigeage collocalisé avec le
  même code, sans toucher à `KrigingImpl`/`Trend` (§0, §7bis) — l'argument
  décisif.
- La perte (incertitude sur `ρ` non propagée, estimation *plug-in*) est
  documentée comme limitation permanente assumée, au même titre que la
  variante bayésienne de Le Gratiet reste v2/hors-scope.

**Conséquence sur `TOUCHPOINTS.md`** : les lignes « seulement si option (a) »
(`Trend.hpp/.cpp`, `KrigingImpl.cpp/.hpp`) sont retirées du scope.

### D2. Plans non-emboîtés : refus, ou mode approché ?
- *Refus strict* : contrôle runtime, message clair, doc explicite.
  Simple, honnête, mais bloque une majorité de cas industriels réels.
- *Mode approché* : remplacer `y_{t-1}(D_t)` par `μ_{t-1}(D_t)` prédit.
  La factorisation n'est plus exacte et la variance est sous-estimée
  (l'incertitude de prédiction du niveau inférieur n'est pas propagée).
- *Co-krigeage complet* (covariance jointe, sans factorisation) : exact
  pour tout plan, mais `O((Σ n_t)³)` et beaucoup plus de code.

**Proposition : v1 = plans emboîtés exigés + contrôle explicite ;
mode approché derrière une option nommée sans ambiguïté
(`allow_non_nested=true`) documentée comme approximation.**

### D3. Forme de la signature `fit`
- `fit(std::vector<arma::vec> y, std::vector<arma::mat> X, …)` — explicite,
  mais nouvelle convention de marshalling à inventer dans 5 bindings.
- Alternative : `fit(arma::vec y, arma::mat X, arma::uvec level)` — un seul
  jeu de données plus un vecteur d'index de niveau. **Aucune** nouvelle
  convention de marshalling (tout est déjà supporté partout), au prix d'une
  API un peu moins naturelle et d'un découpage interne.

**Proposition : sérieusement envisager la variante `level`** — elle divise
probablement par deux le coût côté bindings, qui est le poste dominant.
À arbitrer avec les utilisateurs cibles.

### D4. Options par niveau
`kernel`, `regmodel`, `optim`, `objective`, `noise` : scalaire diffusé à
tous les niveaux, ou vecteur de taille `s` ? (Un vecteur est plus riche —
typiquement un nugget seulement sur la basse-fidélité stochastique — mais
alourdit encore le marshalling.)
**Proposition : accepter les deux — taille 1 = diffusion.**

### D5. Nom de la classe
`MarkovCoKriging` ? `CoKriging` ? `MuFiKriging` ?
Le nom se propage dans 5 bindings + doc ; le changer après coup est coûteux.
`MarkovCoKriging` est explicite et cohérent avec `NestedKriging` /
`WarpKriging` / `MLPKriging`.

⚠ Mise à jour suite à §0 : la classe couvre aussi le co-krigeage
collocalisé (`s=2`, pas de notion de coût/fidélité). Deux options :
- garder `MarkovCoKriging` et documenter le cas `s=2` collocalisé
  comme un usage particulier (paramètre `level` réinterprété comme
  « position dans la chaîne », pas « coût ») — **zéro coût de renommage,
  proposition retenue par défaut** ;
- ou introduire un alias plus neutre (`CoKriging`) si le collocalisé
  s'avère un cas d'usage aussi fréquent que l'AR(1) — à trancher seulement
  si la demande utilisateur le justifie, pas maintenant.

## 7bis. Plan d'implémentation « expositions uniquement » (option b verrouillée)

Découle directement de D1 (§6) : **aucune modification de
`KrigingImpl`/`Trend`/`Covariance`/`Optim`**. Toute la logique nouvelle vit
dans un unique fichier de composition, sur le patron de `NestedKriging`.

### Structure interne

    class MarkovCoKrigingImpl {
      std::vector<std::unique_ptr<Kriging>> m_subKrigings;  // 1 par niveau
      std::vector<arma::vec>                m_rho;          // figé après fit
      …
    };

- **Niveau 1** : `Kriging` standard, inchangé, sur `(X_1, y_1)`.
- **Niveau `t ≥ 2`** : pour un `ρ_t` candidat (scalaire ou `g(x)ᵀρ_t`, cf.
  D4), calcule le résidu `z_t = y_t − ρ_t(D_t) ⊙ ŷ_{t-1}(D_t)`, fit un
  `Kriging` standard sur `(X_t, z_t)` (appel public inchangé), lit sa LL
  concentrée. `ŷ_{t-1}(D_t)` = valeur observée si plan exactement emboîté,
  ou `m_subKrigings[t-1]->predict(D_t)` sinon (mode approché de D2 — déjà
  gratuit, aucun code nouveau requis pour ce cas).

### Optimisation externe de `ρ`

Boucle 1-D (ou `dim(g)`-D) autour du fit ci-dessus, qui maximise la LL
concentrée en `ρ_t`. **Ne pas passer par `Optim.hpp`** : ce module est
spécifique aux bornes/heuristiques de `θ` de `Kriging` et n'expose pas de
minimiseur générique réutilisable (vérifié sur l'API actuelle). À la place,
appeler directement `lbfgsb_cpp` (`dependencies/lbfgsb_cpp/`), **déjà
vendorisé et déjà lié au build** — aucune nouvelle dépendance, aucun
fichier core touché, juste un appel de plus depuis
`MarkovCoKriging.cpp`.

### `predict` / `simulate` / `logLikelihood`

Inchangés par rapport à §3-4 : récursion pure sur `m_subKrigings`, aucun
nouveau besoin côté core.

### Ce qui sort du scope grâce à ce verrouillage

- `Trend.hpp/.cpp` (chemin `F` custom) — supprimé de `TOUCHPOINTS.md`.
- `KrigingImpl.cpp/.hpp` (accepter un `F` fourni) — supprimé.
- Phase 2 de `PLAN.md` (« prototype hors arbre ») devient un prototype
  **définitif**, pas une étape à refaire différemment en Phase 3 : il n'y a
  plus de bascule d'option à opérer, seulement une intégration mécanique.

### Le collocalisé « gratuit »

Le co-krigeage collocalisé (Xu et al. 1992) est le cas `s=2` de cette même
classe, sans autre développement : seule la documentation utilisateur
change (pas de vocabulaire « fidélité »/« coût », juste « variable
secondaire », cf. D5). Premier test de validation naturel : comparer contre
une implémentation `gstat`/gslib du MM1, en plus de l'oracle
`MuFiCokriging` déjà prévu pour l'AR(1) (cf. `PLAN.md` Phase 1).

## 7. Invariants et contraintes à faire respecter

- `X_t` est toujours `n_t × d`, `d` **identique** à tous les niveaux ;
  `y_t` est un vecteur de longueur `n_t` (convention du dépôt, cf. `AGENTS.md`).
- `normalize` : global uniquement, jamais par niveau (§5 de `ANALYSIS.md`).
- Ordre des niveaux : `t=1` = plus basse fidélité. **Documenter clairement**,
  c'est une source d'erreur classique (certains outils numérotent à l'envers).
- `s ≥ 2` ; `s = 1` doit soit être refusé, soit dégénérer proprement en
  `Kriging`.
