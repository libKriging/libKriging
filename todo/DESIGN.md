# Conception : `MultiFidelityKriging`

## 1. Modèle

`s` niveaux de fidélité, `t = 1` le moins cher / le moins fidèle,
`t = s` le code de référence.

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

### D1. Traitement de `ρ` : option (a) ou (b) ?
Voir §2. Impact : modification ou non de `KrigingImpl`.
**Proposition : (b) pour le prototype, (a) pour la version finale.**

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
`MultiFidelityKriging` ? `CoKriging` ? `MuFiKriging` ?
Le nom se propage dans 5 bindings + doc ; le changer après coup est coûteux.
`MultiFidelityKriging` est explicite et cohérent avec `NestedKriging` /
`WarpKriging` / `MLPKriging`.

## 7. Invariants et contraintes à faire respecter

- `X_t` est toujours `n_t × d`, `d` **identique** à tous les niveaux ;
  `y_t` est un vecteur de longueur `n_t` (convention du dépôt, cf. `AGENTS.md`).
- `normalize` : global uniquement, jamais par niveau (§5 de `ANALYSIS.md`).
- Ordre des niveaux : `t=1` = plus basse fidélité. **Documenter clairement**,
  c'est une source d'erreur classique (certains outils numérotent à l'envers).
- `s ≥ 2` ; `s = 1` doit soit être refusé, soit dégénérer proprement en
  `Kriging`.
