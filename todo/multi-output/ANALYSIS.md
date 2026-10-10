# Krigeage multi-sorties : revue biblio, logiciels, pistes pour libKriging

> Analyse du 2026-10-07. État du dépôt : `master` @ `6f50133a`.
> **Aucun code produit.** Voir aussi : GEK (non publié) et multi-fidélité (branche `feature/multi-fidelity-cokriging`, `todo/`).

## 0. Le problème

On observe `q` sorties `y_1..y_q` d'un même code en `n` points `X` (`n × d`).
Aujourd'hui libKriging ne connaît que `y ∈ ℝⁿ` (`KrigingImpl::m_y` est un
`arma::vec`) : l'utilisateur fait `q` fits indépendants dans une boucle.

Deux axes structurent toute la littérature :

| Axe | Valeurs | Conséquence |
|---|---|---|
| Plan d'expériences | **isotopique** (toutes les sorties aux mêmes `X`) / **hétérotopique** (plans différents par sortie, données manquantes) | En isotopique, le couplage inter-sorties apporte très peu à la moyenne prédite (cf. §1.C, « autokrigeabilité ») |
| Taille de `q` | petit (2–20) / grand (sorties fonctionnelles : courbes, champs, séries temporelles, `q` ~ 10²–10⁶) | `q` grand ⇒ seules les approches à `θ` partagés ou par réduction de dimension passent à l'échelle |

Rappel : multi-fidélité (branche `feature/multi-fidelity-cokriging`) et krigeage avec gradients (GEK, travail local non publié) sont
eux-mêmes des **cas particuliers structurés** de multi-sorties
(sorties = niveaux de fidélité, ou `y` + ses `d` dérivées). L'API retenue ici
devrait rester cohérente avec ces deux chantiers.

## 1. Familles de modèles

Notation : `Cov(Y_i(x), Y_j(x')) = K_ij(x, x')`, matrice jointe `nq × nq`.

### A. Sorties indépendantes (référence)

`K_ij = δ_ij σ_i² r(x, x'; θ_i)`. `q` fits, `O(q n³)`. C'est ce que font
DiceKriging, UQLab, SMT, OpenTURNS par défaut. Aucun partage d'information,
mais robuste et trivial. **Déjà possible** dans libKriging (boucle côté
binding).

### B. Hyperparamètres de corrélation partagés — « parallel partial GP »

`K_ij = δ_ij σ_i² r(x, x'; θ)` : **un seul `θ`** pour toutes les sorties,
`β_i` et `σ_i²` propres à chaque sortie.

- Gu & Berger (2016), *Parallel partial Gaussian process emulation for
  computer models with massive output*, Ann. Appl. Stat. 10(3):1317–1347.
- Une seule Cholesky de `R(θ)` : coût `O(n³ + q n²)` au lieu de `O(q n³)`.
- Log-vraisemblance profilée = somme sur `j` des termes en `σ̂_j²` ; le
  gradient en `θ` se somme aussi ⇒ **même optimiseur, même nombre de
  paramètres** qu'un krigeage scalaire.
- Hypothèse forte (même régularité pour toutes les sorties), mais très
  raisonnable pour des sorties fonctionnelles d'un même code (courbe en
  temps, champ spatial).
- C'est aussi ce que fait de facto `sklearn.GaussianProcessRegressor` avec
  `y` 2-D (noyau partagé).

### C. Modèle séparable / Intrinsic Coregionalization Model (ICM)

`K(x, x') = B ⊗ r(x, x'; θ)`, `B` (`q × q`, SDP) = covariance entre sorties.

- Géostat : modèle de corrélation intrinsèque, Wackernagel (2003),
  *Multivariate Geostatistics*, Springer ; Goovaerts (1997).
- Computer experiments : Conti & O'Hagan (2010), *Bayesian emulation of
  complex multi-output and dynamic computer models*, JSPI 140(3) — `B` intégré
  analytiquement ⇒ postérieur matrice-`t` ; Rougier (2008), *Efficient
  emulators for multivariate deterministic functions*, JCGS (« outer
  product emulator », Kronecker aussi dans l'indice de sortie).
- ML : Bonilla, Chai & Williams (2008), *Multi-task Gaussian process
  prediction*, NeurIPS.

Propriétés clés :

1. **Isotopique** : `B̂ = (Y − Fβ̂)ᵀ R⁻¹ (Y − Fβ̂) / n` en forme close ⇒ le
   fit se ramène **exactement à B** (θ partagés) plus le calcul de `B̂`.
2. **Autokrigeabilité** (Wackernagel) : en isotopique, la moyenne prédite de
   chaque sortie est *identique* à celle du modèle B. Le gain de l'ICM porte
   alors uniquement sur la **covariance croisée prédictive** — utile pour
   `simulate` conjoint (trajectoires cohérentes entre sorties), pour des
   combinaisons linéaires de sorties, ou pour l'optimisation multi-objectif.
3. **Hétérotopique** : là, le couplage améliore réellement la prédiction
   (une sortie peu observée emprunte l'information des autres) ; mais il faut
   la Cholesky de la matrice jointe `N × N`, `N = Σ_j n_j` (structure de
   Hadamard, plus de Kronecker).

### D. Linear Model of Coregionalization (LMC) et variantes

`K(x, x') = Σ_{k=1..Q} B_k ⊗ r_k(x, x'; θ_k)` — chaque sortie est un mélange
linéaire de `Q` processus latents indépendants de portées différentes.

- Journel & Huijbregts (1978) ; Goulard & Voltz (1992) pour l'ajustement
  sous contrainte SDP ; Wackernagel (2003).
- SLFM, rang faible : Teh, Seeger & Jordan (2005), *Semiparametric latent
  factor models*, AISTATS.
- OILMM : Bruinsma et al. (2020), *Scalable exact inference in multi-output
  Gaussian processes*, ICML — mélange orthogonal ⇒ découplage exact en `Q`
  GPs indépendants, coût linéaire en `q`.
- Coût générique `O((nq)³)`, estimation délicate (paramétrer `B_k = L_k L_kᵀ`,
  multimodalité, identifiabilité). Peu de gain démontré en computer
  experiments vs. B/C quand le plan est isotopique.

**Ce que le LMC apporte réellement (vs. ICM)** — le gain n'existe que si les
sorties n'ont pas la même régularité, ou si leur corrélation dépend de
l'échelle :

1. *Régularité/portée propre à chaque sortie* : chaque sortie mélange
   différemment des latents de portées différentes (une lisse, une rugueuse).
   L'ICM impose le même `r(·;θ)` à toutes ; A le permet mais sans couplage.
2. *Corrélation inter-sorties dépendant de l'échelle* : deux sorties peuvent
   être corrélées à grande échelle et décorrélées à courte échelle (cohérence
   variable en fréquence) ; dans l'ICM la cohérence est constante. Typique de
   grandeurs physiques différentes issues d'un même code.
3. *Casse l'autokrigeabilité* : dès `Q ≥ 2` portées distinctes, le
   co-krigeage modifie la moyenne prédite même en isotopique — seul modèle de
   cette liste à le faire. Gain souvent modeste en isotopique, net en
   hétérotopique.
4. *Co-krigeage hétérotopique classique* (variable secondaire dense ⇒
   primaire rare) : déjà possible en ICM, mieux traité par le LMC quand la
   relation n'est vraie qu'à certaines échelles.
5. *Cadre unificateur* : `Q = 1` ⇒ ICM ; `B_k` diagonales ⇒ indépendant à
   noyaux mélanges ; `B_k` triangulaires ⇒ AR(1) de Kennedy & O'Hagan à `ρ`
   constant (la multi-fidélité est un LMC). Interprétation par composantes
   (krigeage factoriel de Matheron : filtrage d'une composante de courte
   portée).

Prix : `Q·q(q+1)/2 + Q·d` paramètres (vs `d + q(q+1)/2` pour l'ICM) —
risque de sur-ajustement réel pour les `n` usuels en computer experiments ;
Cholesky jointe `O((nq)³)` sans Kronecker exploitable ; contrainte SDP par
`B_k`, permutations non identifiables, portées qui se confondent,
vraisemblance multimodale (multistart et bornes soignées indispensables).

Pour libKriging : le cas où le LMC apporte vraiment quelque chose en
computer experiments est déjà couvert par `MarkovCoKriging` (branche `feature/multi-fidelity-cokriging`) —
l'AR(1) est un LMC triangulaire, estimé en `s` fits séparés. Un LMC
générique ne vaudrait que pour quelques sorties (`q` petit) de natures
physiques différentes, en hétérotopique, avec assez de points : besoin
plutôt géostatistique, déjà servi par gstat / gstlearn. ⇒ hors périmètre.

### E. Covariances croisées non séparables

Sorties avec régularités ou portées différentes, cohérentes jointement.

- Convolution de processus : Ver Hoef & Barry (1998) ; Higdon (2002) ;
  Boyle & Frean (2005) ; Álvarez & Lawrence (2011), JMLR 12.
- Matérn multivarié : Gneiting, Kleiber & Schlather (2010), JASA 105 ;
  Apanasovich, Genton & Sun (2012), JASA 107.
- Computer experiments : Fricker, Oakley & Urban (2013), *Multivariate
  Gaussian process emulators with nonseparable covariance structures*,
  Technometrics 55(1).
- Revue : Genton & Kleiber (2015), *Cross-covariance functions for
  multivariate geostatistics*, Stat. Sci. 30(2).
- Très expressif, mais contraintes de validité sur les paramètres et coût
  plein `O((nq)³)`. Hors périmètre raisonnable pour libKriging.

### F. Indice de sortie comme variable qualitative

On empile `(x, t)` avec `t ∈ {1..q}` l'indice de la sortie, et on utilise
un noyau produit `k((x,t),(x',t')) = c(t, t') · r(x, x'; θ)`. C'est un ICM
où `c` est paramétré comme une corrélation entre niveaux d'un facteur
qualitatif ; gère nativement l'hétérotopie.

- Qian, Wu & Wu (2008), *Gaussian process models for computer experiments
  with qualitative and quantitative factors*, Technometrics 50(3) ;
  Zhou, Qian & Zhou (2011) (paramétrisation hypersphérique de `c`).
- Roustant, Padonou, Deville, Clément, Perrin, Giorla & Wynn (2020),
  *Group kernels for Gaussian process metamodels with categorical inputs*,
  SIAM/ASA JUQ 8(2).
- Variables latentes : Zhang, Tao, Chen & Apley (2020), *A latent variable
  approach to Gaussian process modeling with qualitative and quantitative
  factors* (LVGP), Technometrics 62(3).
- Revue : Pelamatti et al. (2019), *Efficient global optimization of
  constrained mixed variable problems*, J. Glob. Optim.

> **Point important pour libKriging** : `WarpKriging` sait déjà faire
> `"categorical(L,q)"` (plongement de chaque niveau dans `ℝ^q`, style LVGP —
> `src/lib/include/libKriging/WarpKriging.hpp:21`). Un multi-sorties
> hétérotopique est donc **déjà faisable aujourd'hui** en empilant les données
> avec une colonne « indice de sortie » catégorielle. Limites : (i) variance
> `σ²` commune à toutes les sorties (il faut standardiser chaque sortie au
> préalable) ; (ii) avec un noyau stationnaire positif sur l'espace latent,
> les corrélations inter-sorties restent **positives** (pas de sorties
> anti-corrélées), contrairement à un ICM à `B` libre ; (iii) tendance
> commune. À documenter dans `skills/libkriging/SKILL.md` comme recette.

### G. Réduction de dimension des sorties (`q` grand)

Projeter `Y` (`n × q`) sur `K ≪ q` composantes, krigeage par composante,
reconstruction.

- Higdon, Gattiker, Williams & Rightley (2008), *Computer model calibration
  using high-dimensional output*, JASA 103 (base ACP).
- Marrel, Perot & Mottet (2015), *Development of a surrogate model and
  sensitivity analysis for spatio-temporal numerical simulators*, SERRA
  (ondelettes) ; Perrin, Roustant, Rohmer et al. (2021) — sorties
  fonctionnelles.
- Aucune modification du cœur : ACP + `K` krigeages (A ou B). Variance de
  reconstruction à ajouter (erreur de troncature).

### H. Multi-fidélité / gradients (pour mémoire)

- Kennedy & O'Hagan (2000) ; Le Gratiet (2013) : AR(1) = LMC hiérarchique
  triangulaire → branche `feature/multi-fidelity-cokriging`.
- GEK : `Cov(y, ∂y)` dérivée de `r` → travail local non publié.

### Revues transverses

- Álvarez, Rosasco & Lawrence (2012), *Kernels for vector-valued functions:
  a review*, Found. Trends ML 4(3).
- Liu, Cai & Ong (2018), *Remarks on multi-output Gaussian process
  regression*, Knowledge-Based Systems 144.

## 2. Logiciels existants

| Outil | Langage | Approches multi-sorties | Remarques |
|---|---|---|---|
| **DiceKriging** | R | A (boucle) | aucune |
| **RobustGaSP** (`ppgasp`) | R/C++ | **B** | Gu & Berger ; référence directe pour l'option B, oracle de validation naturel |
| **kergp** | R | F (noyaux qualitatifs, groupes) | Roustant et al. ; oracle pour F |
| **gstat** (`fit.lmc`) | R | C, D (cokrigeage LMC, hétérotopique) | géostat 2-D/3-D, pas orienté computer experiments |
| **gstlearn** (Mines Paris) | C++/R/Python | C, D, cokrigeage complet | successeur de RGeostats |
| **MuFiCokriging** | R | H (AR1) | cf. branche `feature/multi-fidelity-cokriging` |
| **mlegp** | R | G (poids ACP) | ancien |
| **scikit-learn** GPR | Python | B (`y` 2-D, noyau partagé) | pas de covariance croisée |
| **GPy** | Python | C, D (`ICM`, `LCM`, `Coregionalize`) | maintenance faible |
| **GPflow** | Python/TF | A, B, D (`SharedIndependent`, `SeparateIndependent`, `LinearCoregionalization`) | surtout variationnel |
| **GPyTorch** | Python/PyTorch | C Kronecker (`MultitaskKernel`), F Hadamard (`IndexKernel`), D variationnel (`LMCVariationalStrategy`) | le plus complet |
| **BoTorch** | Python | A/B batch (`SingleTaskGP` multi-sorties, `ModelListGP`), C/F (`MultiTaskGP`, `KroneckerMultiTaskGP`) | orienté BO |
| **MOGPTK** | Python | D, E (CONV, CSM, MOSM, SM-LMC) | recherche |
| **OpenTURNS** | C++/Python | A (`TensorizedCovarianceModel`) ; covariance matricielle générale possible | une `KrigingAlgorithm` par sortie en pratique |
| **SMT** | Python | A, F (noyaux mixtes/catégoriels), H (`MFK`) | |
| **emukit** | Python | H (multi-fidélité linéaire/non linéaire) | sur GPy |
| **KernelFunctions.jl** | Julia | A, C, D (`IndependentMOKernel`, `IntrinsicCoregionMOKernel`, `LinearMixingModelKernel`, `LatentFactorMOKernel`) | |
| **UQLab** | MATLAB | A (+ ACP en surcouche) | |

Constat : aucun outil « computer experiments » grand public n'offre à la fois
**B** (rapide, `q` grand) et **C** (covariance croisée pour `simulate`) dans
une API de krigeage classique. RobustGaSP couvre B en R seulement.

## 3. Pistes pour libKriging (recommandation)

### Étape 0 — immédiat, zéro code cœur
- Documenter dans le skill : boucle de fits indépendants (A), recette
  « indice de sortie catégoriel » via `WarpKriging` (F, avec ses limites),
  et ACP + krigeage (G).

### Étape 1 — **modèle B + ICM isotopique** (recommandé)
Généraliser le cœur à `y` matrice `n × q` :

- `m_y : vec → mat`, `m_z`, `m_beta (p × q)`, `m_sigma2 (q)` ; `m_T`, `m_M`,
  `m_circ`, `m_star` **inchangés** (partagés).
- LL / LOO / LMP : somme sur les colonnes (le profilage de `β`, `σ²` est
  colonne par colonne) ; gradient sommé ⇒ l'optimiseur ne change pas.
- `predict` : moyenne `n_new × q`, variance `n_new × q` ; option
  `return_cross` qui renvoie aussi `B̂ = Σ̂` pour la covariance croisée
  `B̂ ⊗ C_post` (formule ICM fermée, cf. §1.C-1).
- `simulate` : tirages conjoints `B̂^{1/2} · Z · C_post^{1/2}` (matrix-normal).
- `update` : identique, `y_u` matriciel.
- Coût de fit ≈ un seul krigeage scalaire pour `q` sorties ⇒ gain réel pour
  les sorties fonctionnelles.
- Oracle : `RobustGaSP::ppgasp` (B) et formules de Conti & O'Hagan (C).

Choix d'API : voir §5 — `y` matriciel **en interne** (`KrigingImpl`), mais
classe publique dédiée `MultiOutputKriging` plutôt qu'une surcharge de
`Kriging::fit`.

### Étape 2 — ICM hétérotopique (optionnel)
Noyau `B[t_i, t_j] · r(x_i, x_j)` sur données empilées, `B = L Lᵀ` de rang
`r ≤ q` paramétré (Cholesky + variances propres). Nouvelle classe de
composition ou extension de `WarpKriging` (le plongement catégoriel existant
est presque cela, il manque variances par niveau et corrélations signées).
Coût `O(N³)` sur `N = Σ n_j`.
Même format de données que `MarkovCoKriging` (`fit(y, X, level)`, D3) :
voir §7.

### Hors périmètre (renvoyer vers GPyTorch / MOGPTK / gstlearn)
LMC à plusieurs portées, convolutions, Matérn multivarié : coût,
identifiabilité et estimation difficiles pour un gain faible en contexte
isotopique de computer experiments.

## 4. Cas particulier : sorties temporelles

`Y` (`n × q`) : pour chaque point `x_i`, une courbe `y(x_i, t_1..t_q)` sur
une grille temporelle commune. Isotopique par construction, `q` souvent
grand (10²–10⁴), souvent `q > n`.

### Options

| Option | Modèle | Coût fit | Atouts | Limites |
|---|---|---|---|---|
| **G — ACP / Karhunen-Loève** | `Y ≈ Ȳ + Σ_k a_k(x) φ_k(t)`, krigeage de chaque `a_k(x)` (θ propres) | `O(K n³)`, `K` ~ 3–20 | gère la **non-stationnarité temporelle** (transitoire puis plateau), `q` quelconque, trajectoires cohérentes par reconstruction, zéro code cœur | erreur de troncature à ajouter à la variance ; composantes supposées indépendantes ; `K ≲ n` |
| **B — PP-GaSP** (Gu & Berger 2016) | un krigeage par pas de temps, `θ` partagés | `O(n³ + q n²)` | aucune hypothèse de forme temporelle, une seule Cholesky | bandes **ponctuelles** seulement (pas de covariance en `t`) ; `B̂` empirique de rang `≤ n − p` si on veut du conjoint |
| **C paramétrique — séparable `R_t(φ) ⊗ R_x(θ)`** (Rougier 2008 ; Conti & O'Hagan 2010) | ICM où `B = σ² R_t(φ)` est un noyau temporel paramétré | `O(n³ + q³)` (eigen de chaque facteur) | **covariance temporelle cohérente** avec 1–2 paramètres en plus ; résout `q > n` (pas de `B̂` singulier) | `R_t` stationnaire ⇒ mal adapté aux régimes transitoires ; `q³` limite à `q` ≲ quelques milliers (au-delà : Toeplitz/FFT si grille régulière, ou ACP) |
| **Émulateurs dynamiques** (Conti, Gosling, Oakley & O'Hagan 2009, Biometrika ; Mohammadi, Challenor et al. 2019, CSDA) | émuler le pas `état(t) → état(t+1)` et itérer | variable | **extrapolation en temps**, horizons longs | propagation d'incertitude délicate ; suppose un code à état explicite |
| ~~Temps comme entrée~~ (empiler `nq` points) | `r_x · r_t` sur `(x,t)` | `O((nq)³)` | — | équivaut à la ligne séparable sans exploiter Kronecker : **à éviter** |

### Point clé : la moyenne ne dépend pas de la structure temporelle

Avec un `β` propre à chaque pas de temps, l'autokrigeabilité (§1.C-2)
s'applique **pour tout `B`, y compris `B = σ² R_t(φ)`** : la moyenne prédite
est celle de PP-GaSP (à l'estimation de `θ̂` près, qui diffère car la
vraisemblance pondère autrement). Modéliser la corrélation temporelle ne sert
donc qu'à :

- `simulate` de **trajectoires** cohérentes ;
- l'incertitude sur des **fonctionnelles de la courbe** (max sur `t`, temps
  passé au-dessus d'un seuil, intégrale, date de pic) — souvent la vraie
  quantité d'intérêt en sûreté ;
- mieux estimer `θ`.

Si l'on ne veut que la moyenne et des bandes ponctuelles, PP-GaSP suffit.

### Vraisemblance séparable (pour l'implémentation)

`Z = Y − F β̂`, `β̂ = (Fᵀ R_x⁻¹ F)⁻¹ Fᵀ R_x⁻¹ Y` (`p × q`, indépendant de
`R_t` : régresseurs identiques ⇒ GLS par colonne), `vec(Z) ~ N(0, σ² R_t ⊗ R_x)` :

    −2 ℓ = q log|R_x| + n log|R_t| + nq log σ² + tr(R_t⁻¹ Zᵀ R_x⁻¹ Z) / σ²
    σ̂²   = tr(R_t⁻¹ Zᵀ R_x⁻¹ Z) / (nq)

Profilé en `(θ, φ)` ; avec `R_x = U_x Λ_x U_xᵀ`, `R_t = U_t Λ_t U_tᵀ` tout se
calcule en `O(n³ + q³ + n q (n + q))`. Prédiction :
`Cov(y(x*, ·)) = σ̂² s²_x(x*) R_t(φ̂)`, avec `s²_x` la variance de krigeage
scalaire usuelle.

### Recommandation pour les sorties temporelles

1. **Par défaut : ACP + krigeage par composante** (G). C'est le plus robuste
   sur des courbes réelles (non stationnaires) et c'est la pratique dominante
   (Higdon 2008, Marrel et al. 2015, UQLab). Faisable tout de suite, en
   surcouche, sans toucher au cœur — à fournir comme recette du skill, voire
   comme petit helper. Choisir `K` par variance expliquée (≥ 99 %) et
   ajouter la variance de troncature.
2. **Étape 1 de la feuille de route (PP-GaSP)** quand les courbes n'ont pas
   de structure de rang faible (chocs, fronts qui se déplacent).
3. **Option séparable `R_t ⊗ R_x`** comme extension naturelle de l'étape 1 si
   les quantités d'intérêt sont des fonctionnelles de trajectoire et
   `q` ≲ 2000 : même `y` matriciel, `+1–2` hyperparamètres, `predict` /
   `simulate` conjoints en `t`.
4. Émulateurs dynamiques : seulement si l'on doit extrapoler en temps —
   hors périmètre.

## 5. API

Brouillon : `draft/MultiOutputKriging.hpp` (vérifié par
`g++ -fsyntax-only`, non branché au build).

### 5.1 Un `y` matriciel suffit-il ?

**En interne, presque.** Pour `"shared"` comme pour `"separable*"`, la
factorisation `T`, `M`, `circ`, `star` de `KrigingImpl` est commune à toutes
les sorties. Ne changent que :

- `m_y`, `m_z` : `n → n × q` ; `m_beta` : `p → p × q` ; `m_sigma2` :
  `double → q` ; dans `KModel` : `ystar`, `Estar`, `betahat` deviennent des
  matrices, `SSEstar` un `rowvec` ; `centerY`/`scaleY` : `double → rowvec` ;
- LL profilée : `−2ℓ = n Σ_j log σ̂_j² + 2q Σ log diag T + nq` ; gradient,
  LOO et LMP se somment de même ⇒ optimiseur inchangé.

`arma::vec` dérive de `arma::mat` : les `solve(T, m_y)` passent tels quels.
Volume : ~230 occurrences de `m_y`/`m_z`/`m_beta`/`m_sigma2` dans
`src/lib/*.cpp` (Kriging 2886 l., KrigingImpl 932 l., WarpKriging 3220 l.)
— mécanique, et `q = 1` redonne exactement le chemin actuel.

**En public, non.** Surcharger `Kriging::fit(const arma::mat& Y, …)` change
tout le reste de l'API :

| Élément | Aujourd'hui | Multi-sorties |
|---|---|---|
| `predict` mean / stdev | `vec` (m) | `mat` (m × q) |
| `predict` cov | `mat` (m × m) | Kronecker `Σ ⊗ C_x`, ou dense (mq × mq) |
| `predict` deriv | `mat` (m × d) | `cube` (m × d × q) |
| `simulate` | `mat` (m × nsim) | `cube` (m × q × nsim) |
| `update`, `update_simulate` | `vec y_u` | `mat Y_u` |
| `y()`, `beta()`, `sigma2()` | `const vec&`, `const double&` | `mat`, `vec` → **casse l'API** |
| `noise` hétérogène | `vec` (n) | par sortie ⇒ casse la factorisation partagée |
| save/load | schéma actuel | nouveau schéma |

Soit deux comportements dans une même classe, des types de retour dépendant
de `q`, une ambiguïté de dispatch dans les 5 bindings (R : vecteur vs matrice
`n × 1` ; Python : 1-D vs 2-D), des options sans sens en multi-sorties, et
aucun endroit naturel pour dire **comment** les sorties sont couplées.

### 5.2 Proposition : deux couches

1. **Cœur** — généraliser `KrigingImpl` à `m_Y` (`n × q`), `q = 1` pour
   `Kriging`/`WarpKriging`/`MLPKriging`. API publique de `Kriging`
   **strictement inchangée** ; ses accesseurs renvoient la colonne 0 (vue
   mémoire `arma::vec(ptr, n, false, true)` pour garder `const vec&`, ou
   retour par valeur).
2. **Public** — classe `MultiOutputKriging(covType, outputModel)` :

| `outputModel` | Modèle | Implémentation |
|---|---|---|
| `"shared"` | PP-GaSP, θ commun, `β_j`, `σ_j²` | `KrigingImpl` généralisé |
| `"separable"` | ICM isotopique, `Σ̂` libre (forme close) | idem + `Σ̂ = Zᵀ R⁻¹ Z / n` |
| `"separable(matern5_2)"` | `Σ = σ² R_t(φ)`, `t` via `set_output_coordinates` | idem + eigen de `R_t`, `φ` dans `gamma` |
| `"pca(K)"` / `"pca(0.99)"` | ACP + `K` `Kriging` indépendants | **composition** (patron `NestedKriging`) — livrable sans toucher au cœur |

Une seule façade pour les trois approches pertinentes en sorties
temporelles : l'utilisateur change de modèle sans changer de code.

### 5.3 Signatures (extrait du brouillon)

```cpp
MultiOutputKriging(const std::string& covType, const std::string& outputModel = "shared");
void set_output_coordinates(const arma::mat& t);              // q × d_t
void fit(const arma::mat& Y, const arma::mat& X, regmodel, normalize, optim, objective, parameters);
std::tuple<arma::mat, arma::mat, arma::mat, arma::cube>        // mean m×q, stdev m×q,
  predict(X_n, return_stdev, return_cov, return_deriv);       //   cov mq×mq, deriv m×d×q
std::tuple<arma::mat, arma::mat> predictCovFactors(X_n);       // (C_x m×m, Σ q×q), Kronecker seuls
arma::cube simulate(nsim, seed, X_n, will_update);             // m × q × nsim
void update(const arma::mat& Y_u, const arma::mat& X_u, bool refit);
arma::mat output_cov() const;                                  // Σ̂ q×q
```

Conventions : `Y` est `n × q` (lignes = observations, comme `X`) ; toute
covariance jointe porte sur `vec(Y_n)` (sortie 1 aux `m` points, puis
sortie 2…), d'où `Cov = Σ ⊗ C_x` pour les modèles de Kronecker.

**Restrictions assumées** (préservent la factorisation partagée) : plan
isotopique ; même `regmodel` pour toutes les sorties ; pas de bruit par
observation/sortie, seulement un rapport de nugget commun ; objectifs `LL`
et `LOO` d'abord ; save/load différé.

Confort optionnel côté Python seulement : `Kriging(y2d, X)` pourrait renvoyer
un `MultiOutputKriging` ; garder l'appel explicite ailleurs.

### 5.4 Séquencement suggéré

1. `"pca"` par composition (aucun impact cœur) + bindings + tests contre une
   ACP/boucle R de référence.
2. Généralisation `KrigingImpl` à `m_Y`, non-régression stricte sur toute la
   suite existante (`q = 1`).
3. `"shared"` (oracle : `RobustGaSP::ppgasp`), puis `"separable"`, puis
   `"separable(<kernel>)"`.

## 6. Questions ouvertes

Tranchées le 2026-10-07 :

- **Q2 — tendance** : même `regmodel` pour toutes les sorties, `β_j` propre
  à chaque sortie.
- **Q3 — nugget/bruit** : refusé dans un premier temps (aucun nugget, aucun
  bruit) ; un éventuel nugget relatif commun viendra plus tard.
- **Q5 — `"separable"` avec `n − p < q`** : refusé, avec un message qui
  oriente vers `"pca"` ou `"separable(<kernel>)"`.
- **Q6 — cov de `predict` en `"shared"`** : matrice dense bloc-diagonale
  `mq × mq` ; `predictCovFactors` réservé aux modèles de Kronecker.

Mise en œuvre de l'étape 2 (§5.4) : plutôt que de changer le type des membres
`m_y`, `m_beta`, `m_sigma2` (ce qui casserait l'API publique de `Kriging`),
seule la couche de factorisation est généralisée — `KModel::ystar/Estar/betahat`
deviennent des `mat`, `populate_Model` accepte un second membre `n × q`,
`compute_ll_grad_theta_vecs` accepte `x` à `q` colonnes, `cross_corr` et
`fit_setup_X_impl` sont extraits. L'état multi-sorties (`Y`, `Z`, `B`, `σ_j²`)
vit dans la classe dérivée qui implémente `"shared"`.

Tranchées le 2026-10-09 :

- **Q1 — cas d'usage cible** : les deux. D'abord les sorties fonctionnelles
  (`q` grand, isotopique : étape 1, faite avec `MultiOutputKriging`), puis
  quelques sorties hétérogènes (`q` petit, hétérotopique : étape 2, ICM sur
  données empilées, §3), dans une PR séparée.
- **Q7 — `LMP` / `LLVecchia` / `LLNystrom`** : hors périmètre pour l'instant ;
  les modèles partagés restent limités à `LL` et `LOO` (`LL` seul pour
  `"separable(<kernel>)"`) ; `"pca"` transmet l'objectif à chaque `Kriging`. Save/load est fait (JSON version 2,
  `"content": "MultiOutputKriging"`).

Reste ouverte (numérotation d'origine) :

4. Format de sortie de `predict` dans les bindings (matrice vs liste par
   sortie), cohérence avec `MarkovCoKriging` (branche `feature/multi-fidelity-cokriging`).
   Choix appliqué aux 4 bindings (2026-10-09) : matrices `m × q`, covariance
   `mq × mq` sur `vec(Y)`, simulations `m × q × nsim`, dérivées `m × d × q`.

## 7. Rapprochement avec `MarkovCoKriging`

Branche `feature/multi-fidelity-cokriging` (PR #350, conception seule) :
AR(1) de Kennedy & O'Hagan / Le Gratiet et co-krigeage collocalisé,
`Z_t = ρ_{t-1} Z_{t-1} + δ_t`, plans emboîtés, `fit(y, X, level)` (D3).

### 7.1 Lien mathématique

À `ρ` constant, même noyau et même `θ` à tous les niveaux, en isotopique,
l'AR(1) est un ICM :

    s = 2 :  Cov = [ σ0²     ρ σ0²        ] ⊗ R(θ)
                   [ ρ σ0²   ρ² σ0² + σ1² ]

- `s = 2` : toute `Σ` 2×2 définie positive s'écrit ainsi (`ρ = Σ01/Σ00`,
  `σ1² = Σ11 − Σ01²/Σ00`, `β1 − ρ β0` pour la tendance de `δ_1`). La
  vraisemblance se factorise en `L(y_0) · L(y_1 | y_0)` dans les deux
  paramétrisations : à `θ` fixé, les deux maximums de vraisemblance coïncident
  avec `"separable"`.
- `s ≥ 3` : la chaîne impose que `y_t` ne dépende que de `y_{t-1}`, ce qui
  donne un ICM contraint (`Σ⁻¹` tridiagonale).
- Ce que le Markov apporte en plus : un `θ_t` par niveau (LMC triangulaire,
  §1.D), et des plans emboîtés hétérotopiques en `O(Σ n_t³)` au lieu de
  `O((Σ n_t)³)`.

### 7.2 À partager

1. **Format des données empilées.** L'étape 2 (§3, ICM hétérotopique)
   reprend exactement la signature `fit(y, X, level)` de D3, avec
   `level ∈ [0, s-1]` : le marshalling dans les bindings est le même.
2. **Conventions de sortie** (Q4) : `predict` en `m × s`, covariance
   `ms × ms` sur `vec(Y)`, `simulate` en `m × s × nsim`, dérivées
   `m × d × s`, `component(i)` pour les sous-`Kriging`, `update` et
   `update_simulate`, JSON de save/load avec un `"content"` propre.
3. **Test croisé** : `s = 2`, isotopique, `θ` fixé et partagé ⇒
   `MarkovCoKriging` doit redonner `MultiOutputKriging("separable")`
   (`ρ`, `σ²`, LL, moyenne et variance prédites). Oracle interne en plus de
   MuFiCokriging.
4. **D2 (plans non emboîtés)** : le « co-krigeage complet » de D2 est
   l'ICM hétérotopique de l'étape 2. Répartition possible : emboîté ⇒
   Markov (factorisé, rapide) ; non emboîté ⇒ étape 2 (exact, `θ` partagé).

### 7.3 Non retenu pour l'instant

`output_model = "markov"` dans `MultiOutputKriging` : son API est `Y n × q`
isotopique, alors que l'intérêt du multi-fidélité vient des plans emboîtés
(peu de points haute fidélité). En isotopique, il n'apporterait que les
`θ_t` par niveau.

À trancher au démarrage de l'étape 2 : une classe hétérotopique unique
`fit(y, X, level)` avec deux modèles, `"icm"` (vraisemblance jointe) et
`"markov"` (factorisée), qui réunirait l'étape 2 et `MarkovCoKriging`. Les
deux sont à concevoir ensemble.
