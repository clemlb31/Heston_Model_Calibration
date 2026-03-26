# Calibration du Modele de Heston sur Options SPY

## Presentation du Projet

---

## Table des matieres

1. [Contexte et motivation](#1--contexte-et-motivation)
2. [Le modele de Black-Scholes et ses limites](#2--le-modele-de-black-scholes-et-ses-limites)
3. [Le modele de Heston](#3--le-modele-de-heston)
4. [Pricing par inversion de Fourier](#4--pricing-par-inversion-de-fourier)
5. [Donnees de marche et preprocessing](#5--donnees-de-marche-et-preprocessing)
6. [Methodes de calibration](#6--methodes-de-calibration)
7. [Architecture du code](#7--architecture-du-code)
8. [Resultats experimentaux](#8--resultats-experimentaux)
9. [Discussion et limites](#9--discussion-et-limites)
10. [Conclusion](#10--conclusion)

---

## 1 — Contexte et motivation

### Le probleme

Sur les marches d'options, on observe un phenomene bien connu : la **volatilite implicite** n'est pas constante.
Elle varie en fonction du strike (le **smile**) et de la maturite (la **structure par terme**).
Ce phenomene contredit directement l'hypothese fondamentale de Black-Scholes : une volatilite constante.

### L'objectif

Calibrer un modele de **volatilite stochastique** (Heston, 1993) sur des donnees reelles d'options SPY (S&P 500) afin de :

- Reproduire fidelement la **nappe de volatilite implicite** observee sur le marche
- Comparer **3 methodes d'optimisation** pour la calibration
- Quantifier l'amelioration par rapport a Black-Scholes
- Fournir un framework reutilisable pour la calibration de modeles de pricing

### Pourquoi le SPY ?

Le SPY (ETF S&P 500) est le sous-jacent le plus liquide au monde pour les options :
- Milliers de contrats disponibles chaque jour
- Nombreuses maturites (de quelques jours a plus d'un an)
- Large eventail de strikes
- Spreads bid-ask serres

Cela en fait un candidat ideal pour tester la calibration d'un modele.

---

## 2 — Le modele de Black-Scholes et ses limites

### Rappel du modele

Black-Scholes (1973) suppose que le sous-jacent suit un mouvement brownien geometrique :

```
dS(t) = (r - q) S(t) dt + sigma * S(t) dW(t)
```

ou `sigma` est **constante**. Le prix d'un call europeen est alors :

```
C = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)

d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)
```

### Implementation dans le projet

La classe `BlackScholes` (`heston/black_scholes.py`) implemente :

| Methode | Role |
|---|---|
| `price(S, K, T, r, q, sigma)` | Prix d'un call europeen (formule fermee) |
| `vega(S, K, T, r, q, sigma)` | Sensibilite du prix a la volatilite (dC/dsigma) |
| `implied_vol(price, S, K, T, r, q)` | Inversion numerique par Newton-Raphson |

**Pourquoi Newton-Raphson pour l'IV ?** La relation prix → volatilite n'a pas de forme fermee inverse. On utilise donc une methode iterative : a chaque iteration, on corrige l'estimation de `sigma` par `sigma -= (BS_price - market_price) / vega`. La convergence est quadratique grace a la monotonie du prix en fonction de sigma.

### Limites observees

Sur les donnees SPY reelles :

- **RMSE IV Black-Scholes : ~4.2%** (avec sigma = IV ATM moyenne a 13%)
- BS ne peut pas capturer le skew (IV plus elevee pour les puts OTM)
- BS ne peut pas capturer la structure par terme (IV varie avec la maturite)
- L'erreur est particulierement forte sur les ailes du smile (Deep OTM)

---

## 3 — Le modele de Heston

### Dynamique du modele

Heston (1993) introduit une **variance stochastique** :

```
dS(t) = (r - q) S(t) dt + sqrt(v(t)) S(t) dW1(t)        [prix]
dv(t) = kappa * (theta - v(t)) dt + sigma * sqrt(v(t)) dW2(t)  [variance]

<dW1, dW2> = rho * dt                                      [correlation]
```

La variance `v(t)` suit un processus de **Cox-Ingersoll-Ross** (CIR), avec retour a la moyenne.

### Les 5 parametres

| Parametre | Symbole | Interpretation | Bornes utilisees |
|---|---|---|---|
| Variance initiale | `v0` | Volatilite spot au carre (sigma_spot^2) | [0.0001, 1.0] |
| Vitesse de retour | `kappa` | Rapidite du retour de v(t) vers theta | [0.001, 20.0] |
| Variance long terme | `theta` | Niveau d'equilibre de la variance | [0.0001, 1.0] |
| Vol of vol | `sigma` | Volatilite de la variance elle-meme | [0.01, 3.0] |
| Correlation | `rho` | Correlation entre le spot et la variance | [-0.99, 0.50] |

### Role de chaque parametre sur le smile

- **`v0`** : Fixe le **niveau general** de la nappe de volatilite. Un v0 eleve deplace tout le smile vers le haut.
- **`kappa`** : Controle la **structure par terme**. Un kappa eleve signifie que les maturites longues convergent vite vers theta, aplatissant le smile aux longues maturites.
- **`theta`** : Determine le **niveau asymptotique** de la volatilite pour les longues maturites.
- **`sigma` (vol of vol)** : Controle la **courbure** du smile. Plus sigma est grand, plus le smile est prononce (queues epaisses dans la distribution du sous-jacent).
- **`rho`** : Controle le **skew** (asymetrie). Un rho negatif (typique pour les actions : -0.7 a -0.3) signifie que quand le prix baisse, la volatilite augmente — c'est l'**effet de levier**. Cela cree un skew negatif : les puts OTM ont une IV plus elevee.

### La condition de Feller

Pour que la variance reste **strictement positive** a tout instant :

```
2 * kappa * theta > sigma^2
```

**Interpretation** : La force de rappel vers la moyenne (`kappa * theta`) doit etre suffisamment grande pour compenser la diffusion de la variance (`sigma^2`).

**En pratique sur donnees reelles** : La condition de Feller est **frequemment violee**. Sur les donnees SPY 2024 :
- `2 * kappa * theta = 0.4153`
- `sigma^2 = 0.9705`
- **Ratio = 0.43** (il faudrait > 1)

Cela signifie que le marche price un vol-of-vol eleve (queues epaisses), ce qui est economiquement justifie : les investisseurs paient une prime pour le risque de crash. Le modele reste numeriquement stable grace a l'approche par Fourier (on ne simule pas les trajectoires).

### Implementation

La classe `HestonParams` (`heston/params.py`) est un `dataclass` avec :
- Conversion vers/depuis un vecteur numpy (`to_array()` / `from_array()`) pour l'optimisation
- Verification de la condition de Feller (`feller_satisfied`)
- Affichage detaille avec volatilites en pourcentage

---

## 4 — Pricing par inversion de Fourier

### Pourquoi Fourier ?

Contrairement a Black-Scholes, le modele de Heston n'a **pas de formule fermee** pour le prix des options. Cependant, sa **fonction caracteristique** (la transformee de Fourier de la densite risque-neutre) a une forme analytique. On peut donc obtenir les prix par **inversion numerique**.

### Formule de pricing

Le prix d'un call europeen dans le modele de Heston s'ecrit :

```
C = S0 * e^(-qT) * P1  -  K * e^(-rT) * P2
```

ou P1 et P2 sont des probabilites obtenues par integration :

```
Pj = 1/2 + (1/pi) * integral_0^infini  Re[ e^(-iu*ln(K)) * fj(u) / (iu) ] du
```

avec `f1` (mesure spot, j=1) et `f2` (mesure forward, j=2) les fonctions caracteristiques de Heston.

### Fonction caracteristique (formulation stable)

On utilise la formulation d'**Albrecher et al. (2007)** qui evite les discontinuites du logarithme complexe :

```
fj(u) = exp(C(u,T) + D(u,T) * v0 + iu * ln(S0))
```

avec :
```
d = sqrt( (rho*sigma*iu - b)^2 - sigma^2 * (2*uj*iu - u^2) )
g = (b - rho*sigma*iu - d) / (b - rho*sigma*iu + d)

C = (r-q)*iu*T + (a/sigma^2) * [ (b - rho*sigma*iu - d)*T - 2*ln((1 - g*e^(-dT)) / (1 - g)) ]
D = ((b - rho*sigma*iu - d) / sigma^2) * (1 - e^(-dT)) / (1 - g*e^(-dT))
```

ou :
- Pour j=1 : `b = kappa - rho*sigma`, `uj = 0.5`
- Pour j=2 : `b = kappa`, `uj = -0.5`
- `a = kappa * theta`

**Pourquoi cette formulation ?** La formulation originale de Heston (1993) souffre de discontinuites dans le logarithme complexe (`ln(1-g*exp(-dT))`), ce qui peut provoquer des sauts dans l'integrale. La formulation d'Albrecher et al. utilise une representation equivalente mais **numeriquement stable** en evitant ces discontinuites.

### Integration numerique

L'integration est faite par **quadrature trapezoidale** sur une grille fixe :

| Parametre | Valeur | Justification |
|---|---|---|
| `n_points` | 500 | Precision suffisante pour < 0.01% d'erreur |
| `u_max` | 100 | Les integrandes decroissent exponentiellement au-dela |
| `u_min` | 1e-8 | Evite la singularite en u=0 |

**Pourquoi la quadrature trapezoidale et non Gauss-Legendre ?** Pour des fonctions lisses et regulieres sur un intervalle large, la quadrature trapezoidale avec un nombre suffisant de points est tres efficace et plus simple a mettre en oeuvre. La grille fixe permet aussi de **reutiliser les valeurs de la fonction caracteristique** pour differents strikes a la meme maturite.

### Optimisation de la vectorisation

Dans `HestonPricer.call_prices_vectorized()` :

1. On regroupe les options par **maturite unique**
2. Pour chaque maturite, on calcule `f1` et `f2` **une seule fois** sur toute la grille `u`
3. On boucle ensuite sur les strikes (operation legere : juste une multiplication par `exp(-iu*ln(K))`)

Cela reduit considerablement le nombre d'evaluations de la fonction caracteristique (couteuse), puisque le nombre de maturites uniques est typiquement 10-20x inferieur au nombre total d'options.

---

## 5 — Donnees de marche et preprocessing

### Source des donnees

Les donnees proviennent du dataset **Kaggle SPY Implied Volatility** couvrant 6 annees :

| Fichier | Annee | Contexte de marche | ~Jours | ~Options/jour |
|---|---|---|---|---|
| `spy_options_data_19.json` | 2019 | Marche haussier, vol basse | 250 | 7000-9500 |
| `spy_options_data_20.json` | 2020 | COVID-19, vol extreme | 250 | 7000-9500 |
| `spy_options_data_21.json` | 2021 | Recovery post-COVID | 250 | 7000-9500 |
| `spy_options_data_22.json` | 2022 | Bear market, hausse taux | 250 | 7000-9500 |
| `spy_options_data_23.json` | 2023 | Rally IA | 250 | 7000-9500 |
| `spy_options_data_24.json` | 2024 | Normalisation | 252 | 7000-9500 |

### Inference du prix spot (S0)

Le prix spot n'est pas directement dans les donnees. On le deduit des **calls deep in-the-money** a courte maturite :

```python
# Appels avec delta >= 0.99 et maturite <= 7 jours
S0 = median(strike + mark_price)
```

**Pourquoi ?** Un call tres profondement dans la monnaie (delta ~ 1) se comporte presque comme le sous-jacent : son prix est approximativement `S - K`, donc `S ~ K + prix_call`. La mediane filtre les outliers.

**Fallback** : Si pas assez de deep ITM calls, on utilise les 5 calls avec le plus grand delta, puis en dernier recours, la mediane des strikes ATM (delta ~ 0.5).

### Pipeline de filtrage

Le filtrage est une etape cruciale. On part de ~4000 contrats bruts par jour pour n'en garder que ~700 de qualite. Chaque filtre a une raison precise :

| Etape | Filtre | Justification |
|---|---|---|
| 1 | Calls uniquement | Les puts sont redondants (parite put-call). Utiliser les calls evite les doubles comptages et les calls sont generalement plus liquides cote equities. |
| 2 | Moneyness : 0.85 <= K/S0 <= 1.15 | Les options tres OTM ou ITM ont des prix tres faibles (bruit) ou sont illiquides. La zone [-15%, +15%] couvre l'essentiel de l'information du smile. |
| 3 | Maturite : 14 a 550 jours | < 14j : trop sensibles aux microstructures, effets de gamma. > 550j : illiquides, peu de volume. |
| 4 | Liquidite : volume > 0, OI > 10 | Options sans volume = pas de vrai prix de marche. Open interest > 10 filtre les positions residuelles. |
| 5 | Prix : bid > 0.01, ask > bid, mid > 0.10 | Elimine les options a prix zero ou avec des spreads inverses (erreurs de donnees). |
| 6 | IV : 3% a 80% | IV < 3% = probablement une erreur. IV > 80% = regime extreme (hors scope du modele). |
| 7 | Non-arbitrage : intrinsic < prix < upper bound | Le prix doit etre superieur a la valeur intrinseque actualisee et inferieur a S0*exp(-qT). Sinon, opportunite d'arbitrage = erreur de donnees. |
| 8 | Min options par maturite : >= 15 | Une maturite avec trop peu de strikes ne permet pas de reconstruire le smile. On supprime ces maturites pour ne garder que des coupes fiables. |

### Resultat du filtrage (exemple SPY 2024-01-04)

```
Brut   : 3889 contrats
Filtre : 746 options retenues, 17 maturites
S0     : 467.41
```

Les options retenues couvrent des strikes de 400 a 535 et des maturites de 15 a 379 jours.

---

## 6 — Methodes de calibration

### Fonction objectif

On cherche les 5 parametres `(v0, kappa, theta, sigma, rho)` qui minimisent l'ecart entre les prix du modele et les prix du marche :

```
min  sum_i  [ (C_i^Heston - C_i^market) / vega_i ]^2
```

**Pourquoi la ponderation par le vega ?**

Diviser par le vega convertit une erreur en prix en une erreur en volatilite implicite. Sans cette ponderation :
- Les options chere (longues maturites, ATM) domineraient la fonction objectif
- Les options bon marche (courtes maturites, OTM) seraient ignorees
- Le fit serait biaise vers les longues maturites

Avec la ponderation vega, chaque option contribue **proportionnellement a son contenu informationnel sur le smile**. Un ecart de 0.01$ sur une option a vega=10 (ATM longue maturite) est equivalent a un ecart de 0.001$ sur une option a vega=1 (OTM courte maturite) — les deux correspondent a ~0.1% d'erreur en IV.

### Calibration multi-jours

Une originalite de ce projet : on peut calibrer sur **plusieurs jours simultanement**.

```python
calibrator = HestonCalibrator(markets)  # markets = liste de MarketData
```

- Chaque jour conserve son propre `S0` et son propre `HestonPricer`
- Les 5 parametres Heston sont **partages** entre tous les jours
- Les residus sont **concatenes** : on optimise sur l'ensemble des options de tous les jours

**Avantages** :
- Plus de donnees = estimation plus robuste (moins de surapprentissage)
- La ponderation par vega normalise naturellement entre des jours avec des S0 differents
- On capture une vision "moyenne" des parametres sur la periode

### Methode 1 : Levenberg-Marquardt (LM)

**Type** : Optimisation locale (moindres carres non-lineaires)

**Algorithme** : Trust Region Reflective (`scipy.optimize.least_squares` avec `method="trf"`)

**Principe** : Combine la descente de gradient (loin de l'optimum) et la methode de Gauss-Newton (pres de l'optimum). A chaque iteration, on resout un sous-probleme quadratique dans une region de confiance.

**Configuration** :
```python
least_squares(residuals, x0,
    bounds=(BOUNDS_LOWER, BOUNDS_UPPER),
    method="trf",
    ftol=1e-10,   # tolerance sur la fonction
    xtol=1e-10,   # tolerance sur les parametres
    max_nfev=300,  # max evaluations
)
```

**Point de depart** (si non fourni) :
```python
atm_iv = mean(toutes les IV du marche)
x0 = [atm_iv^2, 1.5, atm_iv^2, 0.3, -0.7]
```
On initialise `v0` et `theta` a la variance ATM moyenne, `kappa` a une valeur moderee, `sigma` conservateur, et `rho` typique des actions.

**Avantages** : Tres rapide (~8 secondes sur 3600 options, ~97 evaluations)

**Inconvenients** : Sensible au point de depart. Peut converger vers un minimum local.

### Methode 2 : Evolution Differentielle (DE)

**Type** : Optimisation globale (meta-heuristique stochastique)

**Principe** : Maintient une **population** de solutions candidates. A chaque generation, de nouveaux candidats sont crees par **mutation** (combinaison de 3 individus) et **croisement** (melange avec le parent). Le meilleur survit.

**Configuration** :
```python
differential_evolution(objective, bounds,
    maxiter=40,         # generations max
    popsize=12,         # taille de la population
    strategy="best1bin", # strategie de mutation
    mutation=(0.5, 1.5), # facteur de mutation dithered
    recombination=0.8,   # probabilite de croisement
    tol=1e-8,
    seed=42,
)
```

**Pourquoi "best1bin" ?**
- `best` : la mutation utilise le **meilleur** individu comme base (convergence plus rapide)
- `1` : un seul vecteur de difference (simplicite)
- `bin` : croisement binomial (chaque composante est choisie independamment)

**Pourquoi mutation dithered [0.5, 1.5] ?** Le facteur de mutation est tire aleatoirement dans [0.5, 1.5] a chaque generation. Cela evite la stagnation en variant l'amplitude d'exploration.

**Avantages** : Exploration globale, pas de dependance au point de depart

**Inconvenients** : Lent (~242 secondes, ~3249 evaluations). Precision finale parfois limitee.

### Methode 3 : Hybride (DE + LM) — Recommandee

**Principe** : Combiner le meilleur des deux mondes.

```
Phase 1 : DE rapide (25 iterations) → exploration globale
Phase 2 : LM depuis le resultat DE → raffinement local
```

**Pourquoi cette approche ?**
- DE explore largement l'espace des parametres en peu d'iterations (pas besoin de convergence complete)
- LM affine la solution avec une convergence quadratique locale
- On obtient la **robustesse** de DE avec la **precision** de LM

**Performance** : ~141 secondes, 1884 evaluations. Meilleure precision que DE seul, meme robustesse.

### Comparaison des methodes

| Critere | LM | DE | Hybride |
|---|---|---|---|
| Temps (5 jours, 3602 opts) | ~8s | ~242s | ~141s |
| Evaluations | ~97 | ~3249 | ~1884 |
| RMSE IV | 2.74% | 2.75% | 2.74% |
| Robustesse | Sensible au x0 | Robuste | Robuste |
| Precision | Haute (si bon x0) | Moyenne | Haute |
| Recommande pour | Prototypage rapide | Exploration | Production |

---

## 7 — Architecture du code

### Structure du projet

```
Heston_Model_Calibration/
|-- heston/                    # Package principal
|   |-- __init__.py            # Exports publics
|   |-- params.py              # HestonParams (5 parametres + Feller)
|   |-- black_scholes.py       # Prix BS, vega, IV Newton-Raphson
|   |-- pricer.py              # HestonPricer (Fourier)
|   |-- market_data.py         # MarketData (chargement, filtrage)
|   |-- calibrator.py          # HestonCalibrator (LM, DE, hybride)
|   |-- results.py             # CalibrationResult, compare_results
|-- data/                      # Donnees SPY (JSON, 2019-2024)
|-- calibration.ipynb          # Notebook d'analyse complet
|-- test.ipynb                 # Notebook de validation
|-- main.py                    # Point d'entree CLI
|-- requirements.txt           # Dependances
```

### Flux de calcul

```
                    MarketData.from_json()
                         |
           Chargement JSON + filtrage multi-etapes
                         |
                    [MarketData]  (liste de jours)
                         |
                HestonCalibrator(markets)
                    /    |    \
                   /     |     \
                LM      DE    Hybride
                  \      |      /
                   \     |     /
             _all_residuals(x)      <-- appele par l'optimiseur
                     |
        HestonPricer.call_prices_vectorized()
                     |
          _characteristic_function()    <-- coeur du calcul
                     |
              CalibrationResult
```

### Choix de conception

**Pourquoi un dataclass pour HestonParams ?**
Immutabilite logique + conversion facile vers numpy (interface avec scipy). Le `__repr__` affiche les volatilites en pourcentage pour faciliter l'interpretation.

**Pourquoi separer HestonPricer de HestonCalibrator ?**
Le pricer est lie a un S0 specifique (un jour de trading). Le calibrateur gere plusieurs jours. Cette separation permet de reutiliser le pricer independamment (ex: pricing apres calibration).

**Pourquoi les poids 1/vega dans MarketData.__post_init__ ?**
Les poids sont calcules une seule fois au chargement (ils ne dependent pas des parametres Heston). Les stocker dans MarketData evite de les recalculer a chaque evaluation de la fonction objectif.

**Pourquoi les bornes sur les parametres ?**
Sans bornes, l'optimiseur peut explorer des zones non-physiques (variance negative, correlation > 1). Les bornes `BOUNDS_LOWER` et `BOUNDS_UPPER` dans le calibrateur garantissent des parametres economiquement sensibles.

### Dependances

| Package | Usage |
|---|---|
| `numpy` | Calcul numerique, tableaux, operations vectorisees |
| `scipy` | Optimisation (`least_squares`, `differential_evolution`), statistiques (`norm`) |
| `pandas` | Manipulation des donnees brutes, filtrage, statistiques par groupe |
| `matplotlib` | Visualisation (smiles, erreurs, comparaisons) |

---

## 8 — Resultats experimentaux

### Validation synthetique

On genere une nappe de volatilite avec des parametres connus et on verifie que la calibration les retrouve :

| Parametre | Vrai | LM | DE | Hybride |
|---|---|---|---|---|
| v0 | 0.0400 | 0.0398 | 0.0397 | 0.0398 |
| kappa | 1.5000 | 1.3532 | 1.2652 | 1.3532 |
| theta | 0.0500 | 0.0511 | 0.0520 | 0.0511 |
| sigma | 0.4000 | 0.4107 | 0.4076 | 0.4107 |
| rho | -0.7000 | -0.6928 | -0.6923 | -0.6928 |
| RMSE IV | — | 0.27% | 0.27% | 0.27% |

**Observations** :
- Les 3 methodes retrouvent les parametres avec une excellente precision
- `kappa` est le parametre le plus difficile a identifier (correle avec `theta`)
- Le bruit (0.3% sur les IV) est correctement absorbe
- Le RMSE IV residuel (0.27%) correspond au niveau de bruit injecte

### Calibration sur donnees reelles (SPY Janvier 2024)

**Donnees** : 5 jours (2-8 janvier 2024), 3602 options, 17-18 maturites par jour

**Parametres calibres (methode hybride)** :

| Parametre | Valeur | Interpretation |
|---|---|---|
| v0 = 0.0180 | vol spot = 13.41% | Volatilite basse debut 2024 |
| kappa = 10.00 | Retour rapide | Fort mean-reversion (atteint la borne sup) |
| theta = 0.0208 | vol LT = 14.41% | Vol long-terme legerement superieure |
| sigma = 0.985 | Vol of vol elevee | Queues epaisses pricees par le marche |
| rho = -0.663 | Correlation negative | Effet de levier classique |

### Heston vs Black-Scholes

| Metrique | Black-Scholes | Heston |
|---|---|---|
| RMSE IV | 4.23% | 2.85% |
| Amelioration | — | **1.5x plus precis** |
| Capture du skew | Non | Oui |
| Structure par terme | Non | Oui |

**Erreur par zone de moneyness** :

L'avantage de Heston est particulierement visible sur les **ailes du smile** :
- **Deep OTM puts** (K/S < 0.90) : zone ou le skew est le plus prononce, BS echoue completement
- **ATM** (K/S ~ 1.0) : les deux modeles sont proches (BS est calibre sur cette zone)
- **OTM calls** (K/S > 1.05) : Heston capture mieux la legere convexite du smile

### Condition de Feller sur donnees reelles

Les 3 methodes donnent une condition de Feller **violee** :

```
2 * kappa * theta = 0.4153
sigma^2           = 0.9705
Ratio             = 0.43  (il faudrait > 1)
```

C'est un resultat **attendu et documente** dans la litterature quantitative :
- Les marches pricent un vol-of-vol eleve (risque de crash)
- La condition de Feller est une propriete mathematique, pas une contrainte economique
- Le pricing par Fourier reste stable meme sans Feller (pas de simulation de trajectoires)

---

## 9 — Discussion et limites

### Points forts du projet

- **Framework complet** : du chargement des donnees brutes au resultat final, tout est integre
- **Stabilite numerique** : formulation Albrecher et al. pour l'inversion de Fourier
- **Filtrage rigoureux** : 8 etapes de filtrage assurent la qualite des donnees
- **Multi-jours** : calibration robuste en aggregeant plusieurs jours
- **Comparaison systematique** : 3 methodes testees dans les memes conditions
- **Validation synthetique** : verification que le code fonctionne avant d'attaquer les donnees reelles

### Limites

1. **Modele a 1 facteur** : Heston ne capture qu'un seul facteur de volatilite. Des modeles a 2 facteurs (Double Heston) ou avec sauts (Bates) peuvent mieux capturer les courtes maturites.

2. **Parametres statiques** : Les 5 parametres sont calibres comme des constantes. En realite, ils evoluent dans le temps. Une calibration glissante (jour par jour) donnerait une vue dynamique.

3. **Kappa a la borne superieure** : `kappa = 10` atteint la borne superieure imposee, ce qui suggere que le modele "force" un retour a la moyenne rapide. Cela peut indiquer que le modele a un seul facteur est insuffisant pour capturer simultanement le smile court-terme et long-terme.

4. **Pas d'options americaines** : Le pricing par Fourier est limite aux options europeennes. Les options SPY sont de style americain, mais les calls sur indice sont rarement exerces tot (la prime de temps est presque toujours positive), ce qui limite l'impact.

5. **Taux et dividendes constants** : r = 5% et q = 1.3% sont fixes. En realite, la courbe des taux et les dividendes futurs varient. Une extension utiliserait les taux forward et les dividendes discrets.

### Pistes d'amelioration

- **Modele de Bates** (Heston + sauts) pour mieux capturer les courtes maturites
- **Calibration par FFT** (Carr & Madan, 1999) pour accelerer le pricing de centaines d'options simultanement
- **Calibration glissante** pour suivre l'evolution des parametres au fil du temps
- **Surface de volatilite locale** (Dupire) comme benchmark supplementaire

---

## 10 — Conclusion

Ce projet demontre qu'une implementation soigneuse du modele de Heston, combinee a des donnees reelles bien filtrees et a des methodes d'optimisation adaptees, permet de reproduire fidelement la nappe de volatilite implicite du SPY.

**Resultats cles** :
- Le modele de Heston est **1.5x plus precis** que Black-Scholes sur les donnees SPY 2024
- La methode **hybride (DE + LM)** offre le meilleur compromis robustesse/precision
- La **condition de Feller** est systematiquement violee sur donnees reelles, ce qui est coherent avec la litterature
- Le **filtrage des donnees** est une etape aussi importante que le modele lui-meme

Le code est structure de maniere modulaire, permettant d'etendre facilement le framework a d'autres modeles de volatilite stochastique ou a d'autres sous-jacents.

---

## References

- **Heston, S.L.** (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.* The Review of Financial Studies, 6(2), 327-343.
- **Albrecher, H., Mayer, P., Schoutens, W., Tistaert, J.** (2007). *The Little Heston Trap.* Wilmott Magazine, January 2007.
- **Black, F. & Scholes, M.** (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy, 81(3), 637-654.
- **Carr, P. & Madan, D.** (1999). *Option valuation using the fast Fourier transform.* Journal of Computational Finance, 2(4), 61-73.
- **Cox, J.C., Ingersoll, J.E., Ross, S.A.** (1985). *A Theory of the Term Structure of Interest Rates.* Econometrica, 53(2), 385-407.
