# Calibration du Modele de Heston

Calibration du modele de Heston sur des donnees reelles d'options SPY (S&P 500) par inversion de Fourier.

Trois methodes d'optimisation sont comparees (Levenberg-Marquardt, Evolution Differentielle, Hybride) sur plusieurs jours simultanement, avec comparaison face a Black-Scholes.

## Donnees

Options SPY (2019-2024) issues de Kaggle :
https://www.kaggle.com/datasets/shankerabhigyan/s-and-p500-options-spy-implied-volatility-2019-24/data

6 fichiers JSON, ~250 jours de trading chacun, ~7000-9500 contrats/jour.

Les fichiers doivent etre places dans `data/`.

## Structure du projet

```
heston/
  __init__.py          # Exports publics
  params.py            # HestonParams (5 parametres + condition de Feller)
  black_scholes.py     # Prix, vega, volatilite implicite (Newton-Raphson)
  pricer.py            # HestonPricer (inversion de Fourier)
  market_data.py       # MarketData (chargement JSON, filtrage, donnees synthetiques)
  calibrator.py        # HestonCalibrator (LM, DE, Hybride)
  results.py           # CalibrationResult + compare_results
main.py                # Point d'entree CLI
calibration.ipynb      # Notebook d'analyse complet
requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

**CLI :**
```bash
python main.py
```

**Notebook :**
Ouvrir `calibration.ipynb` pour l'analyse complete avec visualisations.

**En code :**
```python
from heston import MarketData, HestonCalibrator, compare_results

# Charger 5 jours de donnees
markets = MarketData.from_json("data/spy_options_data_24.json", day_indices=list(range(5)))

# Calibrer avec les 3 methodes
calibrator = HestonCalibrator(markets)
results = [
    calibrator.calibrate(method="lm"),
    calibrator.calibrate(method="de"),
    calibrator.calibrate(method="hybrid"),
]
compare_results(results)
```

---

## Theorie

### Le modele de Heston

Le modele de Black-Scholes suppose une volatilite constante, ce qui ne correspond pas a la realite : les marches d'options exhibent un **smile de volatilite** (la volatilite implicite varie avec le strike) et une **structure par terme** (elle varie aussi avec la maturite).

Le modele de Heston (1993) introduit une **volatilite stochastique** : la variance du sous-jacent suit elle-meme un processus aleatoire.

#### Dynamique

Sous la mesure risque-neutre :

$$dS(t) = (r - q)\, S(t)\, dt + \sqrt{v(t)}\, S(t)\, dW_1(t)$$

$$dv(t) = \kappa\, (\theta - v(t))\, dt + \sigma\, \sqrt{v(t)}\, dW_2(t)$$

avec $\langle dW_1, dW_2 \rangle = \rho\, dt$.

#### Les 5 parametres

| Parametre | Symbole | Role |
|-----------|---------|------|
| `v0` | $v_0$ | Variance instantanee ($= \sigma_{\text{spot}}^2$) |
| `kappa` | $\kappa$ | Vitesse de retour a la moyenne de la variance |
| `theta` | $\theta$ | Variance long terme ($= \sigma_{\text{LT}}^2$) |
| `sigma` | $\sigma$ | Volatilite de la variance (vol of vol) |
| `rho` | $\rho$ | Correlation entre le spot et la variance |

**Interpretation :**
- $\kappa$ eleve : la variance revient vite vers $\theta$, le smile est plus stable
- $\sigma$ eleve : forte variabilite de la variance, smile plus prononce
- $\rho < 0$ (typique en equity) : quand le spot baisse, la vol monte — c'est le **skew** (effet leverage)
- $v_0$ vs $\theta$ : si $v_0 > \theta$, la vol spot est au-dessus de l'equilibre long terme (et inversement)

#### Condition de Feller

$$2\kappa\theta > \sigma^2$$

Si cette condition est satisfaite, le processus de variance reste **strictement positif** (il ne touche jamais zero). En pratique, les calibrations sur donnees reelles violent souvent cette condition car le marche price un vol of vol eleve.

### Pricing par inversion de Fourier

Le modele de Heston admet une **fonction caracteristique en forme fermee**, ce qui permet de calculer les prix d'options sans simulation Monte Carlo.

Le prix d'un call europeen s'ecrit :

$$C = S_0\, e^{-qT}\, P_1 - K\, e^{-rT}\, P_2$$

ou $P_1$ et $P_2$ sont obtenues par integration numerique de la fonction caracteristique :

$$P_j = \frac{1}{2} + \frac{1}{\pi} \int_0^{\infty} \text{Re}\left[\frac{e^{-iu \ln K}\, f_j(u)}{iu}\right] du$$

L'implementation utilise la **formulation stable d'Albrecher et al. (2007)** pour eviter les discontinuites du logarithme complexe, et une **quadrature trapezoidale** sur 500 points (rapide et precise).

### Filtrage des donnees

Les donnees brutes contiennent beaucoup de bruit. Le filtrage applique :

1. **Calls uniquement** (puts redondants par parite call-put)
2. **Moneyness** : K/S0 entre 0.85 et 1.15 (exclut les options tres OTM/ITM, peu liquides)
3. **Maturite** : entre 14 et 550 jours
4. **Liquidite** : volume > 0, open interest > 10, bid > 0, ask > bid
5. **IV raisonnable** : entre 3% et 80%
6. **Non-arbitrage** : prix > valeur intrinseque, prix < borne superieure
7. **Smiles complets** : au moins N strikes par maturite (configurable, defaut = 8) — les maturites avec trop peu de points sont eliminees

Le **spot S0** est deduit des deep ITM calls a maturite courte (delta >= 0.99).

---

## Methodes de calibration

La calibration consiste a trouver les 5 parametres $(v_0, \kappa, \theta, \sigma, \rho)$ qui minimisent l'ecart entre les prix du modele et les prix de marche.

**Fonction objectif :** somme des residus ponderes par le vega inverse :

$$r_i = \frac{C_i^{\text{Heston}} - C_i^{\text{marche}}}{\mathcal{V}_i}$$

$$\min_{v_0, \kappa, \theta, \sigma, \rho} \sum_{i=1}^{N} r_i^2$$

La ponderation par $1/\mathcal{V}_i$ (inverse du vega) transforme les erreurs de prix en erreurs de volatilite implicite, ce qui donne un poids egal a chaque option independamment de son niveau de prix ou de sa maturite.

**Bornes des parametres :**

| Parametre | Min | Max |
|-----------|-----|-----|
| $v_0$ | 0.0001 | 1.0 |
| $\kappa$ | 0.001 | 10.0 |
| $\theta$ | 0.0001 | 1.0 |
| $\sigma$ | 0.01 | 3.0 |
| $\rho$ | -0.99 | 0.50 |

### Levenberg-Marquardt (`method="lm"`)

Optimisation **locale** par moindres carres non-lineaires (`scipy.optimize.least_squares`, methode TRF).

- **Avantage** : tres rapide (quelques secondes), exploite la structure des residus
- **Inconvenient** : sensible au point de depart, peut converger vers un minimum local
- **Point de depart par defaut** : $v_0 = \bar{\sigma}_{\text{ATM}}^2$, $\kappa = 1.5$, $\theta = \bar{\sigma}_{\text{ATM}}^2$, $\sigma = 0.3$, $\rho = -0.7$

### Evolution Differentielle (`method="de"`)

Optimisation **globale** par algorithme evolutionnaire (`scipy.optimize.differential_evolution`).

- **Avantage** : explore tout l'espace des parametres, pas de dependance au point de depart
- **Inconvenient** : plus lent (population de candidats evaluee sur de nombreuses generations)
- **Strategie** : `best1bin`, mutation adaptative [0.5, 1.5], recombinaison 0.8

### Hybride DE + LM (`method="hybrid"`)

**Methode recommandee.** Combinaison des deux precedentes :

1. **Phase 1 — DE** : exploration globale rapide (moins d'iterations que le DE pur) pour trouver une bonne region
2. **Phase 2 — LM** : raffinement local precis depuis le resultat de la phase 1

Combine la robustesse de DE avec la precision de LM.

### Calibration multi-jours

Le calibrateur accepte une **liste de `MarketData`** (un par jour). Chaque jour garde son propre S0 et son propre pricer, mais les 5 parametres Heston sont **partages** : on suppose qu'ils sont stables sur la periode.

Les residus de tous les jours sont concatenes dans une seule fonction objectif, ce qui donne :
- Plus de donnees = calibration plus robuste
- Moins d'overfitting au bruit d'un seul snapshot
- La ponderation par vega normalise naturellement les differents niveaux de S0

---

## Resultats

Sur les donnees SPY 2024, la methode hybride donne :

- **RMSE IV Heston** : ~2-3%
- **RMSE IV Black-Scholes** : ~5-8%
- **Amelioration** : Heston 2-3x plus precis que BS

La condition de Feller est  **violee** sur donnees reelles (le marche price un vol of vol eleve), ce qui est un resultat connu en finance quantitative.
