import numpy as np
import time
from scipy.optimize import least_squares, differential_evolution

from .params import HestonParams
from .pricer import HestonPricer
from .black_scholes import BlackScholes
from .market_data import MarketData
from .results import CalibrationResult


class HestonCalibrator:
    """
    Calibrateur unifie du modele de Heston.

    Fonctionne sur un ou plusieurs jours de donnees.
    Chaque jour garde son propre S0/pricer, mais les parametres Heston
    sont partages (les residus sont agreges sur tous les jours).

    Methodes disponibles :
      - "lm"     : Levenberg-Marquardt (rapide, sensible au x0)
      - "de"     : Evolution Differentielle (global, plus lent)
      - "hybrid" : DE puis LM (recommande)
    """

    BOUNDS_LOWER = np.array([1e-4, 1e-3, 1e-4, 1e-2, -0.99])
    BOUNDS_UPPER = np.array([1.0,  20.0, 1.0,  3.0,   0.50])

    def __init__(self, markets: list[MarketData] | MarketData):
        if isinstance(markets, MarketData):
            markets = [markets]

        self.markets = markets
        self.pricers = [HestonPricer(m.S0, m.r, m.q) for m in markets]
        self.total_options = sum(len(m.strikes) for m in markets)
        self._n_evals = 0

    def _all_residuals(self, x: np.ndarray) -> np.ndarray:
        """Residus ponderes par vega, agreges sur tous les jours."""
        self._n_evals += 1
        params = HestonParams.from_array(x)
        all_res = []
        for market, pricer in zip(self.markets, self.pricers):
            try:
                model_prices = pricer.call_prices_vectorized(
                    market.strikes, market.maturities, params
                )
                residuals = (model_prices - market.market_prices) * market.weights
            except Exception:
                residuals = np.full(len(market.strikes), 1e6)
            all_res.append(residuals)
        return np.concatenate(all_res)

    def _scalar_objective(self, x: np.ndarray) -> float:
        res = self._all_residuals(x)
        return np.sum(res**2)

    def calibrate(
        self,
        method: str = "hybrid",
        x0: np.ndarray | None = None,
        de_maxiter: int = 25,
        de_popsize: int = 12,
        lm_max_nfev: int = 300,
        seed: int = 42,
    ) -> CalibrationResult:
        """
        Lance la calibration.

        Args:
            method: "lm", "de", ou "hybrid"
            x0: point de depart pour LM (auto si None)
            de_maxiter: iterations max pour DE
            de_popsize: taille population DE
            lm_max_nfev: evaluations max pour LM
            seed: graine aleatoire pour DE
        """
        n_days = len(self.markets)
        method = method.lower()
        label = {
            "lm": "Levenberg-Marquardt",
            "de": "Evolution Differentielle",
            "hybrid": "Hybride (DE + LM)",
        }[method]

        print(f"\n>> {label} ({n_days} jour(s), {self.total_options} options)...")
        self._n_evals = 0
        t0 = time.time()

        if method == "lm":
            x_opt = self._run_lm(x0, lm_max_nfev)
        elif method == "de":
            x_opt = self._run_de(de_maxiter, de_popsize, seed)
        elif method == "hybrid":
            x_de = self._run_de(de_maxiter, de_popsize, seed)
            print(f"  Phase DE terminee, raffinement LM...")
            x_opt = self._run_lm(x_de, lm_max_nfev)
        else:
            raise ValueError(f"Methode inconnue : {method}")

        return self._build_result(x_opt, t0, label)

    def _run_lm(self, x0: np.ndarray | None, max_nfev: int) -> np.ndarray:
        if x0 is None:
            atm_iv = np.mean(
                np.concatenate([m.market_ivs for m in self.markets])
            )
            x0 = np.array([atm_iv**2, 1.5, atm_iv**2, 0.3, -0.7])

        result = least_squares(
            self._all_residuals,
            x0,
            bounds=(self.BOUNDS_LOWER, self.BOUNDS_UPPER),
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
            max_nfev=max_nfev,
        )
        return result.x

    def _run_de(self, maxiter: int, popsize: int, seed: int) -> np.ndarray:
        bounds = list(zip(self.BOUNDS_LOWER, self.BOUNDS_UPPER))
        result = differential_evolution(
            self._scalar_objective,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-8,
            seed=seed,
            strategy="best1bin",
            mutation=(0.5, 1.5),
            recombination=0.8,
        )
        return result.x

    def _build_result(self, x_opt: np.ndarray, t0: float,
                      method_name: str) -> CalibrationResult:
        params = HestonParams.from_array(x_opt)

        all_model_prices, all_market_prices = [], []
        all_model_ivs, all_market_ivs = [], []

        for market, pricer in zip(self.markets, self.pricers):
            mp = pricer.call_prices_vectorized(
                market.strikes, market.maturities, params
            )
            all_model_prices.append(mp)
            all_market_prices.append(market.market_prices)
            all_market_ivs.append(market.market_ivs)

            ivs = np.array([
                BlackScholes.implied_vol(p, market.S0, K, T, market.r, market.q)
                for p, K, T in zip(mp, market.strikes, market.maturities)
            ])
            all_model_ivs.append(ivs)

        all_model_prices = np.concatenate(all_model_prices)
        all_market_prices = np.concatenate(all_market_prices)
        all_model_ivs = np.concatenate(all_model_ivs)
        all_market_ivs = np.concatenate(all_market_ivs)

        rmse_price = np.sqrt(np.mean((all_model_prices - all_market_prices)**2))
        rmse_iv = np.sqrt(np.mean((all_model_ivs - all_market_ivs)**2))

        return CalibrationResult(
            params=params,
            rmse_price=rmse_price,
            rmse_iv=rmse_iv,
            elapsed_seconds=time.time() - t0,
            method=method_name,
            n_function_evals=self._n_evals,
            n_days=len(self.markets),
            n_options=self.total_options,
        )
