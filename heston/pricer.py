import numpy as np
from .params import HestonParams
from .black_scholes import BlackScholes


class HestonPricer:
    """
    Pricing semi-analytique du modele de Heston.
    Formulation stable d'Albrecher et al. (2007).
    Integration par quadrature sur grille fixe.
    """

    def __init__(self, S0: float, r: float, q: float = 0.0,
                 n_points: int = 500, u_max: float = 100.0):
        self.S0 = S0
        self.r = r
        self.q = q
        self.u_grid = np.linspace(1e-8, u_max, n_points)
        self.du = self.u_grid[1] - self.u_grid[0]

    def _characteristic_function(self, u: np.ndarray, T: float,
                                  params: HestonParams, j: int) -> np.ndarray:
        """
        Fonction caracteristique de Heston (formulation stable).
        j=1 pour P1 (mesure spot), j=2 pour P2 (mesure forward).
        """
        v0, kappa, theta, sigma, rho = (
            params.v0, params.kappa, params.theta, params.sigma, params.rho
        )

        if j == 1:
            b = kappa - rho * sigma
            uj = 0.5
        else:
            b = kappa
            uj = -0.5

        a = kappa * theta
        iu = 1j * u

        d = np.sqrt((rho * sigma * iu - b)**2 - sigma**2 * (2 * uj * iu - u**2))
        g = (b - rho * sigma * iu - d) / (b - rho * sigma * iu + d)

        exp_dT = np.exp(-d * T)

        C = (self.r - self.q) * iu * T + (a / sigma**2) * (
            (b - rho * sigma * iu - d) * T
            - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
        )
        D = ((b - rho * sigma * iu - d) / sigma**2) * (
            (1.0 - exp_dT) / (1.0 - g * exp_dT)
        )

        return np.exp(C + D * v0 + iu * np.log(self.S0))

    def call_price(self, K: float, T: float, params: HestonParams) -> float:
        """Prix d'un call europeen via inversion de Fourier."""
        log_K = np.log(K)
        u = self.u_grid

        f1 = self._characteristic_function(u, T, params, j=1)
        f2 = self._characteristic_function(u, T, params, j=2)

        integrand1 = np.real(np.exp(-1j * u * log_K) * f1 / (1j * u))
        integrand2 = np.real(np.exp(-1j * u * log_K) * f2 / (1j * u))

        P1 = 0.5 + (1.0 / np.pi) * np.trapezoid(integrand1, dx=self.du)
        P2 = 0.5 + (1.0 / np.pi) * np.trapezoid(integrand2, dx=self.du)

        price = self.S0 * np.exp(-self.q * T) * P1 - K * np.exp(-self.r * T) * P2
        return max(price, 0.0)

    def call_prices_vectorized(self, strikes: np.ndarray,
                                maturities: np.ndarray,
                                params: HestonParams) -> np.ndarray:
        """Prix de plusieurs calls (vectorise sur la grille u, boucle sur K/T)."""
        prices = np.zeros(len(strikes))
        unique_T = np.unique(maturities)

        for T in unique_T:
            mask = maturities == T
            f1 = self._characteristic_function(self.u_grid, T, params, j=1)
            f2 = self._characteristic_function(self.u_grid, T, params, j=2)

            for i in np.where(mask)[0]:
                log_K = np.log(strikes[i])
                exp_factor = np.exp(-1j * self.u_grid * log_K)
                int1 = np.real(exp_factor * f1 / (1j * self.u_grid))
                int2 = np.real(exp_factor * f2 / (1j * self.u_grid))
                P1 = 0.5 + (1.0 / np.pi) * np.trapezoid(int1, dx=self.du)
                P2 = 0.5 + (1.0 / np.pi) * np.trapezoid(int2, dx=self.du)
                prices[i] = max(
                    self.S0 * np.exp(-self.q * T) * P1
                    - strikes[i] * np.exp(-self.r * T) * P2,
                    0.0,
                )
        return prices

    def implied_vol(self, K: float, T: float, params: HestonParams) -> float:
        """IV Heston pour un strike/maturite donne."""
        price = self.call_price(K, T, params)
        return BlackScholes.implied_vol(price, self.S0, K, T, self.r, self.q)
