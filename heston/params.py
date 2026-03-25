import numpy as np
from dataclasses import dataclass


@dataclass
class HestonParams:
    """Parametres du modele de Heston."""
    v0: float       # variance initiale
    kappa: float    # vitesse de retour a la moyenne
    theta: float    # variance long terme
    sigma: float    # vol of vol
    rho: float      # correlation spot / variance

    def to_array(self) -> np.ndarray:
        return np.array([self.v0, self.kappa, self.theta, self.sigma, self.rho])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "HestonParams":
        return cls(v0=x[0], kappa=x[1], theta=x[2], sigma=x[3], rho=x[4])

    @property
    def feller_satisfied(self) -> bool:
        """Condition de Feller : 2*kappa*theta > sigma^2."""
        return 2 * self.kappa * self.theta > self.sigma ** 2

    def __repr__(self):
        feller = "OK" if self.feller_satisfied else "NON"
        return (
            f"HestonParams(\n"
            f"  v0    = {self.v0:.6f}  (vol0  = {np.sqrt(self.v0):.4f})\n"
            f"  kappa = {self.kappa:.6f}\n"
            f"  theta = {self.theta:.6f}  (volLT = {np.sqrt(self.theta):.4f})\n"
            f"  sigma = {self.sigma:.6f}\n"
            f"  rho   = {self.rho:.6f}\n"
            f"  Feller: {feller}  (2kT={2*self.kappa*self.theta:.4f} vs s2={self.sigma**2:.4f})\n"
            f")"
        )
