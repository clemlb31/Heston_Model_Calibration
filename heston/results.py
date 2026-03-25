import numpy as np
from dataclasses import dataclass
from .params import HestonParams


@dataclass
class CalibrationResult:
    """Resultat d'une calibration."""
    params: HestonParams
    rmse_price: float
    rmse_iv: float
    elapsed_seconds: float
    method: str
    n_function_evals: int
    n_days: int = 1
    n_options: int = 0

    def __repr__(self):
        return (
            f"== {self.method} ==\n"
            f"  Temps      : {self.elapsed_seconds:.2f} s\n"
            f"  Evals      : {self.n_function_evals}\n"
            f"  Jours      : {self.n_days}\n"
            f"  Options    : {self.n_options}\n"
            f"  RMSE prix  : {self.rmse_price:.6f}\n"
            f"  RMSE IV    : {self.rmse_iv:.4f}  ({self.rmse_iv*100:.2f}%)\n"
            f"  {self.params}\n"
        )


def compare_results(results: list[CalibrationResult],
                    true_params: HestonParams | None = None):
    """Affiche un tableau comparatif des resultats."""
    print("\n" + "=" * 90)
    print("  COMPARAISON DES METHODES DE CALIBRATION")
    print("=" * 90)

    header = (
        f"{'Methode':<28} {'Temps':>8} {'Evals':>7} "
        f"{'RMSE IV':>9} {'v0':>7} {'k':>7} {'th':>7} {'sig':>7} {'rho':>7}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        p = r.params
        print(
            f"{r.method:<28} {r.elapsed_seconds:>7.2f}s {r.n_function_evals:>7} "
            f"{r.rmse_iv:>9.4f} {p.v0:>7.4f} {p.kappa:>7.4f} {p.theta:>7.4f} "
            f"{p.sigma:>7.4f} {p.rho:>7.4f}"
        )

    if true_params is not None:
        t = true_params
        print("-" * len(header))
        print(
            f"{'VRAIS PARAMETRES':<28} {'':>8} {'':>7} "
            f"{'':>9} {t.v0:>7.4f} {t.kappa:>7.4f} {t.theta:>7.4f} "
            f"{t.sigma:>7.4f} {t.rho:>7.4f}"
        )

    print("=" * 90)
    best = min(results, key=lambda r: r.rmse_iv)
    print(f"\n  Meilleure methode : {best.method} (RMSE IV = {best.rmse_iv:.4f})")
