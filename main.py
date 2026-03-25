"""
Calibration du modele de Heston sur donnees SPY.
Compare 3 methodes (LM, DE, Hybride) sur plusieurs jours.
"""

import warnings
from heston import (
    HestonParams,
    MarketData,
    HestonCalibrator,
    compare_results,
)

warnings.filterwarnings("ignore")


def run_real_data():
    """Calibration sur donnees SPY."""

    # Charger jours depuis un fichier
    markets = MarketData.from_json(
        "data/spy_options_data_24.json",
        day_indices=list(range(5)),  # 5 premiers jours
    )

    if not markets:
        print("Aucune donnee chargee.")
        return []

    # Calibrer les 3 methodes sur les memes jours
    calibrator = HestonCalibrator(markets)
    results = [
        calibrator.calibrate(method="lm"),
        calibrator.calibrate(method="de", de_maxiter=40),
        calibrator.calibrate(method="hybrid"),
    ]

    compare_results(results)
    return results


if __name__ == "__main__":
    run_real_data()
