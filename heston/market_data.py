import numpy as np
import pandas as pd
import json
from dataclasses import dataclass, field

from .black_scholes import BlackScholes
from .params import HestonParams
from .pricer import HestonPricer


@dataclass
class MarketData:
    """Conteneur pour les donnees de marche d'un jour (nappe de vol)."""
    S0: float
    r: float
    q: float
    date: str
    strikes: np.ndarray
    maturities: np.ndarray
    market_ivs: np.ndarray
    market_prices: np.ndarray = field(default=None)
    df: pd.DataFrame = field(default=None, repr=False)

    def __post_init__(self):
        if self.market_prices is None:
            self.market_prices = np.array([
                BlackScholes.price(self.S0, K, T, self.r, self.q, iv)
                for K, T, iv in zip(self.strikes, self.maturities, self.market_ivs)
            ])
        self.vegas = np.array([
            BlackScholes.vega(self.S0, K, T, self.r, self.q, iv)
            for K, T, iv in zip(self.strikes, self.maturities, self.market_ivs)
        ])
        self.weights = np.where(self.vegas > 1e-6, 1.0 / self.vegas, 0.0)

    def __repr__(self):
        n_mats = len(np.unique(self.maturities))
        return f"MarketData({self.date}, S0={self.S0:.2f}, {len(self.strikes)} opts, {n_mats} mats)"

    # ── Chargement depuis JSON Kaggle SPY ──

    @classmethod
    def from_json(
        cls,
        filepath: str,
        day_indices: list[int] | int | None = None,
        r: float = 0.05,
        q: float = 0.013,
        min_maturity_days: int = 14,
        max_maturity_days: int = 550,
        moneyness_range: tuple[float, float] = (0.85, 1.15),
        iv_range: tuple[float, float] = (0.03, 0.80),
        min_volume: int = 1,
        min_oi: int = 10,
        min_options_per_maturity: int = 5,
    ) -> list["MarketData"]:
        """
        Charge un ou plusieurs jours depuis un fichier JSON Kaggle SPY.

        Args:
            filepath: chemin vers le fichier JSON
            day_indices: index des jours a charger (int, liste, ou None=tous)
        """
        with open(filepath, 'r') as f:
            all_data = json.load(f)

        n_total = len(all_data)

        if day_indices is None:
            day_indices = list(range(n_total))
        elif isinstance(day_indices, int):
            day_indices = [day_indices]

        fname = filepath.split('/')[-1]
        print(f"Fichier : {fname} ({n_total} jours, {len(day_indices)} selectionnes)")

        filter_kwargs = dict(
            r=r, q=q,
            min_maturity_days=min_maturity_days,
            max_maturity_days=max_maturity_days,
            moneyness_range=moneyness_range,
            iv_range=iv_range,
            min_volume=min_volume,
            min_oi=min_oi,
            min_options_per_maturity=min_options_per_maturity,
        )

        markets = []
        for idx in day_indices:
            if idx >= n_total:
                print(f"  [SKIP] jour {idx} : hors limites (max={n_total-1})")
                continue
            try:
                market = cls._build_from_raw_day(all_data[idx], **filter_kwargs)
                if market is not None:
                    markets.append(market)
            except Exception as e:
                print(f"  [SKIP] jour {idx} : {e}")

        total_opts = sum(len(m.strikes) for m in markets)
        print(f"Total : {len(markets)} jours charges, {total_opts} options\n")
        return markets

    @classmethod
    def from_json_multi(
        cls,
        filepaths: list[str],
        days_per_file: int | None = 5,
        **kwargs,
    ) -> list["MarketData"]:
        """
        Charge depuis plusieurs fichiers JSON avec espacement uniforme.
        """
        all_markets = []
        for filepath in filepaths:
            dates = cls._list_dates(filepath)
            n_days = len(dates)

            if days_per_file is None or days_per_file >= n_days:
                indices = list(range(n_days))
            else:
                indices = np.linspace(0, n_days - 1, days_per_file, dtype=int).tolist()

            markets = cls.from_json(filepath, day_indices=indices, **kwargs)
            all_markets.extend(markets)

        return all_markets

    @staticmethod
    def _list_dates(filepath: str) -> list[str]:
        with open(filepath, 'r') as f:
            data = json.load(f)
        dates = []
        for day in data:
            for record in day:
                if record and 'date' in record:
                    dates.append(record['date'])
                    break
        return dates

    @classmethod
    def _build_from_raw_day(
        cls,
        raw_day: list[dict],
        r: float, q: float,
        min_maturity_days: int, max_maturity_days: int,
        moneyness_range: tuple, iv_range: tuple,
        min_volume: int, min_oi: int,
        min_options_per_maturity: int,
    ) -> "MarketData | None":
        """Construit un MarketData a partir des donnees brutes d'un jour."""
        records = [d for d in raw_day if d is not None]
        if not records:
            return None

        df = pd.DataFrame(records)

        num_cols = [
            'strike', 'last', 'mark', 'bid', 'ask', 'bid_size', 'ask_size',
            'volume', 'open_interest', 'implied_volatility',
            'delta', 'gamma', 'theta', 'vega', 'rho',
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['date'] = pd.to_datetime(df['date'])
        df['expiration'] = pd.to_datetime(df['expiration'])
        df['maturity_days'] = (df['expiration'] - df['date']).dt.days
        df['maturity_years'] = df['maturity_days'] / 365.25
        df['mid_price'] = (df['bid'] + df['ask']) / 2

        date_str = df['date'].iloc[0].strftime('%Y-%m-%d')

        # Calls uniquement
        df = df[df['type'] == 'call'].copy()
        if len(df) == 0:
            return None

        # Deduire S0 depuis les deep ITM calls
        S0 = cls._infer_spot(df)
        df['moneyness'] = df['strike'] / S0

        # Filtrage
        n_raw = len(df)
        mask = (
            (df['maturity_days'] >= min_maturity_days)
            & (df['maturity_days'] <= max_maturity_days)
            & (df['moneyness'] >= moneyness_range[0])
            & (df['moneyness'] <= moneyness_range[1])
            & (df['implied_volatility'] >= iv_range[0])
            & (df['implied_volatility'] <= iv_range[1])
            & (df['volume'] > min_volume)
            & (df['open_interest'] > min_oi)
            & (df['mid_price'] > 0.10)
            & (df['bid'] > 0.01)
            & (df['ask'] > df['bid'])
        )
        df = df[mask].copy()
        if len(df) == 0:
            return None

        # Bornes de non-arbitrage
        intrinsic = np.maximum(
            S0 * np.exp(-q * df['maturity_years'])
            - df['strike'] * np.exp(-r * df['maturity_years']),
            0,
        )
        upper = S0 * np.exp(-q * df['maturity_years'])
        df = df[(df['mid_price'] > intrinsic + 0.01) & (df['mid_price'] < upper)].copy()

        # Min options par maturite
        counts = df.groupby('maturity_days').size()
        good_mats = counts[counts >= min_options_per_maturity].index.tolist()
        df = df[df['maturity_days'].isin(good_mats)].copy()
        df = df.sort_values(['maturity_days', 'strike']).reset_index(drop=True)

        if len(df) < 10:
            return None

        n_mats = df['maturity_days'].nunique()
        print(f"  {date_str} : S0={S0:.2f}, {n_raw} -> {len(df)} opts, {n_mats} mats")

        return cls(
            S0=S0, r=r, q=q, date=date_str,
            strikes=df['strike'].values,
            maturities=df['maturity_years'].values,
            market_ivs=df['implied_volatility'].values,
            df=df,
        )

    @staticmethod
    def _infer_spot(df: pd.DataFrame) -> float:
        """Deduit S0 depuis les deep ITM calls a maturite courte."""
        short_mat = df[df['maturity_days'] <= 7]
        if len(short_mat) > 0:
            deep_itm = short_mat[short_mat['delta'] >= 0.99]
            if len(deep_itm) > 0:
                return (deep_itm['strike'] + deep_itm['mark']).median()
            top5 = short_mat.nlargest(5, 'delta')
            return (top5['strike'] + top5['mark']).median()
        atm = df.iloc[(df['delta'] - 0.5).abs().argsort()[:10]]
        return atm['strike'].median()

    # ── Donnees synthetiques ──

    @classmethod
    def generate_synthetic(
        cls,
        true_params: HestonParams | None = None,
        n_strikes: int = 8,
        n_maturities: int = 5,
        S0: float = 100.0,
        r: float = 0.03,
        q: float = 0.01,
        noise: float = 0.002,
    ) -> "MarketData":
        """Genere une nappe de vol synthetique pour validation."""
        if true_params is None:
            true_params = HestonParams(v0=0.04, kappa=1.5, theta=0.05, sigma=0.4, rho=-0.7)

        pricer = HestonPricer(S0, r, q)
        T_list = np.array([0.08, 0.17, 0.25, 0.5, 1.0])[:n_maturities]
        moneyness = np.linspace(0.85, 1.15, n_strikes)

        strikes_all, maturities_all, ivs_all = [], [], []
        for T in T_list:
            for m in moneyness:
                K = S0 * m
                price = pricer.call_price(K, T, true_params)
                if price > 1e-6:
                    iv = BlackScholes.implied_vol(price, S0, K, T, r, q)
                    iv += np.random.normal(0, noise)
                    iv = max(iv, 0.01)
                    strikes_all.append(K)
                    maturities_all.append(T)
                    ivs_all.append(iv)

        print(f"[Synthetique] S0={S0}, {len(ivs_all)} points, params={true_params}")

        return cls(
            S0=S0, r=r, q=q, date="synthetic",
            strikes=np.array(strikes_all),
            maturities=np.array(maturities_all),
            market_ivs=np.array(ivs_all),
        )
