# -*- coding: utf-8 -*-
"""
Neural Network Options Pricer with Greeks & Volatility Surface
Author: Spencer Purdy

Description: Production-grade options pricing system using Physics-Informed Neural Networks
             (PINNs), deep learning for Greeks calculation, implied volatility surface modeling,
             American options pricing with early exercise boundary, deep hedging strategies using
             reinforcement learning, and arbitrage detection across option chains.

             Pre-trained models are loaded at startup from files in the repository root:
               - pinn_model.pth           : PINN weights and normalization statistics
               - ppo_hedging_agent.zip    : PPO hedging agent (stable-baselines3)
               - training_metrics.json    : Training history and evaluation results
               - options_data.csv         : Real AAPL options data used for training

Problem Statement: Develop an advanced options pricing system that leverages machine learning to
                   complement classical models like Black-Scholes, while providing real-time Greeks,
                   volatility surface analysis, and optimal hedging strategies.

Real-World Application: Derivatives trading desks, options market makers, hedge funds, quantitative
                        trading firms, and institutional investors requiring advanced options analytics.

Key Features:
- Physics-Informed Neural Networks (PINNs) incorporating Black-Scholes PDE
- Automatic differentiation for exact Greeks computation (Delta, Gamma, Vega, Theta, Rho)
- Implied volatility surface modeling and 3D visualization
- American options pricing with Least Squares Monte Carlo and early exercise boundary
- Deep reinforcement learning (PPO) for optimal delta hedging strategies
- Arbitrage detection across option chains (put-call parity, butterfly spreads)
- Trained on 100% real AAPL options data from yfinance
- Interactive Gradio interface with 8 analysis tabs

Technical Components:
- PINN architecture with input/output normalization and Black-Scholes PDE loss
- Stock-price-augmented training data for accurate autograd Greeks
- RBF interpolation for volatility surface reconstruction
- Least-Squares Monte Carlo (LSM) for American options
- PPO agent with normalized observations for dynamic hedging
- Put-call parity scanning with dividend yield adjustment

Limitations:
- Assumes European options for PINN (American via LSM)
- Transaction costs simplified (no market impact)
- Liquidity constraints not modeled
- Volatility surface assumes smooth interpolation
- Model risk from neural network approximation
- Limited to equity options (no exotic derivatives)

License: MIT License
Author: Spencer Purdy
Purpose: Portfolio demonstration of derivatives pricing and deep learning skills

Disclaimer: This is a simulation system for educational and portfolio demonstration purposes.
            Real options trading involves significant risks and regulatory requirements.
"""

# ============================================================================
# INSTALLATION (uncomment to run in Colab or local)
# ============================================================================

# !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# !pip install -q stable-baselines3[extra] gymnasium pandas numpy requests plotly gradio scipy scikit-learn matplotlib

# ============================================================================
# IMPORTS
# ============================================================================

import os
import json
import time
import random
import warnings
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

import numpy as np
np.random.seed(RANDOM_SEED)

import pandas as pd
from scipy.stats import norm
from scipy.interpolate import griddata

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from stable_baselines3 import PPO
import gymnasium as gym

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CPU inference on HF Spaces
DEVICE = "cpu"

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class OptionsPricerConfig:
    """Configuration for Neural Network Options Pricer.

    Contains all hyperparameters, architecture settings, and market parameters.
    Values are populated from training_metrics.json at startup so the inference
    environment matches the training environment exactly.
    """

    random_seed: int = 42
    device: str = DEVICE

    # Market parameters (overwritten from training_metrics.json at startup)
    symbol: str = "AAPL"
    stock_price: float = 150.0
    historical_vol: float = 0.30
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0

    # PINN architecture (must match the saved checkpoint)
    pinn_hidden_dims: List[int] = field(default_factory=lambda: [128, 256, 256, 128])
    pinn_activation: str = "tanh"

    # Normalization statistics (loaded from checkpoint at startup)
    input_mean: Optional[List[float]] = None
    input_std: Optional[List[float]] = None
    output_mean: float = 1.0
    output_std: float = 1.0

    # Implied volatility solver
    iv_initial_guess: float = 0.3
    iv_max_iterations: int = 100
    iv_tolerance: float = 1e-6

    # Volatility surface grid resolution
    vol_surface_strikes: int = 25
    vol_surface_maturities: int = 15

    # American options (Least Squares Monte Carlo)
    american_simulation_paths: int = 10000
    american_time_steps: int = 100
    lsm_basis_functions: int = 5

    # Deep hedging (overwritten from training_metrics.json)
    hedging_S0: float = 150.0
    hedging_K: float = 150.0
    hedging_T: float = 0.25
    hedging_r: float = 0.05
    hedging_sigma: float = 0.30
    hedging_n_steps: int = 63

    # Arbitrage detection: minimum profit (dollars) to flag as opportunity.
    # Uses a fixed dollar threshold instead of a fraction of underlying
    # to avoid flooding results with small violations that vanish after
    # transaction costs and bid-ask spreads.
    arbitrage_min_profit: float = 2.00

    # Train/test split ratio (for reproducing the test set)
    train_test_split: float = 0.8

# ============================================================================
# BLACK-SCHOLES MODEL
# ============================================================================

class BlackScholesModel:
    """Classical Black-Scholes option pricing model.

    Provides analytical solutions for European option prices and all five
    standard Greeks. Used as the benchmark for PINN evaluation and as the
    pricing engine inside the hedging environment.

    Supports continuous dividend yield via the Merton extension:
        C = S*exp(-qT)*N(d1) - K*exp(-rT)*N(d2)
    where q is the continuous dividend yield.
    """

    def __init__(self, r: float = 0.05, q: float = 0.0):
        """Initialize with risk-free rate and continuous dividend yield."""
        self.r = r
        self.q = q

    def _d1(self, S: float, K: float, T: float, sigma: float) -> float:
        """Compute d1 of the Black-Scholes formula."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (np.log(S / K) + (self.r - self.q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    def _d2(self, S: float, K: float, T: float, sigma: float) -> float:
        """Compute d2 of the Black-Scholes formula."""
        return self._d1(S, K, T, sigma) - sigma * np.sqrt(T)

    def call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """European call option price."""
        if T <= 0:
            return max(S - K, 0)
        d1 = self._d1(S, K, T, sigma)
        d2 = self._d2(S, K, T, sigma)
        return max(S * np.exp(-self.q * T) * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2), 0)

    def put_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """European put option price."""
        if T <= 0:
            return max(K - S, 0)
        d1 = self._d1(S, K, T, sigma)
        d2 = self._d2(S, K, T, sigma)
        return max(K * np.exp(-self.r * T) * norm.cdf(-d2) - S * np.exp(-self.q * T) * norm.cdf(-d1), 0)

    def delta(self, S: float, K: float, T: float, sigma: float, option_type: str = 'call') -> float:
        """Option Delta: sensitivity of price to underlying."""
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            return -1.0 if S < K else 0.0
        d1 = self._d1(S, K, T, sigma)
        if option_type == 'call':
            return np.exp(-self.q * T) * norm.cdf(d1)
        return -np.exp(-self.q * T) * norm.cdf(-d1)

    def gamma(self, S: float, K: float, T: float, sigma: float) -> float:
        """Option Gamma: second derivative of price with respect to underlying."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = self._d1(S, K, T, sigma)
        return np.exp(-self.q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def vega(self, S: float, K: float, T: float, sigma: float) -> float:
        """Option Vega: sensitivity to volatility (per 1% change)."""
        if T <= 0:
            return 0.0
        d1 = self._d1(S, K, T, sigma)
        return S * np.exp(-self.q * T) * norm.pdf(d1) * np.sqrt(T) / 100

    def theta(self, S: float, K: float, T: float, sigma: float, option_type: str = 'call') -> float:
        """Option Theta: time decay per calendar day."""
        if T <= 0:
            return 0.0
        d1 = self._d1(S, K, T, sigma)
        d2 = self._d2(S, K, T, sigma)
        term1 = -(S * np.exp(-self.q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type == 'call':
            term2 = self.q * S * np.exp(-self.q * T) * norm.cdf(d1)
            term3 = -self.r * K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            term2 = -self.q * S * np.exp(-self.q * T) * norm.cdf(-d1)
            term3 = self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)
        return (term1 + term2 + term3) / 365

    def rho(self, S: float, K: float, T: float, sigma: float, option_type: str = 'call') -> float:
        """Option Rho: sensitivity to interest rate (per 1% change)."""
        if T <= 0:
            return 0.0
        d2 = self._d2(S, K, T, sigma)
        if option_type == 'call':
            return K * T * np.exp(-self.r * T) * norm.cdf(d2) / 100
        return -K * T * np.exp(-self.r * T) * norm.cdf(-d2) / 100

    def implied_volatility(self, option_price: float, S: float, K: float,
                           T: float, option_type: str = 'call') -> float:
        """Newton-Raphson implied volatility solver."""
        if T <= 0 or option_price <= 0:
            return 0.3
        sigma = 0.3
        for _ in range(100):
            if option_type == 'call':
                price = self.call_price(S, K, T, sigma)
            else:
                price = self.put_price(S, K, T, sigma)
            vega_val = self.vega(S, K, T, sigma) * 100
            diff = price - option_price
            if abs(diff) < 1e-6:
                return sigma
            if vega_val > 1e-10:
                sigma = sigma - diff / vega_val
                sigma = max(0.01, min(sigma, 5.0))
            else:
                break
        return sigma

# ============================================================================
# PHYSICS-INFORMED NEURAL NETWORK (PINN)
# ============================================================================

class PINN(nn.Module):
    """Physics-Informed Neural Network for option pricing.

    Incorporates the Black-Scholes PDE as a physics constraint during training.
    Uses input standardization and output scaling for stable learning with Tanh
    activations. Normalization statistics are stored as PyTorch buffers so they
    persist through save/load and are moved with .to(device).

    The model was trained on call options with stock-price-augmented data to
    ensure meaningful partial derivatives (Greeks) via automatic differentiation.

    Architecture:
        Input: [S, K, T, sigma, r] -- 5 features, standardized internally
        Hidden: [128, 256, 256, 128] with Tanh activation
        Output: Option price in original dollar units

    Normalization:
        Input:  x_norm = (x_raw - input_mean) / input_std
        Output: price = network(x_norm) * output_std + output_mean
    """

    def __init__(self, hidden_dims: List[int], activation: str,
                 input_mean: np.ndarray, input_std: np.ndarray,
                 output_mean: float, output_std: float):
        """Initialize PINN with architecture and normalization parameters.

        Args:
            hidden_dims: List of hidden layer widths
            activation: Activation function name ('tanh', 'relu', 'swish')
            input_mean: Per-feature mean from training data (shape 5)
            input_std: Per-feature std from training data (shape 5)
            output_mean: Mean of training target prices
            output_std: Std of training target prices
        """
        super(PINN, self).__init__()

        # Normalization parameters stored as registered buffers
        self.register_buffer('input_mean', torch.FloatTensor(input_mean))
        self.register_buffer('input_std', torch.FloatTensor(input_std))
        self.output_mean = output_mean
        self.output_std = output_std

        # Build sequential network
        input_dim = 5
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'swish':
                layers.append(nn.SiLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier uniform initialization for stable gradient flow."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Forward pass with internal normalization.

        Args:
            x_raw: [batch, 5] tensor with [S, K, T, sigma, r] in original units

        Returns:
            Option price predictions [batch, 1] in dollar units
        """
        x_norm = (x_raw - self.input_mean) / self.input_std
        z = self.network(x_norm)
        return z * self.output_std + self.output_mean

    def compute_greeks(self, S: float, K: float, T: float,
                       sigma: float, r: float) -> Dict[str, float]:
        """Compute all five Greeks using PyTorch automatic differentiation.

        Uses autograd to compute exact partial derivatives of the network
        output with respect to each raw input variable. Because the PINN was
        trained with stock-price-augmented data, the dV/dS gradient reflects
        real price sensitivity rather than normalization artifacts.

        Args:
            S: Stock price
            K: Strike price
            T: Time to maturity (years)
            sigma: Implied volatility (annual)
            r: Risk-free rate

        Returns:
            Dictionary with price and all five Greeks
        """
        self.eval()
        x = torch.tensor([[S, K, T, sigma, r]], dtype=torch.float32, requires_grad=True)
        V = self.forward(x)

        # First-order gradients with respect to the full input vector
        dV = torch.autograd.grad(V, x, create_graph=True, retain_graph=True)[0]

        # Delta: dV/dS
        delta = dV[0, 0]

        # Gamma: d2V/dS2
        d2V = torch.autograd.grad(delta, x, create_graph=True, retain_graph=True)[0]
        gamma = d2V[0, 0]

        # Vega: dV/dsigma, scaled per 1% change in vol
        vega = dV[0, 3] / 100.0

        # Theta: -dV/dT (negative because T is time-to-maturity), per calendar day
        theta = -dV[0, 2] / 365.0

        # Rho: dV/dr, scaled per 1% change in rate
        rho = dV[0, 4] / 100.0

        return {
            'price': V.item(),
            'delta': delta.item(),
            'gamma': gamma.item(),
            'vega': vega.item(),
            'theta': theta.item(),
            'rho': rho.item()
        }

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Predict option prices for a DataFrame of options.

        Args:
            df: DataFrame with columns underlying_price, strike, maturity,
                volatility, risk_free_rate

        Returns:
            1D numpy array of predicted prices
        """
        self.eval()
        cols = ['underlying_price', 'strike', 'maturity', 'volatility', 'risk_free_rate']
        X = torch.FloatTensor(df[cols].values)
        with torch.no_grad():
            preds = self.forward(X).numpy().flatten()
        return preds

# ============================================================================
# IMPLIED VOLATILITY SURFACE
# ============================================================================

class VolatilitySurface:
    """Implied volatility surface modeling and 3D visualization.

    Fits a smooth surface through implied volatility observations using
    scipy griddata interpolation, parameterized by moneyness (K/S) and
    time to maturity.
    """

    def __init__(self, config: OptionsPricerConfig):
        self.config = config
        self.bs = BlackScholesModel(r=config.risk_free_rate, q=config.dividend_yield)
        self.surface_data = None

    def fit(self, options_data: pd.DataFrame) -> None:
        """Fit volatility surface from options data.

        Uses market-provided implied volatility if the 'volatility' column is
        present, otherwise computes IV from market prices using Newton-Raphson.
        """
        logger.info("Fitting volatility surface...")
        df = options_data.copy()

        if 'volatility' in df.columns:
            df['implied_vol'] = df['volatility']
        else:
            ivs = []
            for _, row in df.iterrows():
                try:
                    iv = self.bs.implied_volatility(
                        row['market_price'], row['underlying_price'],
                        row['strike'], row['maturity'],
                        row.get('option_type', 'call')
                    )
                    ivs.append(iv)
                except Exception:
                    ivs.append(np.nan)
            df['implied_vol'] = ivs

        # Filter valid IVs and compute moneyness
        valid = df[(df['implied_vol'] > 0.01) & (df['implied_vol'] < 3.0)].copy()
        valid['moneyness'] = valid['strike'] / valid['underlying_price']
        self.surface_data = valid
        logger.info(f"Fitted volatility surface with {len(valid)} data points")

    def create_surface_grid(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Create interpolated grid for 3D surface visualization.

        Returns:
            Tuple of (moneyness_grid, maturity_grid, iv_grid) or Nones
        """
        if self.surface_data is None or len(self.surface_data) < 10:
            return None, None, None

        moneyness_range = np.linspace(
            self.surface_data['moneyness'].quantile(0.05),
            self.surface_data['moneyness'].quantile(0.95),
            self.config.vol_surface_strikes
        )
        maturity_range = np.linspace(
            self.surface_data['maturity'].quantile(0.05),
            self.surface_data['maturity'].quantile(0.95),
            self.config.vol_surface_maturities
        )

        moneyness_grid, maturity_grid = np.meshgrid(moneyness_range, maturity_range)
        points = self.surface_data[['moneyness', 'maturity']].values
        values = self.surface_data['implied_vol'].values

        iv_grid = griddata(points, values, (moneyness_grid, maturity_grid),
                           method='cubic', fill_value=np.nan)
        mask = np.isnan(iv_grid)
        if mask.any():
            iv_grid[mask] = griddata(
                points, values,
                (moneyness_grid[mask], maturity_grid[mask]),
                method='linear', fill_value=np.nanmean(values)
            )

        return moneyness_grid, maturity_grid, iv_grid

# ============================================================================
# AMERICAN OPTIONS PRICING (LEAST SQUARES MONTE CARLO)
# ============================================================================

class AmericanOptionPricer:
    """Price American options using the Longstaff-Schwartz LSM method.

    Uses Monte Carlo simulation with backward induction and Laguerre polynomial
    regression to determine the optimal early exercise boundary.
    """

    def __init__(self, config: OptionsPricerConfig):
        self.config = config

    def _laguerre_basis(self, x: np.ndarray, degree: int) -> np.ndarray:
        """Compute Laguerre polynomial basis functions for regression."""
        basis = np.zeros((len(x), degree + 1))
        basis[:, 0] = 1
        if degree >= 1:
            basis[:, 1] = 1 - x
        for i in range(2, degree + 1):
            basis[:, i] = ((2 * i - 1 - x) * basis[:, i-1] - (i - 1) * basis[:, i-2]) / i
        return basis

    def price_american_put(self, S0: float, K: float, T: float,
                           r: float, sigma: float) -> Dict[str, Any]:
        """Price American put option using LSM method.

        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility

        Returns:
            Dictionary with price, exercise boundary, sample paths, and time grid
        """
        logger.info(f"Pricing American put: S={S0}, K={K}, T={T}, r={r}, sigma={sigma}")

        n_paths = self.config.american_simulation_paths
        n_steps = self.config.american_time_steps
        dt = T / n_steps

        np.random.seed(self.config.random_seed)
        Z = np.random.standard_normal((n_paths, n_steps))
        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0
        for t in range(n_steps):
            S[:, t+1] = S[:, t] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t])

        cash_flows = np.maximum(K - S[:, -1], 0)
        exercise_times = np.full(n_paths, T)

        for t in range(n_steps - 1, 0, -1):
            discount = np.exp(-r * dt)
            cash_flows *= discount
            itm = S[:, t] < K

            if np.sum(itm) > 0:
                X = S[itm, t]
                Y = cash_flows[itm]
                basis = self._laguerre_basis(X / K, self.config.lsm_basis_functions)
                coeffs = np.linalg.lstsq(basis, Y, rcond=None)[0]
                continuation_value = basis @ coeffs
                exercise_value = K - X
                exercise = exercise_value > continuation_value

                itm_indices = np.where(itm)[0]
                exercise_indices = itm_indices[exercise]
                cash_flows[exercise_indices] = exercise_value[exercise]
                exercise_times[exercise_indices] = t * dt

        option_price = np.mean(cash_flows * np.exp(-r * dt))

        boundary = []
        for t in range(n_steps + 1):
            exercised_at_t = np.abs(exercise_times - t * dt) < dt / 2
            if np.sum(exercised_at_t) > 0:
                boundary.append(np.mean(S[exercised_at_t, t]))
            else:
                boundary.append(np.nan)

        return {
            'price': option_price,
            'exercise_boundary': boundary,
            'time_grid': np.linspace(0, T, n_steps + 1),
            'paths': S[:100, :],
            'exercise_times': exercise_times[:100]
        }

# ============================================================================
# DEEP HEDGING ENVIRONMENT
# ============================================================================

class HedgingEnvironment(gym.Env):
    """Gymnasium environment for deep delta hedging with normalized observations.

    The agent manages a portfolio consisting of a short call option and a
    dynamic hedge position in the underlying stock. At each step, the agent
    chooses a target hedge ratio. The environment executes the trade, simulates
    stock price evolution via GBM, and rewards low PnL variance.

    Observation space (all values roughly in [0, 1] for stable learning):
        [S/S0, ttm/T, hedge_position, bs_delta]
        - S/S0:            normalized stock price (starts at 1.0)
        - ttm/T:           fraction of time remaining (1.0 -> 0.0)
        - hedge_position:  current shares held
        - bs_delta:        Black-Scholes delta as a learning signal

    Action space:
        Continuous [-1, 1] mapped to target hedge ratio [0, 1.5]

    Reward:
        -(pnl / S0)^2, encouraging zero PnL (perfect replication)
    """

    def __init__(self, S0, K, T, r, sigma, n_steps=63):
        super(HedgingEnvironment, self).__init__()
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.bs = BlackScholesModel(r=r)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([3.0, 1.0, 1.5, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.S = self.S0
        self.t = 0.0
        self.hedge_position = 0.0
        self.cash = 0.0
        self.option_value = self.bs.call_price(self.S, self.K, self.T, self.sigma)
        self.initial_option_value = self.option_value
        self.pnl_history = []
        return self._get_obs(), {}

    def _get_obs(self):
        """Return current observation vector with normalized values."""
        ttm = max(self.T - self.t, 0.0)
        ttm_frac = ttm / self.T if self.T > 0 else 0.0
        bs_delta = self.bs.delta(self.S, self.K, ttm, self.sigma, 'call') if ttm > 0.001 else (1.0 if self.S > self.K else 0.0)
        return np.array([
            self.S / self.S0,
            ttm_frac,
            self.hedge_position,
            bs_delta
        ], dtype=np.float32)

    def step(self, action):
        """Execute one hedging step."""
        target_hedge = float(action[0] + 1.0) * 0.75
        target_hedge = np.clip(target_hedge, 0.0, 1.5)
        hedge_change = target_hedge - self.hedge_position

        txn_cost = 0.0001 * abs(hedge_change) * self.S
        self.cash -= hedge_change * self.S + txn_cost
        self.hedge_position = target_hedge

        dW = np.random.standard_normal()
        self.S = self.S * np.exp(
            (self.r - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * dW
        )
        self.t += self.dt
        self.current_step += 1

        ttm = max(self.T - self.t, 0.0)
        if ttm > 0.001:
            self.option_value = self.bs.call_price(self.S, self.K, ttm, self.sigma)
        else:
            self.option_value = max(self.S - self.K, 0)

        portfolio_value = -self.option_value + self.hedge_position * self.S + self.cash
        pnl = portfolio_value + self.initial_option_value
        self.pnl_history.append(pnl)

        reward = float(-(pnl / self.S0) ** 2)

        done = self.current_step >= self.n_steps
        if done:
            final_payoff = max(self.S - self.K, 0)
            final_pnl = -final_payoff + self.hedge_position * self.S + self.cash + self.initial_option_value
            reward = float(-(final_pnl / self.S0) ** 2)
            pnl = final_pnl

        return self._get_obs(), reward, done, False, {'pnl': pnl}

    def render(self, mode='human'):
        pass

# ============================================================================
# ARBITRAGE DETECTOR
# ============================================================================

class ArbitrageDetector:
    """Detect arbitrage opportunities in option chains.

    Checks two categories of no-arbitrage conditions:
    1. Put-call parity with dividend adjustment:
       C - P = S*exp(-qT) - K*exp(-rT)
       where q is the continuous dividend yield.
    2. Butterfly spread convexity: C(K2) <= w1*C(K1) + w3*C(K3)

    A fixed dollar threshold (arbitrage_min_profit) filters out violations
    that would vanish after real-world transaction costs and bid-ask spreads.
    """

    def __init__(self, config: OptionsPricerConfig):
        self.config = config

    def check_put_call_parity(self, call_price: float, put_price: float,
                               S: float, K: float, T: float, r: float,
                               q: float = 0.0) -> Dict[str, Any]:
        """Check put-call parity: C - P = S*exp(-qT) - K*exp(-rT).

        The dividend-adjusted version accounts for continuous dividend yield q,
        which shifts the forward price of the stock downward. Without this
        adjustment, dividend-paying stocks produce systematic false positives.
        """
        theoretical_diff = S * np.exp(-q * T) - K * np.exp(-r * T)
        actual_diff = call_price - put_price
        arbitrage_amount = abs(actual_diff - theoretical_diff)
        is_arbitrage = arbitrage_amount > self.config.arbitrage_min_profit

        if is_arbitrage:
            if actual_diff > theoretical_diff:
                strategy = "Buy put, sell call, buy stock, borrow cash"
            else:
                strategy = "Buy call, sell put, sell stock, lend cash"
        else:
            strategy = "No arbitrage"

        return {
            'is_arbitrage': is_arbitrage,
            'arbitrage_amount': arbitrage_amount,
            'strategy': strategy,
            'theoretical_diff': theoretical_diff,
            'actual_diff': actual_diff
        }

    def check_butterfly_spread(self, lower_price: float, middle_price: float,
                                upper_price: float, K1: float, K2: float, K3: float) -> Dict[str, Any]:
        """Check butterfly spread convexity condition.

        No-arbitrage requires: C(K2) <= w1*C(K1) + w3*C(K3)
        where w1 = (K3-K2)/(K3-K1), w3 = (K2-K1)/(K3-K1).
        """
        weight_lower = (K3 - K2) / (K3 - K1)
        weight_upper = (K2 - K1) / (K3 - K1)
        theoretical_middle = weight_lower * lower_price + weight_upper * upper_price
        violation_amount = max(0, middle_price - theoretical_middle)
        is_arbitrage = violation_amount > self.config.arbitrage_min_profit

        return {
            'is_arbitrage': is_arbitrage,
            'arbitrage_amount': violation_amount,
            'strategy': "Sell middle strike, buy butterfly" if is_arbitrage else "No arbitrage",
            'theoretical_price': theoretical_middle,
            'actual_price': middle_price
        }

    def scan_chain(self, options_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Scan entire options chain for arbitrage opportunities."""
        logger.info("Scanning options chain for arbitrage opportunities...")
        arbitrages = []
        q = self.config.dividend_yield
        r = self.config.risk_free_rate

        grouped = options_data.groupby(['maturity', 'strike'])
        for (maturity, strike), group in grouped:
            calls = group[group['option_type'] == 'call']
            puts = group[group['option_type'] == 'put']
            if len(calls) > 0 and len(puts) > 0:
                result = self.check_put_call_parity(
                    calls['market_price'].iloc[0], puts['market_price'].iloc[0],
                    calls['underlying_price'].iloc[0], strike, maturity, r, q
                )
                if result['is_arbitrage']:
                    arbitrages.append({
                        'type': 'Put-Call Parity',
                        'strike': strike, 'maturity': maturity, **result
                    })

        call_data = options_data[options_data['option_type'] == 'call']
        for maturity, group in call_data.groupby('maturity'):
            sorted_g = group.sort_values('strike')
            strikes = sorted_g['strike'].values
            prices = sorted_g['market_price'].values
            for i in range(len(strikes) - 2):
                K1, K2, K3 = strikes[i], strikes[i+1], strikes[i+2]
                if K3 - K1 > 0:
                    result = self.check_butterfly_spread(prices[i], prices[i+1], prices[i+2], K1, K2, K3)
                    if result['is_arbitrage']:
                        arbitrages.append({
                            'type': 'Butterfly Spread',
                            'strikes': f"${K1:.0f} / ${K2:.0f} / ${K3:.0f}",
                            'maturity': maturity, **result
                        })

        logger.info(f"Found {len(arbitrages)} potential arbitrage opportunities")
        return arbitrages

# ============================================================================
# LOAD PRE-TRAINED MODELS AND DATA AT STARTUP
# ============================================================================

def load_pretrained_models(config: OptionsPricerConfig) -> Tuple[Optional[PINN], Optional[Any], Optional[Dict], Optional[pd.DataFrame]]:
    """Load all pre-trained model files from the repository root.

    Reads training_metrics.json first to extract normalization statistics and
    training configuration, then reconstructs the PINN architecture and loads
    weights. Also loads the PPO hedging agent and the options CSV.

    Returns:
        Tuple of (pinn_model, hedging_agent, training_metrics, options_data)
    """
    pinn_model = None
    hedging_agent = None
    training_metrics = None
    options_data = None

    # ---- Training metrics (load first for config values) ----
    metrics_path = "training_metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                training_metrics = json.load(f)
            logger.info(f"Loaded training metrics from {metrics_path}")

            config.stock_price = training_metrics.get('stock_price', config.stock_price)
            config.historical_vol = training_metrics.get('historical_vol', config.historical_vol)
            config.symbol = training_metrics.get('symbol', config.symbol)
            config.dividend_yield = training_metrics.get('dividend_yield', config.dividend_yield)

            tc = training_metrics.get('config', {})
            config.risk_free_rate = tc.get('risk_free_rate', config.risk_free_rate)
            config.pinn_hidden_dims = tc.get('pinn_hidden_dims', config.pinn_hidden_dims)
            config.pinn_activation = tc.get('pinn_activation', config.pinn_activation)
            config.hedging_S0 = tc.get('hedging_S0', config.hedging_S0)
            config.hedging_K = tc.get('hedging_K', config.hedging_K)
            config.hedging_T = tc.get('hedging_T', config.hedging_T)
            config.hedging_r = tc.get('hedging_r', config.hedging_r)
            config.hedging_sigma = tc.get('hedging_sigma', config.hedging_sigma)
            config.hedging_n_steps = tc.get('hedging_n_steps_per_episode', config.hedging_n_steps)

            norm = training_metrics.get('normalization', {})
            config.input_mean = norm.get('input_mean')
            config.input_std = norm.get('input_std')
            config.output_mean = norm.get('output_mean', 1.0)
            config.output_std = norm.get('output_std', 1.0)

            logger.info(f"  Symbol: {config.symbol}, Price: ${config.stock_price:.2f}, "
                        f"Vol: {config.historical_vol:.2%}, Div: {config.dividend_yield:.2%}")
        except Exception as e:
            logger.error(f"Error loading training metrics: {e}")
    else:
        logger.warning(f"Training metrics not found at {metrics_path}")

    # ---- PINN model ----
    pinn_path = "pinn_model.pth"
    if os.path.exists(pinn_path):
        try:
            checkpoint = torch.load(pinn_path, map_location='cpu', weights_only=False)

            hidden_dims = checkpoint.get('hidden_dims', config.pinn_hidden_dims)
            activation = checkpoint.get('activation', config.pinn_activation)
            im = np.array(checkpoint.get('input_mean', config.input_mean or [0]*5), dtype=np.float32)
            ist = np.array(checkpoint.get('input_std', config.input_std or [1]*5), dtype=np.float32)
            om = checkpoint.get('output_mean', config.output_mean)
            ost = checkpoint.get('output_std', config.output_std)

            pinn_model = PINN(
                hidden_dims=hidden_dims, activation=activation,
                input_mean=im, input_std=ist,
                output_mean=om, output_std=ost
            )
            pinn_model.load_state_dict(checkpoint['model_state_dict'])
            pinn_model.eval()

            n_params = sum(p.numel() for p in pinn_model.parameters())
            logger.info(f"Loaded PINN model from {pinn_path} ({n_params:,} parameters)")
        except Exception as e:
            logger.error(f"Error loading PINN model: {e}")
            traceback.print_exc()
    else:
        logger.warning(f"PINN model not found at {pinn_path}")

    # ---- PPO hedging agent ----
    hedging_path = "ppo_hedging_agent.zip"
    if os.path.exists(hedging_path):
        try:
            dummy_env = HedgingEnvironment(
                S0=config.hedging_S0, K=config.hedging_K, T=config.hedging_T,
                r=config.hedging_r, sigma=config.hedging_sigma,
                n_steps=config.hedging_n_steps
            )
            hedging_agent = PPO.load(hedging_path, env=dummy_env, device='cpu')
            logger.info(f"Loaded PPO hedging agent from {hedging_path}")
        except Exception as e:
            logger.error(f"Error loading hedging agent: {e}")
            traceback.print_exc()
    else:
        logger.warning(f"Hedging agent not found at {hedging_path}")

    # ---- Options data CSV ----
    data_path = "options_data.csv"
    if os.path.exists(data_path):
        try:
            options_data = pd.read_csv(data_path)
            logger.info(f"Loaded {len(options_data)} options from {data_path}")
        except Exception as e:
            logger.error(f"Error loading options data: {e}")
    else:
        logger.warning(f"Options data not found at {data_path}")

    return pinn_model, hedging_agent, training_metrics, options_data

# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

class OptionsVisualizer:
    """Plotly visualization factory for options analysis."""

    @staticmethod
    def plot_training_history(loss_history: List[Dict]) -> go.Figure:
        """Plot PINN training loss curves and physics weight schedule."""
        df = pd.DataFrame(loss_history)
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=('Training and Validation Loss', 'Loss Components'),
                            vertical_spacing=0.15)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['train_mse'], name='Train MSE',
                                 line=dict(color='#2563EB')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_loss'], name='Val MSE',
                                 line=dict(color='#DC2626')), row=1, col=1)
        if 'train_physics' in df.columns:
            fig.add_trace(go.Scatter(x=df['epoch'], y=df['train_physics'], name='Physics Loss',
                                     line=dict(color='#059669')), row=2, col=1)
        if 'physics_weight' in df.columns:
            fig.add_trace(go.Scatter(x=df['epoch'], y=df['physics_weight'], name='Physics Weight',
                                     line=dict(color='#D97706', dash='dot')), row=2, col=1)
        fig.update_yaxes(title_text="MSE", type='log', row=1, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_layout(height=600, showlegend=True, title_text="PINN Training Progress")
        return fig

    @staticmethod
    def plot_volatility_surface(M_grid, T_grid, IV_grid) -> go.Figure:
        """Plot 3D implied volatility surface."""
        fig = go.Figure(data=[go.Surface(
            x=M_grid, y=T_grid, z=IV_grid * 100,
            colorscale='Viridis', colorbar=dict(title='IV (%)')
        )])
        fig.update_layout(
            title='Implied Volatility Surface',
            scene=dict(xaxis_title='Moneyness (K/S)',
                       yaxis_title='Time to Maturity (years)',
                       zaxis_title='Implied Volatility (%)'),
            height=600, width=1000
        )
        return fig

    @staticmethod
    def plot_american_paths(paths, time_grid, K) -> go.Figure:
        """Plot sample Monte Carlo paths for American option pricing."""
        fig = go.Figure()
        for i in range(min(30, len(paths))):
            fig.add_trace(go.Scatter(x=time_grid, y=paths[i], mode='lines',
                                     line=dict(color='#93C5FD', width=0.8),
                                     showlegend=False, opacity=0.5))
        fig.add_hline(y=K, line_dash="dash", line_color="#DC2626", annotation_text="Strike Price")
        fig.update_layout(title='American Option: Sample Monte Carlo Paths',
                          xaxis_title='Time (years)', yaxis_title='Stock Price ($)',
                          height=500, width=1000)
        return fig

    @staticmethod
    def plot_model_comparison(bs_prices, pinn_prices, market_prices) -> go.Figure:
        """Scatter plot comparing BS and PINN predictions against market prices."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=market_prices, y=bs_prices, mode='markers',
                                 name='Black-Scholes', marker=dict(color='#2563EB', size=5, opacity=0.6)))
        fig.add_trace(go.Scatter(x=market_prices, y=pinn_prices, mode='markers',
                                 name='PINN', marker=dict(color='#DC2626', size=5, opacity=0.6)))
        min_v = min(market_prices.min(), min(bs_prices.min(), pinn_prices.min()))
        max_v = max(market_prices.max(), max(bs_prices.max(), pinn_prices.max()))
        fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode='lines',
                                 name='Perfect Prediction', line=dict(color='black', dash='dash')))
        fig.update_layout(title='Model Comparison: Predicted vs Market Price',
                          xaxis_title='Market Price ($)', yaxis_title='Model Price ($)',
                          height=600, width=800)
        return fig

    @staticmethod
    def plot_hedging_pnl(rl_pnls, bs_pnls) -> go.Figure:
        """Histogram comparing RL and BS hedging PnL distributions."""
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=rl_pnls, name='RL Agent', opacity=0.7,
                                    marker_color='#2563EB', nbinsx=40))
        fig.add_trace(go.Histogram(x=bs_pnls, name='BS Delta', opacity=0.7,
                                    marker_color='#DC2626', nbinsx=40))
        fig.update_layout(barmode='overlay', title='Hedging PnL Distribution',
                          xaxis_title='PnL ($)', yaxis_title='Count',
                          height=500, width=900)
        return fig

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_gradio_interface():
    """Create the complete Gradio interface with 8 analysis tabs."""

    config = OptionsPricerConfig()
    pinn_model, hedging_agent, training_metrics, options_data = load_pretrained_models(config)
    bs_model = BlackScholesModel(r=config.risk_free_rate, q=config.dividend_yield)
    vol_surface = VolatilitySurface(config)
    american_pricer = AmericanOptionPricer(config)
    arb_detector = ArbitrageDetector(config)
    viz = OptionsVisualizer()

    if options_data is not None:
        try:
            vol_surface.fit(options_data)
        except Exception as e:
            logger.warning(f"Could not pre-fit volatility surface: {e}")

    with gr.Blocks(title="Neural Network Options Pricer", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # Neural Network Options Pricer with Greeks & Volatility Surface

        **Advanced options pricing using Physics-Informed Neural Networks (PINNs), autograd Greeks,
        volatility surface modeling, American options, deep hedging, and arbitrage detection.**

        *Author: Spencer Purdy | Trained on real AAPL options data | Powered by PyTorch*
        """)

        with gr.Tabs():

            # ================================================================
            # TAB 1: TRAINING SUMMARY
            # ================================================================
            with gr.TabItem("Training Summary"):
                gr.Markdown("### Pre-Trained Model Summary\n\n"
                            "Models were trained on real AAPL options data from yfinance on H100 GPU. "
                            "This tab displays the training results and loss curves.")
                with gr.Row():
                    with gr.Column():
                        summary_output = gr.Markdown("Click the button below to load the training summary.")
                    with gr.Column():
                        training_plot = gr.Plot(label="Training Loss Curves")
                load_summary_btn = gr.Button("Load Training Summary", variant="primary")

                def load_training_summary():
                    if training_metrics is None:
                        return "Training metrics file not found.", None
                    pm = training_metrics.get('pinn', {})
                    hm = training_metrics.get('hedging', {})
                    hp = hm.get('hedging_params', {})

                    md = "### Training Configuration\n\n"
                    md += f"- **Data Source:** {training_metrics.get('data_source', 'N/A')}\n"
                    md += f"- **Symbol:** {training_metrics.get('symbol', 'N/A')}\n"
                    md += f"- **Stock Price:** ${training_metrics.get('stock_price', 0):.2f}\n"
                    md += f"- **Historical Volatility:** {training_metrics.get('historical_vol', 0):.2%}\n"
                    md += f"- **Dividend Yield:** {training_metrics.get('dividend_yield', 0):.2%}\n"
                    md += f"- **Total Options:** {training_metrics.get('n_options_total', 0)}\n"
                    n_calls = training_metrics.get('n_calls', 0)
                    n_puts = training_metrics.get('n_puts', 0)
                    md += f"- **Calls / Puts:** {n_calls} / {n_puts}\n"
                    md += f"- **Train / Test (calls):** {training_metrics.get('n_train', 0)} / {training_metrics.get('n_test', 0)}\n"
                    n_aug = training_metrics.get('n_augmented', 0)
                    if n_aug > 0:
                        md += f"- **Augmented Data Points:** {n_aug}\n"
                    md += f"- **GPU:** {training_metrics.get('gpu_name', 'N/A')}\n\n"

                    md += "### PINN Model Results\n\n"
                    md += f"- **Epochs Trained:** {pm.get('epochs_trained', 0)}\n"
                    md += f"- **Training Time:** {pm.get('training_time_seconds', 0):.1f}s\n"
                    md += f"- **Best Validation MSE:** {pm.get('best_val_loss', 0):.6f}\n"
                    md += f"- **PINN RMSE:** ${pm.get('pinn_rmse', 0):.4f}\n"
                    md += f"- **Black-Scholes RMSE:** ${pm.get('bs_rmse', 0):.4f}\n"
                    md += f"- **R-squared:** {pm.get('r_squared', 0):.6f}\n"
                    md += f"- **Correlation:** {pm.get('correlation', 0):.6f}\n"
                    md += f"- **MSE vs Black-Scholes:** {pm.get('improvement_mse_pct', 0):.2f}%\n\n"

                    gc = pm.get('greeks_check', {})
                    if gc:
                        gp = gc.get('pinn', {})
                        gb = gc.get('bs', {})
                        params = gc.get('params', {})
                        md += "### Greeks Validation (Near-ATM Call)\n\n"
                        md += f"- **Parameters:** S=${params.get('S', 0):.2f}, K=${params.get('K', 0):.0f}, "
                        md += f"T={params.get('T', 0)}, sigma={params.get('sigma', 0):.4f}\n"
                        md += f"- **PINN Delta:** {gp.get('delta', 0):.6f} vs BS Delta: {gb.get('delta', 0):.6f} "
                        md += f"(error: {abs(gp.get('delta', 0) - gb.get('delta', 0)):.6f})\n"
                        md += f"- **PINN Gamma:** {gp.get('gamma', 0):.6f} vs BS Gamma: {gb.get('gamma', 0):.6f}\n"
                        md += f"- **PINN Vega:** {gp.get('vega', 0):.6f} vs BS Vega: {gb.get('vega', 0):.6f}\n\n"

                    md += "### Deep Hedging Results\n\n"
                    md += f"- **Training Time:** {hm.get('training_time_seconds', 0):.1f}s\n"
                    md += f"- **Parameters:** S0=${hp.get('S0', 0):.2f}, K=${hp.get('K', 0):.0f}, "
                    md += f"T={hp.get('T', 0)}, sigma={hp.get('sigma', 0):.2%}\n"
                    md += f"- **RL Agent:** Mean PnL=${hm.get('rl_mean_pnl', 0):.4f}, "
                    md += f"Std=${hm.get('rl_std_pnl', 0):.4f}, Sharpe={hm.get('rl_sharpe', 0):.2f}\n"
                    md += f"- **BS Baseline:** Mean PnL=${hm.get('bs_mean_pnl', 0):.4f}, "
                    md += f"Std=${hm.get('bs_std_pnl', 0):.4f}, Sharpe={hm.get('bs_sharpe', 0):.2f}\n"

                    loss_hist = pm.get('loss_history', [])
                    plot = viz.plot_training_history(loss_hist) if loss_hist else None
                    return md, plot

                load_summary_btn.click(fn=load_training_summary, outputs=[summary_output, training_plot])

            # ================================================================
            # TAB 2: GREEKS CALCULATOR
            # ================================================================
            with gr.TabItem("Greeks Calculator"):
                gr.Markdown("### Real-Time Greeks Calculation\n\n"
                            "Calculate option Greeks using both Black-Scholes (analytical) and PINN "
                            "(automatic differentiation via PyTorch autograd).")
                with gr.Row():
                    with gr.Column():
                        greeks_S = gr.Number(label="Stock Price (S)", value=round(config.stock_price, 2))
                        greeks_K = gr.Number(label="Strike Price (K)", value=round(config.stock_price / 5) * 5)
                        greeks_T = gr.Number(label="Time to Maturity (years)", value=0.5)
                        greeks_sigma = gr.Number(label="Volatility", value=round(config.historical_vol, 2))
                        greeks_r = gr.Number(label="Risk-Free Rate", value=config.risk_free_rate)
                        greeks_type = gr.Radio(["call", "put"], label="Option Type", value="call")
                        calc_greeks_btn = gr.Button("Calculate Greeks", variant="primary")
                    with gr.Column():
                        greeks_output = gr.Markdown("Enter parameters and click Calculate Greeks.")

                def calculate_greeks(S, K, T, sigma, r, opt_type):
                    try:
                        md = "### Greeks Calculation Results\n\n"
                        if opt_type == 'call':
                            bs_price = bs_model.call_price(S, K, T, sigma)
                        else:
                            bs_price = bs_model.put_price(S, K, T, sigma)
                        bs_delta = bs_model.delta(S, K, T, sigma, opt_type)
                        bs_gamma = bs_model.gamma(S, K, T, sigma)
                        bs_vega = bs_model.vega(S, K, T, sigma)
                        bs_theta = bs_model.theta(S, K, T, sigma, opt_type)
                        bs_rho = bs_model.rho(S, K, T, sigma, opt_type)

                        md += "#### Black-Scholes Model (Analytical)\n\n"
                        md += f"- **Price:** ${bs_price:.4f}\n"
                        md += f"- **Delta:** {bs_delta:.6f}\n"
                        md += f"- **Gamma:** {bs_gamma:.6f}\n"
                        md += f"- **Vega:** {bs_vega:.6f}\n"
                        md += f"- **Theta:** {bs_theta:.6f}\n"
                        md += f"- **Rho:** {bs_rho:.6f}\n\n"

                        if pinn_model is not None:
                            pg = pinn_model.compute_greeks(S, K, T, sigma, r)
                            md += "#### PINN Model (Automatic Differentiation)\n\n"
                            md += f"- **Price:** ${pg['price']:.4f}\n"
                            md += f"- **Delta:** {pg['delta']:.6f}\n"
                            md += f"- **Gamma:** {pg['gamma']:.6f}\n"
                            md += f"- **Vega:** {pg['vega']:.6f}\n"
                            md += f"- **Theta:** {pg['theta']:.6f}\n"
                            md += f"- **Rho:** {pg['rho']:.6f}\n\n"
                            md += "#### Comparison\n\n"
                            price_diff = abs(pg['price'] - bs_price)
                            md += f"- **Price Difference:** ${price_diff:.4f}"
                            if bs_price > 0:
                                md += f" ({price_diff/bs_price*100:.2f}%)"
                            md += "\n"
                            md += f"- **Delta Difference:** {abs(pg['delta'] - bs_delta):.6f}\n"
                        else:
                            md += "PINN model not loaded. Only Black-Scholes results shown.\n"
                        return md
                    except Exception as e:
                        return f"### Error\n\n```\n{str(e)}\n{traceback.format_exc()}\n```"

                calc_greeks_btn.click(fn=calculate_greeks,
                                     inputs=[greeks_S, greeks_K, greeks_T, greeks_sigma, greeks_r, greeks_type],
                                     outputs=greeks_output)

            # ================================================================
            # TAB 3: VOLATILITY SURFACE
            # ================================================================
            with gr.TabItem("Volatility Surface"):
                gr.Markdown("### Implied Volatility Surface\n\n"
                            "3D visualization of the implied volatility surface fitted from real "
                            "AAPL options data using cubic/linear interpolation.")
                fit_surface_btn = gr.Button("Fit and Visualize Surface", variant="primary", size="lg")
                surface_plot = gr.Plot(label="3D Volatility Surface")
                surface_output = gr.Markdown("Click button to generate the volatility surface.")

                def fit_and_plot_surface():
                    try:
                        if options_data is None:
                            return None, "No options data loaded."
                        vol_surface.fit(options_data)
                        M_grid, T_grid, IV_grid = vol_surface.create_surface_grid()
                        if M_grid is None:
                            return None, "Insufficient data points."
                        plot = viz.plot_volatility_surface(M_grid, T_grid, IV_grid)
                        md = "### Volatility Surface Fitted\n\n"
                        md += f"- **Data Points:** {len(vol_surface.surface_data)}\n"
                        md += f"- **Moneyness Range:** {M_grid.min():.3f} -- {M_grid.max():.3f}\n"
                        md += f"- **Maturity Range:** {T_grid.min():.3f} -- {T_grid.max():.3f} years\n"
                        md += f"- **IV Range:** {IV_grid.min()*100:.1f}% -- {IV_grid.max()*100:.1f}%\n"
                        return plot, md
                    except Exception as e:
                        return None, f"### Error\n\n```\n{str(e)}\n{traceback.format_exc()}\n```"

                fit_surface_btn.click(fn=fit_and_plot_surface, outputs=[surface_plot, surface_output])

            # ================================================================
            # TAB 4: AMERICAN OPTIONS
            # ================================================================
            with gr.TabItem("American Options"):
                gr.Markdown("### American Options Pricing\n\n"
                            "Price American put options using the Longstaff-Schwartz Least Squares "
                            "Monte Carlo method with Laguerre polynomial basis functions.")
                with gr.Row():
                    with gr.Column():
                        am_S = gr.Number(label="Stock Price (S)", value=round(config.stock_price, 2))
                        am_K = gr.Number(label="Strike Price (K)", value=round(config.stock_price / 5) * 5)
                        am_T = gr.Number(label="Time to Maturity (years)", value=1.0)
                        am_sigma = gr.Number(label="Volatility", value=round(config.historical_vol, 2))
                        am_r = gr.Number(label="Risk-Free Rate", value=config.risk_free_rate)
                        am_btn = gr.Button("Price American Put", variant="primary")
                    with gr.Column():
                        am_output = gr.Markdown("Enter parameters and click Price American Put.")
                am_plot = gr.Plot(label="Sample Paths")

                def price_american(S, K, T, sigma, r):
                    try:
                        result = american_pricer.price_american_put(S, K, T, r, sigma)
                        european_price = bs_model.put_price(S, K, T, sigma)
                        premium = result['price'] - european_price
                        md = "### American Put Option Results\n\n"
                        md += f"- **American Put Price:** ${result['price']:.4f}\n"
                        md += f"- **European Put Price:** ${european_price:.4f}\n"
                        md += f"- **Early Exercise Premium:** ${premium:.4f}"
                        if european_price > 0:
                            md += f" ({premium/european_price*100:.2f}%)"
                        md += "\n\n#### Simulation Parameters\n\n"
                        md += f"- Monte Carlo paths: {american_pricer.config.american_simulation_paths:,}\n"
                        md += f"- Time steps: {american_pricer.config.american_time_steps}\n"
                        md += f"- LSM basis functions: {american_pricer.config.lsm_basis_functions}\n"
                        plot = viz.plot_american_paths(result['paths'], result['time_grid'], K)
                        return md, plot
                    except Exception as e:
                        return f"### Error\n\n```\n{str(e)}\n{traceback.format_exc()}\n```", None

                am_btn.click(fn=price_american, inputs=[am_S, am_K, am_T, am_sigma, am_r],
                             outputs=[am_output, am_plot])

            # ================================================================
            # TAB 5: DEEP HEDGING
            # ================================================================
            with gr.TabItem("Deep Hedging"):
                gr.Markdown("### Deep Reinforcement Learning for Hedging\n\n"
                            "Run the pre-trained PPO agent on simulated hedging episodes and compare "
                            "against the Black-Scholes delta hedging baseline.")
                with gr.Row():
                    with gr.Column():
                        hedge_episodes = gr.Slider(label="Number of Episodes",
                                                   minimum=10, maximum=500, value=100, step=10)
                        hedge_btn = gr.Button("Run Hedging Simulation", variant="primary")
                    with gr.Column():
                        hedge_output = gr.Markdown("Click Run Hedging Simulation to begin.")
                hedge_plot = gr.Plot(label="PnL Distribution")

                def run_hedging_simulation(n_episodes):
                    try:
                        if hedging_agent is None:
                            return "Hedging agent not loaded.", None
                        n_episodes = int(n_episodes)
                        env = HedgingEnvironment(
                            S0=config.hedging_S0, K=config.hedging_K, T=config.hedging_T,
                            r=config.hedging_r, sigma=config.hedging_sigma,
                            n_steps=config.hedging_n_steps
                        )

                        # RL agent evaluation
                        rl_pnls = []
                        for _ in range(n_episodes):
                            obs, _ = env.reset()
                            done = False
                            while not done:
                                action, _ = hedging_agent.predict(obs, deterministic=True)
                                obs, _, done, _, info = env.step(action)
                            rl_pnls.append(info['pnl'])

                        # Black-Scholes delta baseline using normalized observation
                        bs_calc = BlackScholesModel(r=config.hedging_r)
                        bs_pnls = []
                        for _ in range(n_episodes):
                            obs, _ = env.reset()
                            done = False
                            while not done:
                                S_cur = obs[0] * config.hedging_S0
                                ttm = obs[1] * config.hedging_T
                                if ttm > 0.001:
                                    target_delta = bs_calc.delta(S_cur, config.hedging_K, ttm,
                                                                 config.hedging_sigma, 'call')
                                else:
                                    target_delta = 1.0 if S_cur > config.hedging_K else 0.0
                                action = np.array([target_delta / 0.75 - 1.0], dtype=np.float32)
                                action = np.clip(action, -1.0, 1.0)
                                obs, _, done, _, info = env.step(action)
                            bs_pnls.append(info['pnl'])

                        rl_arr = np.array(rl_pnls)
                        bs_arr = np.array(bs_pnls)
                        ann_factor = np.sqrt(252 / config.hedging_T)
                        rl_sharpe = np.mean(rl_arr) / (np.std(rl_arr) + 1e-8) * ann_factor
                        bs_sharpe = np.mean(bs_arr) / (np.std(bs_arr) + 1e-8) * ann_factor

                        md = f"### Hedging Simulation Results ({n_episodes} episodes)\n\n"
                        md += f"**Option:** {config.symbol} Call, S0=${config.hedging_S0:.2f}, "
                        md += f"K=${config.hedging_K:.0f}, T={config.hedging_T}, sigma={config.hedging_sigma:.2%}\n\n"
                        md += "#### RL Agent (PPO)\n\n"
                        md += f"- **Mean PnL:** ${np.mean(rl_arr):.4f}\n"
                        md += f"- **Std PnL:** ${np.std(rl_arr):.4f}\n"
                        md += f"- **Sharpe Ratio:** {rl_sharpe:.2f}\n"
                        md += f"- **Range:** [${np.min(rl_arr):.4f}, ${np.max(rl_arr):.4f}]\n\n"
                        md += "#### Black-Scholes Delta Hedge\n\n"
                        md += f"- **Mean PnL:** ${np.mean(bs_arr):.4f}\n"
                        md += f"- **Std PnL:** ${np.std(bs_arr):.4f}\n"
                        md += f"- **Sharpe Ratio:** {bs_sharpe:.2f}\n"
                        md += f"- **Range:** [${np.min(bs_arr):.4f}, ${np.max(bs_arr):.4f}]\n"
                        plot = viz.plot_hedging_pnl(rl_pnls, bs_pnls)
                        return md, plot
                    except Exception as e:
                        return f"### Error\n\n```\n{str(e)}\n{traceback.format_exc()}\n```", None

                hedge_btn.click(fn=run_hedging_simulation, inputs=[hedge_episodes],
                                outputs=[hedge_output, hedge_plot])

            # ================================================================
            # TAB 6: ARBITRAGE DETECTION
            # ================================================================
            with gr.TabItem("Arbitrage Detection"):
                gr.Markdown("### Arbitrage Opportunity Scanner\n\n"
                            "Scan the real options chain for violations of put-call parity "
                            "(dividend-adjusted) and butterfly spread convexity conditions. "
                            "A minimum profit threshold of $2.00 filters out violations that "
                            "would vanish after transaction costs and bid-ask spreads.")
                scan_btn = gr.Button("Scan for Arbitrage", variant="primary", size="lg")
                arb_output = gr.Markdown("Click Scan for Arbitrage to begin.")

                def scan_arbitrage():
                    try:
                        if options_data is None:
                            return "No options data loaded."
                        arbitrages = arb_detector.scan_chain(options_data)
                        if not arbitrages:
                            md = "### Arbitrage Scan Results\n\n"
                            md += "**No arbitrage opportunities detected.**\n\n"
                            md += (f"The options chain appears to be arbitrage-free within the "
                                   f"${config.arbitrage_min_profit:.2f} minimum profit threshold "
                                   f"(accounts for transaction costs and bid-ask spreads).\n")
                        else:
                            md = f"### Arbitrage Scan Results\n\n"
                            md += f"**Found {len(arbitrages)} potential arbitrage opportunities:**\n\n"
                            for i, arb in enumerate(arbitrages[:15], 1):
                                md += f"#### Opportunity {i}: {arb['type']}\n\n"
                                if 'strike' in arb:
                                    md += f"- **Strike:** ${arb['strike']:.2f}\n"
                                if 'strikes' in arb:
                                    md += f"- **Strikes:** {arb['strikes']}\n"
                                md += f"- **Maturity:** {arb['maturity']:.4f} years\n"
                                md += f"- **Amount:** ${arb['arbitrage_amount']:.4f}\n"
                                md += f"- **Strategy:** {arb['strategy']}\n\n"
                            if len(arbitrages) > 15:
                                md += f"*... and {len(arbitrages) - 15} more*\n\n"
                        md += "#### Note\n\n"
                        md += ("These are theoretical arbitrage opportunities identified using "
                               "dividend-adjusted put-call parity and butterfly spread convexity. "
                               "In practice, execution risk, margin requirements, and timing "
                               "differences may reduce or eliminate the apparent profit.\n")
                        return md
                    except Exception as e:
                        return f"### Error\n\n```\n{str(e)}\n{traceback.format_exc()}\n```"

                scan_btn.click(fn=scan_arbitrage, outputs=arb_output)

            # ================================================================
            # TAB 7: MODEL COMPARISON
            # ================================================================
            with gr.TabItem("Model Comparison"):
                gr.Markdown("### Black-Scholes vs PINN Performance\n\n"
                            "Compare pricing accuracy of Black-Scholes and PINN models against "
                            "real market prices on the held-out test set.")
                compare_btn = gr.Button("Generate Comparison", variant="primary", size="lg")
                compare_plot = gr.Plot(label="Model Comparison")
                compare_output = gr.Markdown("Click Generate Comparison to begin.")

                def compare_models():
                    try:
                        if options_data is None or pinn_model is None:
                            return None, "Ensure both PINN model and options data are loaded."
                        calls = options_data[options_data['option_type'] == 'call']
                        shuffled = calls.sample(frac=1.0, random_state=RANDOM_SEED)
                        test_size = int(len(shuffled) * (1 - config.train_test_split))
                        test_df = shuffled.iloc[-test_size:]

                        market_prices = test_df['market_price'].values
                        bs_prices = test_df['bs_price'].values
                        pinn_prices = pinn_model.predict_batch(test_df)

                        bs_mse = np.mean((bs_prices - market_prices)**2)
                        pinn_mse = np.mean((pinn_prices - market_prices)**2)
                        bs_mae = np.mean(np.abs(bs_prices - market_prices))
                        pinn_mae = np.mean(np.abs(pinn_prices - market_prices))
                        improvement = (bs_mse - pinn_mse) / bs_mse * 100 if bs_mse > 0 else 0

                        ss_res = np.sum((market_prices - pinn_prices)**2)
                        ss_tot = np.sum((market_prices - np.mean(market_prices))**2)
                        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                        corr = np.corrcoef(pinn_prices, market_prices)[0, 1] if len(pinn_prices) > 1 else 0

                        md = f"### Model Comparison Results (Test Set: {len(test_df)} options)\n\n"
                        md += "#### Black-Scholes Model\n\n"
                        md += f"- **MSE:** {bs_mse:.6f}\n"
                        md += f"- **MAE:** ${bs_mae:.4f}\n"
                        md += f"- **RMSE:** ${np.sqrt(bs_mse):.4f}\n\n"
                        md += "#### PINN Model\n\n"
                        md += f"- **MSE:** {pinn_mse:.6f}\n"
                        md += f"- **MAE:** ${pinn_mae:.4f}\n"
                        md += f"- **RMSE:** ${np.sqrt(pinn_mse):.4f}\n"
                        md += f"- **R-squared:** {r2:.6f}\n"
                        md += f"- **Correlation:** {corr:.6f}\n\n"
                        md += f"#### MSE Difference: {improvement:.2f}%\n\n"
                        if improvement > 0:
                            md += f"PINN outperforms Black-Scholes by {improvement:.1f}% on MSE.\n"
                        else:
                            md += ("Black-Scholes achieves lower error on this dataset. This is expected "
                                   "when BS uses each option's own implied volatility as input, making it a "
                                   "near-perfect pricer by construction. The PINN learns a general pricing "
                                   "function across all strikes and maturities without per-option IV input.\n")
                        plot = viz.plot_model_comparison(bs_prices, pinn_prices, market_prices)
                        return plot, md
                    except Exception as e:
                        return None, f"### Error\n\n```\n{str(e)}\n{traceback.format_exc()}\n```"

                compare_btn.click(fn=compare_models, outputs=[compare_plot, compare_output])

            # ================================================================
            # TAB 8: DOCUMENTATION
            # ================================================================
            with gr.TabItem("Documentation"):
                gr.Markdown("""
                ## Neural Network Options Pricer -- Technical Documentation

                ### Overview

                This system implements advanced options pricing using Physics-Informed Neural
                Networks (PINNs) and deep reinforcement learning. It demonstrates expertise in
                derivatives pricing, quantitative finance, and machine learning engineering.

                ### Architecture

                **PINN Model:**

                The Physics-Informed Neural Network takes five inputs (stock price, strike, time to
                maturity, implied volatility, risk-free rate) and outputs a predicted option price.
                Input standardization and output scaling ensure stable training with Tanh activations.

                - Input: [S, K, T, sigma, r] (5 features, standardized)
                - Hidden layers: [128, 256, 256, 128] with Tanh
                - Output: Option price (destandardized to dollars)
                - Training loss: MSE(data) + physics_weight * MSE(PDE residual)

                The Black-Scholes PDE is enforced as a physics constraint via automatic differentiation:

                ```
                -dV/dT + 0.5 * sigma^2 * S^2 * d2V/dS2 + r * S * dV/dS - r * V = 0
                ```

                Training uses a curriculum schedule where the physics weight ramps from 0 to 0.01
                over a configurable epoch range, allowing the network to first learn the price
                surface from data before gradually enforcing the PDE.

                The model is trained on call options only, with stock-price-augmented data (S shifted
                by +/-5% to +/-20%) to ensure meaningful gradients in the S direction. This enables
                accurate Delta and Gamma computation via PyTorch autograd.

                **Deep Hedging Agent:**

                A PPO (Proximal Policy Optimization) agent learns to dynamically hedge a short call
                option position by adjusting shares of the underlying stock at each timestep. The
                observation includes normalized stock price (S/S0), time fraction remaining (ttm/T),
                current hedge position, and Black-Scholes delta as a learning hint. The action
                specifies a target hedge ratio in [0, 1.5]. The reward minimizes squared PnL
                deviation from zero, accounting for transaction costs.

                ### Data

                All models are trained on real AAPL options data fetched from yfinance, including
                calls and puts across multiple expiration dates. Options are filtered for liquidity
                (volume >= 5, open interest >= 10, bid-ask spread <= 50% of mid) and priced at the
                bid-ask midpoint.

                ### Key Features

                1. **Physics-Informed Neural Networks (PINNs):**
                   Combines data-driven learning with Black-Scholes PDE constraints for physically
                   consistent option pricing across the entire strike/maturity surface.

                2. **Automatic Differentiation for Greeks:**
                   Exact Greeks via PyTorch autograd (Delta, Gamma, Vega, Theta, Rho). Stock-price
                   augmented training data ensures accurate dV/dS gradients.

                3. **Implied Volatility Surface:**
                   3D visualization using cubic/linear interpolation across moneyness and maturity.
                   Captures the volatility smile and term structure from real market data.

                4. **American Options Pricing:**
                   Least Squares Monte Carlo (LSM) with Laguerre polynomial basis functions and
                   early exercise boundary estimation.

                5. **Deep Hedging (RL):**
                   PPO agent with normalized observations for dynamic delta hedging with proportional
                   transaction costs. Evaluated against the analytical Black-Scholes delta hedge.

                6. **Arbitrage Detection:**
                   Dividend-adjusted put-call parity scanning and butterfly spread convexity checks
                   with a minimum profit threshold to filter out false positives.

                ### Limitations

                1. PINN assumes the training data distribution; extrapolation outside the observed
                   strike/maturity range may be unreliable.

                2. The hedging agent is trained on a single option contract (near-ATM, 3-month).
                   Performance may differ for other configurations.

                3. Arbitrage signals may be false positives due to execution risk and timing.

                4. Transaction costs are simplified (no market impact modeling).

                5. American option pricing uses Monte Carlo which introduces simulation noise.

                ### References

                1. Raissi et al. (2019) -- "Physics-informed neural networks"
                2. Black & Scholes (1973) -- "The Pricing of Options and Corporate Liabilities"
                3. Longstaff & Schwartz (2001) -- "Valuing American Options by Simulation"
                4. Buehler et al. (2019) -- "Deep Hedging"
                5. Schulman et al. (2017) -- "Proximal Policy Optimization Algorithms"

                ### Author

                **Spencer Purdy** -- Portfolio demonstration project showcasing derivatives pricing,
                physics-informed machine learning, and quantitative finance expertise.

                ### Disclaimer

                This is a simulation system for educational and portfolio demonstration purposes.
                Real options trading involves significant risks. Past performance does not guarantee
                future results. Always consult with financial professionals before trading options
                with real capital.
                """)

        gr.Markdown("""
        ---
        **Spencer Purdy | Neural Network Options Pricer v2.0**

        *Built with PyTorch, Stable-Baselines3, and Gradio | Trained on real AAPL options data*
        """)

    return interface

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Neural Network Options Pricer with Greeks & Volatility Surface")
    logger.info("Author: Spencer Purdy")
    logger.info("=" * 80)

    logger.info("Creating Gradio interface...")
    interface = create_gradio_interface()

    logger.info("Launching application...")
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )