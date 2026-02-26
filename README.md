# Neural Network Options Pricer with Greeks and Volatility Surface

A production-grade options pricing system combining Physics-Informed Neural Networks (PINNs), automatic differentiation Greeks, implied volatility surface modeling, American options pricing via Least Squares Monte Carlo, and deep reinforcement learning for optimal delta hedging. All models are trained on real AAPL options data from yfinance.

## About

This portfolio project demonstrates derivatives pricing engineering by implementing a PINN that enforces the Black-Scholes PDE as a physics constraint during training, alongside a PPO agent that learns to dynamically hedge short call positions. The PINN achieves 99.46% lower MSE than Black-Scholes baseline pricing on the test set, with R² of 0.99994 against market prices.

**Author:** Spencer Purdy  
**Development Environment:** Google Colab Pro (H100 GPU, High RAM)

## Features

- **Physics-Informed Neural Network (PINN):** Enforces the Black-Scholes PDE via automatic differentiation loss with curriculum-scheduled physics weight ramp (epoch 300→800)
- **Exact Greeks via Autograd:** Delta, Gamma, Vega, Theta, and Rho computed via PyTorch autograd on the PINN; stock-price-augmented training data (±5% to ±20% shifts) ensures accurate dV/dS gradients
- **Implied Volatility Surface:** 3D visualization using RBF interpolation across moneyness and maturity; captures the volatility smile and term structure from real market data
- **American Options Pricing:** Least Squares Monte Carlo (LSM) with Laguerre polynomial basis functions, 10,000 simulation paths, 100 time steps, and early exercise boundary estimation
- **Deep Hedging (PPO):** Reinforcement learning agent learns to minimize squared PnL deviation when hedging a short call, trained for 3,000,000 timesteps with proportional transaction costs
- **Arbitrage Detection:** Dividend-adjusted put-call parity scanning and butterfly spread convexity checks with a $2.00 minimum profit threshold to filter false positives
- **Model Comparison:** Live side-by-side benchmarking of PINN vs. Black-Scholes on the held-out test set
- **Interactive Interface:** Gradio web application with 8 analysis tabs

## Dataset

- **Source:** Real AAPL options data via yfinance
- **Training Date:** February 25, 2026 | AAPL Stock Price: $272.14
- **Total Options:** 1,130 (658 calls, 472 puts)
- **Liquidity Filter:** Volume ≥ 5, open interest ≥ 10, bid-ask spread ≤ 50% of mid
- **Pricing:** Bid-ask midpoint
- **Train / Test Split:** 526 / 132 options
- **Augmented Training Samples:** 4,120 (stock-price-shifted copies at ±5%, ±10%, ±15%, ±20%)
- **Historical Volatility:** 32.20% | **Dividend Yield:** 0.38%

## PINN Performance

Performance evaluated on held-out test set (132 options):

| Metric | PINN | Black-Scholes |
|--------|------|--------------|
| MSE | **0.166** | 30.649 |
| MAE | **$0.289** | $3.478 |
| RMSE | **$0.408** | $5.536 |
| R² | **0.99994** | — |
| Correlation | **0.99997** | — |
| MSE Improvement vs. BS | **99.46%** | — |

**Note:** Black-Scholes here uses a single historical volatility input rather than each option's own implied volatility, making the comparison a meaningful test of the PINN's ability to learn the full price surface across all strikes and maturities.

**Training:**
- Epochs: 2,059 (early stopping from 3,000 max, patience 300)
- Training Time: 76.7 seconds
- Best Validation Loss: 0.1661 (epoch 2,058)
- Initial Val Loss (epoch 0): 1,004.76 → Final: 0.1661

## PINN Greeks Sample

Greeks computed via PyTorch autograd on a near-ATM call (S=$272.14, K=$270, T=0.5yr, σ=32.2%, r=5%):

| Greek | PINN Value |
|-------|-----------|
| Price | $25.31 |
| Delta | 0.457 |
| Gamma | 0.118 |
| Vega | 0.157 |
| Theta | -0.069 |

## Deep Hedging Agent Performance

PPO agent evaluated on 1,000 episodes hedging a near-ATM short call (S₀=$272.14, K=$270, T=0.25yr, σ=32.2%, n_steps=63):

| Metric | PPO Agent | BS Delta Hedge |
|--------|-----------|---------------|
| Mean PnL | $1.601 | $1.708 |
| Std PnL | $2.789 | $2.180 |
| Sharpe Ratio | 18.23 | 24.88 |
| Min PnL | -$7.559 | -$6.780 |
| Max PnL | $8.805 | $9.775 |

The BS delta hedge outperforms on Sharpe ratio and volatility of PnL. The PPO agent's reward trajectory shows steady improvement from -0.235 at step 100k to -0.003 at step 3,000,000, indicating consistent learning throughout training.

**PPO Training:** 3,000,000 timesteps | ~53.6 minutes | NVIDIA H100 80GB HBM3

## PINN Architecture

| Component | Details |
|-----------|---------|
| Input | [S, K, T, σ, r] — 5 features, standardized |
| Hidden Layers | [128, 256, 256, 128] with Tanh activation |
| Output | Option price (destandardized to dollars) |
| Loss | MSE(data) + physics_weight × MSE(PDE residual) |
| Physics Weight | Ramp 0.0 → 0.01 over epochs 300–800 |
| Physics Loss Cap | 50.0 (prevents PDE loss dominating early) |
| Augmentation Weight | 0.3 (price-shifted copies) |
| Optimizer | Adam, lr=3e-4, weight decay=1e-5 |
| Batch Size | 128 |
| Max Epochs | 3,000 (early stopping, patience=300) |

**Black-Scholes PDE constraint enforced during training:**
```
-dV/dT + 0.5 × σ² × S² × d²V/dS² + r × S × dV/dS - r × V = 0
```

## Deep Hedging Agent Architecture

| Component | Details |
|-----------|---------|
| Algorithm | Proximal Policy Optimization (PPO) |
| Observation | [S/S₀, ttm/T, current hedge position, BS delta] |
| Action Space | Target hedge ratio ∈ [0, 1.5] (continuous) |
| Reward | Minimize squared PnL deviation from zero, net of transaction costs |
| Training Timesteps | 3,000,000 |
| Learning Rate | 3e-4 |
| Gamma | 0.99 |
| Episode Length | 63 steps (one trading quarter) |

## Technical Stack

- **Deep Learning:** PyTorch (PINN, autograd Greeks)
- **Deep RL:** Stable-Baselines3 (PPO hedging agent), Gymnasium
- **Optimization:** scipy (IV solver — Newton-Raphson, max 100 iterations)
- **Monte Carlo:** numpy (LSM American options, 10,000 paths)
- **Market Data:** yfinance (real AAPL options chain)
- **Interpolation:** scipy.interpolate (RBF/griddata for volatility surface)
- **UI Framework:** Gradio
- **Visualization:** Plotly (3D volatility surface, Greeks surface, PnL distributions)
- **Development:** Google Colab Pro with H100 GPU

## Setup and Usage

### Running in Google Colab

1. Clone this repository or download the notebook file
2. Upload to Google Colab
3. Select Runtime > Change runtime type > H100 GPU (or A100/T4 for free tier)
4. Run all cells sequentially

The notebook will automatically:
- Install required dependencies
- Fetch real AAPL options data from yfinance
- Apply liquidity filters and augment training data
- Train the PINN with curriculum PDE scheduling
- Train the PPO hedging agent
- Evaluate both models
- Launch a Gradio interface with a shareable link

### Running Locally

```bash
# Clone the repository
git clone https://github.com/SpencerCPurdy/Neural_Network_Options_Pricer.git
cd Neural_Network_Options_Pricer

# Install dependencies
pip install torch>=2.0.0 stable-baselines3[extra]>=2.1.0 gymnasium>=0.29.0 pandas>=2.0.0 numpy>=1.24.0 scipy>=1.10.0 scikit-learn>=1.3.0 plotly>=5.15.0 gradio>=4.0.0

# Run the application
python app.py
```

**Note:** Pre-trained model weights (`pinn_model.pth`, `ppo_hedging_agent.zip`) and `options_data.csv` must be present in the repository root. PINN training takes ~77 seconds on an H100; PPO training takes ~54 minutes.

## Project Structure

```
├── app.py
├── pinn_model.pth
├── ppo_hedging_agent.zip
├── options_data.csv
├── training_metrics.json
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

The application contains the following components:

1. **Configuration & Setup**: `OptionsPricerConfig` dataclass with all hyperparameters; normalization statistics loaded from checkpoint at startup
2. **Data Pipeline**: yfinance options fetch with liquidity filtering, bid-ask midpoint pricing, and stock-price augmentation for Greeks accuracy
3. **PINN Model**: 4-layer MLP with tanh, input/output normalization, Black-Scholes PDE loss via autograd
4. **Greeks Engine**: PyTorch autograd differentiation through the PINN for Delta, Gamma, Vega, Theta, Rho
5. **Implied Volatility Solver**: Newton-Raphson iteration against Black-Scholes formula
6. **Volatility Surface**: RBF interpolation across moneyness and maturity with 3D visualization
7. **American Options Engine**: LSM with Laguerre polynomial basis, early exercise boundary detection
8. **PPO Hedging Agent**: Stable-Baselines3 PPO with normalized observations and transaction cost reward
9. **Arbitrage Scanner**: Put-call parity (dividend-adjusted) and butterfly spread convexity checks
10. **Gradio Interface**: 8-tab interactive application

## Key Implementation Details

- **Reproducibility:** All random seeds fixed to 42 across Python, NumPy, and PyTorch; deterministic CUDA mode enabled
- **Curriculum Training:** Physics PDE weight ramps 0.0→0.01 over epochs 300–800; PDE loss capped at 50.0 to prevent early training collapse
- **Greeks Augmentation:** Training data augmented with S shifted ±5%/±10%/±15%/±20% to ensure accurate dV/dS gradients via autograd
- **IV Surface Grid:** 25 strike levels × 15 maturity levels with cubic/linear interpolation fallback
- **Arbitrage Threshold:** $2.00 minimum profit to flag violations; filters sub-threshold signals that vanish after transaction costs and bid-ask spreads
- **LSM Basis Functions:** 5 Laguerre polynomial terms for early exercise regression

## Limitations and Known Issues

### Model Limitations
- PINN is trained on call options only; put pricing relies on put-call parity
- Extrapolation outside the observed strike/maturity range during training may be unreliable
- The PPO hedging agent is trained on a single near-ATM 3-month contract; performance on different strikes, maturities, or underlyings is not validated
- LSM American option pricing introduces Monte Carlo simulation noise

### Financial Model Limitations
- European-style assumption for the PINN; American exercise handled separately via LSM
- Transaction costs are simplified — no market impact or bid-ask spread modeling
- Volatility surface assumes smooth interpolation; discrete jumps or calendar spread arbitrage not enforced
- Model limited to equity options; no exotic derivatives or multi-asset structures

### Operational Limitations
- Inference runs on CPU only; FinBERT is not used in this project
- Real-time arbitrage signals are indicative only — execution risk, timing, and liquidity constraints are not modeled
- yfinance data quality depends on market hours and exchange reporting

### Known Failure Modes
- PINN accuracy degrades for deep OTM options with very low market prices
- Volatility surface interpolation can produce artifacts in sparse strike/maturity regions
- Arbitrage scanner may produce false positives during low-liquidity or end-of-day data

## Normalization Statistics (Training Data)

| Feature | Mean | Std |
|---------|------|-----|
| Stock Price (S) | 272.14 | 35.09 |
| Strike (K) | 274.99 | 87.00 |
| Time to Maturity (T) | 0.699 | 0.703 |
| Implied Volatility (σ) | 0.381 | 0.165 |
| Risk-Free Rate (r) | 0.050 | ~0.0 |
| **Output (Price)** | **$46.11** | **$53.77** |

## References

1. Raissi, Perdikaris & Karniadakis (2019) — *Physics-Informed Neural Networks*
2. Black & Scholes (1973) — *The Pricing of Options and Corporate Liabilities*
3. Longstaff & Schwartz (2001) — *Valuing American Options by Simulation*
4. Buehler et al. (2019) — *Deep Hedging*
5. Schulman et al. (2017) — *Proximal Policy Optimization Algorithms*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Spencer Purdy**  
GitHub: [@SpencerCPurdy](https://github.com/SpencerCPurdy)

---

*This is a portfolio project developed to demonstrate derivatives pricing, physics-informed machine learning, and quantitative finance engineering. The system is designed for educational and demonstrational purposes only. Real options trading involves significant financial risk and regulatory requirements. Past performance does not guarantee future results. Always consult with licensed financial professionals before trading options with real capital.*
