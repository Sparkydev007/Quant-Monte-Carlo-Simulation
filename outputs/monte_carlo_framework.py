"""Production-grade Monte Carlo simulation framework for quantitative finance.

This module implements vectorized stochastic-process simulation, option pricing,
and distributional risk metrics with reproducible random-number generation.
"""

from __future__ import annotations

import argparse
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from math import exp, log, sqrt
from pathlib import Path
from typing import Mapping, Optional, Protocol, Tuple
from urllib.parse import parse_qs, urlparse
import webbrowser

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm, qmc


FloatArray = NDArray[np.float64]


class OptionType(str, Enum):
    """Supported vanilla option payoff directions."""

    CALL = "call"
    PUT = "put"


class SamplingMethod(str, Enum):
    """Supported Monte Carlo sampling methods."""

    PRNG = "prng"
    SOBOL = "sobol"


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration shared by all stochastic-process simulators.

    Args:
        n_paths: Number of simulated paths requested by the caller.
        n_steps: Number of time steps over the simulation horizon.
        maturity: Time horizon in years.
        seed: Seed for NumPy's modern ``default_rng`` and scrambled Sobol QMC.
        sampling_method: Pseudo-random or Sobol quasi-random sampling.
        antithetic: Whether to use antithetic variates for Gaussian shocks.
        dtype: Floating-point dtype used for generated arrays.

    Raises:
        ValueError: If path count, step count, or maturity is invalid.
    """

    n_paths: int
    n_steps: int
    maturity: float
    seed: Optional[int] = None
    sampling_method: SamplingMethod = SamplingMethod.PRNG
    antithetic: bool = False
    dtype: np.dtype = np.dtype(np.float64)

    def __post_init__(self) -> None:
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.maturity <= 0.0:
            raise ValueError("maturity must be positive.")

    @property
    def dt(self) -> float:
        """Length of one simulation step in years."""

        return self.maturity / self.n_steps


class Payoff(Protocol):
    """Protocol for vectorized payoff functions."""

    def __call__(self, paths: FloatArray) -> FloatArray:
        """Evaluate a payoff for each path."""


class RandomShockGenerator:
    """Generate reproducible Gaussian shocks for PRNG and Sobol sampling.

    Mathematical assumptions:
        Sobol dimensions are interpreted as independent uniforms in ``(0, 1)``
        and transformed to standard normal variates via the inverse Gaussian CDF.
        For antithetic variates, only half the required paths are sampled and the
        negative shocks are appended, improving estimator variance for monotone
        payoffs. Path count is rounded up internally and trimmed back to the
        requested number of paths.
    """

    _EPSILON = np.finfo(np.float64).eps

    def __init__(self, config: SimulationConfig) -> None:
        """Initialize the random shock generator.

        Args:
            config: Simulation configuration carrying sampling method and seed.
        """

        self._config = config
        self._rng = np.random.default_rng(config.seed)

    def standard_normals(self, shape: Tuple[int, ...]) -> FloatArray:
        """Return standard normal shocks with the requested final shape.

        Args:
            shape: Output shape. The first dimension is interpreted as paths.

        Returns:
            A NumPy array of standard normal random variables.
        """

        if len(shape) < 2:
            raise ValueError("shape must include path and stochastic dimensions.")

        n_paths = shape[0]
        base_paths = (n_paths + 1) // 2 if self._config.antithetic else n_paths
        base_shape = (base_paths, *shape[1:])

        if self._config.sampling_method == SamplingMethod.PRNG:
            base = self._rng.standard_normal(size=base_shape).astype(self._config.dtype, copy=False)
        elif self._config.sampling_method == SamplingMethod.SOBOL:
            base = self._sobol_normals(base_shape)
        else:
            raise ValueError(f"Unsupported sampling method: {self._config.sampling_method}")

        if not self._config.antithetic:
            return base[:n_paths]

        antithetic = np.concatenate((base, -base), axis=0)
        return antithetic[:n_paths]

    def poisson(self, lam: float, shape: Tuple[int, ...]) -> NDArray[np.int64]:
        """Return Poisson random variables from the reproducible PRNG.

        Args:
            lam: Poisson intensity over one time step.
            shape: Output shape.

        Returns:
            Integer-valued Poisson samples.
        """

        return self._rng.poisson(lam=lam, size=shape)

    def _sobol_normals(self, shape: Tuple[int, ...]) -> FloatArray:
        dimension = int(np.prod(shape[1:]))
        sampler = qmc.Sobol(d=dimension, scramble=True, seed=self._config.seed)
        exponent = int(np.ceil(np.log2(shape[0])))
        uniforms = sampler.random_base2(m=exponent)[: shape[0]]
        uniforms = np.clip(uniforms, self._EPSILON, 1.0 - self._EPSILON)
        normals = norm.ppf(uniforms).reshape(shape)
        return normals.astype(self._config.dtype, copy=False)


class StochasticProcess(ABC):
    """Abstract base class for vectorized stochastic-process simulators."""

    def __init__(self, initial_value: ArrayLike, config: SimulationConfig) -> None:
        """Initialize a stochastic process.

        Args:
            initial_value: Initial asset level(s). Scalar for single-asset models
                or a one-dimensional vector for multi-asset models.
            config: Simulation configuration.
        """

        self.initial_value = np.asarray(initial_value, dtype=config.dtype)
        self.config = config
        self.shocks = RandomShockGenerator(config)

    @abstractmethod
    def simulate(self) -> FloatArray:
        """Generate simulated asset paths.

        Returns:
            Array with shape ``(n_paths, n_steps + 1)`` for single-asset models
            or ``(n_paths, n_steps + 1, n_assets)`` for multi-asset models.
        """


class GeometricBrownianMotion(StochasticProcess):
    """Single-asset Geometric Brownian Motion under lognormal dynamics.

    Dynamics:
        ``dS_t / S_t = mu dt + sigma dW_t``.
    """

    def __init__(self, initial_value: float, mu: float, sigma: float, config: SimulationConfig) -> None:
        """Initialize a single-asset GBM process.

        Args:
            initial_value: Initial spot price.
            mu: Annualized drift.
            sigma: Annualized volatility.
            config: Simulation configuration.
        """

        super().__init__(initial_value, config)
        if sigma < 0.0:
            raise ValueError("sigma must be non-negative.")
        self.mu = float(mu)
        self.sigma = float(sigma)

    def simulate(self) -> FloatArray:
        """Generate GBM paths using exact log-Euler discretization."""

        z = self.shocks.standard_normals((self.config.n_paths, self.config.n_steps))
        dt = self.config.dt
        increments = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * sqrt(dt) * z
        log_paths = np.cumsum(increments, axis=1)
        paths = np.empty((self.config.n_paths, self.config.n_steps + 1), dtype=self.config.dtype)
        paths[:, 0] = float(self.initial_value)
        paths[:, 1:] = float(self.initial_value) * np.exp(log_paths)
        return paths


class MertonJumpDiffusion(StochasticProcess):
    """Single-asset Merton jump-diffusion with lognormal jumps.

    Dynamics:
        ``dS_t / S_t = (mu - lambda * kappa) dt + sigma dW_t + (J - 1)dN_t``,
        where ``log(J) ~ Normal(jump_mean, jump_vol^2)`` and
        ``kappa = E[J - 1]``.
    """

    def __init__(
        self,
        initial_value: float,
        mu: float,
        sigma: float,
        jump_intensity: float,
        jump_mean: float,
        jump_vol: float,
        config: SimulationConfig,
    ) -> None:
        """Initialize a Merton jump-diffusion process.

        Args:
            initial_value: Initial spot price.
            mu: Annualized drift before jump compensation.
            sigma: Annualized diffusion volatility.
            jump_intensity: Expected jumps per year.
            jump_mean: Mean of log jump size.
            jump_vol: Volatility of log jump size.
            config: Simulation configuration.
        """

        super().__init__(initial_value, config)
        if sigma < 0.0 or jump_intensity < 0.0 or jump_vol < 0.0:
            raise ValueError("sigma, jump_intensity, and jump_vol must be non-negative.")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.jump_intensity = float(jump_intensity)
        self.jump_mean = float(jump_mean)
        self.jump_vol = float(jump_vol)

    def simulate(self) -> FloatArray:
        """Generate Merton jump-diffusion paths with compound normal jumps."""

        z_diffusion = self.shocks.standard_normals((self.config.n_paths, self.config.n_steps))
        z_jump = self.shocks.standard_normals((self.config.n_paths, self.config.n_steps))
        dt = self.config.dt

        jump_counts = self.shocks.poisson(self.jump_intensity * dt, (self.config.n_paths, self.config.n_steps))
        jump_log_sum = jump_counts * self.jump_mean + np.sqrt(jump_counts) * self.jump_vol * z_jump
        jump_compensator = self.jump_intensity * (exp(self.jump_mean + 0.5 * self.jump_vol**2) - 1.0)
        drift = (self.mu - jump_compensator - 0.5 * self.sigma**2) * dt
        increments = drift + self.sigma * sqrt(dt) * z_diffusion + jump_log_sum

        log_paths = np.cumsum(increments, axis=1)
        paths = np.empty((self.config.n_paths, self.config.n_steps + 1), dtype=self.config.dtype)
        paths[:, 0] = float(self.initial_value)
        paths[:, 1:] = float(self.initial_value) * np.exp(log_paths)
        return paths


class MultiAssetGeometricBrownianMotion(StochasticProcess):
    """Multi-asset GBM with correlated Brownian drivers via Cholesky factorization.

    Dynamics:
        ``dS_i / S_i = mu_i dt + sigma_i dW_i`` with
        ``Cov[dW_i, dW_j] = corr_ij dt``. The covariance matrix supplied to this
        class is interpreted as the instantaneous covariance of returns, so
        asset volatilities are the square roots of its diagonal.
    """

    def __init__(
        self,
        initial_values: ArrayLike,
        mu: ArrayLike,
        covariance_matrix: ArrayLike,
        config: SimulationConfig,
    ) -> None:
        """Initialize a correlated multi-asset GBM process.

        Args:
            initial_values: Initial spot vector with shape ``(n_assets,)``.
            mu: Annualized drift vector with shape ``(n_assets,)``.
            covariance_matrix: Positive-definite annualized return covariance
                matrix with shape ``(n_assets, n_assets)``.
            config: Simulation configuration.
        """

        super().__init__(initial_values, config)
        self.mu = np.asarray(mu, dtype=config.dtype)
        self.covariance_matrix = np.asarray(covariance_matrix, dtype=config.dtype)
        self._validate_inputs()
        self._cholesky = np.linalg.cholesky(self.covariance_matrix)
        self._variances = np.diag(self.covariance_matrix)

    @property
    def n_assets(self) -> int:
        """Number of modeled assets."""

        return int(self.initial_value.shape[0])

    def simulate(self) -> FloatArray:
        """Generate correlated multi-asset GBM paths using vectorized tensor ops."""

        independent_z = self.shocks.standard_normals((self.config.n_paths, self.config.n_steps, self.n_assets))
        correlated_z = independent_z @ self._cholesky.T
        dt = self.config.dt
        drift = (self.mu - 0.5 * self._variances) * dt
        increments = drift + sqrt(dt) * correlated_z
        log_paths = np.cumsum(increments, axis=1)

        paths = np.empty((self.config.n_paths, self.config.n_steps + 1, self.n_assets), dtype=self.config.dtype)
        paths[:, 0, :] = self.initial_value
        paths[:, 1:, :] = self.initial_value * np.exp(log_paths)
        return paths

    def _validate_inputs(self) -> None:
        if self.initial_value.ndim != 1:
            raise ValueError("initial_values must be a one-dimensional vector.")
        if self.mu.shape != self.initial_value.shape:
            raise ValueError("mu must have the same shape as initial_values.")
        expected_shape = (self.n_assets, self.n_assets)
        if self.covariance_matrix.shape != expected_shape:
            raise ValueError(f"covariance_matrix must have shape {expected_shape}.")
        if not np.allclose(self.covariance_matrix, self.covariance_matrix.T):
            raise ValueError("covariance_matrix must be symmetric.")
        if np.any(np.diag(self.covariance_matrix) < 0.0):
            raise ValueError("covariance_matrix diagonal entries must be non-negative.")


@dataclass(frozen=True)
class OptionPricer:
    """Vectorized option-pricing utilities over simulated path tensors."""

    risk_free_rate: float

    def european_option(
        self,
        paths: FloatArray,
        strike: float,
        maturity: float,
        option_type: OptionType = OptionType.CALL,
        asset_index: int = 0,
    ) -> float:
        """Price a European option from simulated terminal prices.

        Args:
            paths: Simulated paths, either single-asset or multi-asset.
            strike: Option strike.
            maturity: Time to maturity in years for discounting.
            option_type: Call or put.
            asset_index: Asset index used when ``paths`` is multi-asset.

        Returns:
            Discounted Monte Carlo price.
        """

        terminal = self._asset_path(paths, asset_index)[:, -1]
        payoff = self._vanilla_payoff(terminal, strike, option_type)
        return float(np.mean(payoff) * exp(-self.risk_free_rate * maturity))

    def arithmetic_asian_option(
        self,
        paths: FloatArray,
        strike: float,
        maturity: float,
        option_type: OptionType = OptionType.CALL,
        asset_index: int = 0,
        include_initial: bool = False,
    ) -> float:
        """Price an arithmetic-average Asian option from simulated paths.

        Args:
            paths: Simulated paths, either single-asset or multi-asset.
            strike: Option strike.
            maturity: Time to maturity in years for discounting.
            option_type: Call or put.
            asset_index: Asset index used when ``paths`` is multi-asset.
            include_initial: Whether to include the initial spot in the average.

        Returns:
            Discounted Monte Carlo Asian option price.
        """

        asset_paths = self._asset_path(paths, asset_index)
        averaging_slice = slice(None) if include_initial else slice(1, None)
        average_price = np.mean(asset_paths[:, averaging_slice], axis=1)
        payoff = self._vanilla_payoff(average_price, strike, option_type)
        return float(np.mean(payoff) * exp(-self.risk_free_rate * maturity))

    @staticmethod
    def payoff_distribution_arithmetic_asian(
        paths: FloatArray,
        strike: float,
        option_type: OptionType = OptionType.CALL,
        asset_index: int = 0,
        include_initial: bool = False,
    ) -> FloatArray:
        """Return undiscounted Asian option payoffs for risk analysis."""

        asset_paths = OptionPricer._asset_path(paths, asset_index)
        averaging_slice = slice(None) if include_initial else slice(1, None)
        average_price = np.mean(asset_paths[:, averaging_slice], axis=1)
        return OptionPricer._vanilla_payoff(average_price, strike, option_type)

    @staticmethod
    def portfolio_terminal_values(paths: FloatArray, weights: ArrayLike) -> FloatArray:
        """Compute terminal portfolio values from multi-asset paths.

        Args:
            paths: Multi-asset simulated paths with shape
                ``(n_paths, n_steps + 1, n_assets)``.
            weights: Asset units or portfolio weights with shape ``(n_assets,)``.

        Returns:
            Terminal portfolio distribution.
        """

        if paths.ndim != 3:
            raise ValueError("portfolio_terminal_values requires multi-asset paths.")
        weights_array = np.asarray(weights, dtype=paths.dtype)
        if weights_array.shape != (paths.shape[2],):
            raise ValueError("weights must have shape (n_assets,).")
        return paths[:, -1, :] @ weights_array

    @staticmethod
    def _asset_path(paths: FloatArray, asset_index: int) -> FloatArray:
        if paths.ndim == 2:
            return paths
        if paths.ndim == 3:
            return paths[:, :, asset_index]
        raise ValueError("paths must be a 2D single-asset or 3D multi-asset tensor.")

    @staticmethod
    def _vanilla_payoff(underlying: FloatArray, strike: float, option_type: OptionType) -> FloatArray:
        if option_type == OptionType.CALL:
            return np.maximum(underlying - strike, 0.0)
        if option_type == OptionType.PUT:
            return np.maximum(strike - underlying, 0.0)
        raise ValueError(f"Unsupported option_type: {option_type}")

@dataclass(frozen=True)
class RiskMetrics:
    """Container for distributional risk metrics."""

    mean: float
    standard_deviation: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    def as_dict(self) -> Mapping[str, float]:
        """Return metrics as a mapping suitable for logging or tabulation."""

        return {
            "Mean": self.mean,
            "Standard Deviation": self.standard_deviation,
            "VaR 95%": self.var_95,
            "VaR 99%": self.var_99,
            "CVaR 95%": self.cvar_95,
            "CVaR 99%": self.cvar_99,
        }


class RiskAnalyzer:
    """Compute vectorized distributional risk metrics.

    Convention:
        VaR and CVaR are reported as positive loss magnitudes over a supplied
        profit-and-loss distribution. If a value distribution is supplied,
        convert it to PnL first, e.g. ``terminal_value - initial_value``.
    """

    @staticmethod
    def summarize(pnl: ArrayLike) -> RiskMetrics:
        """Summarize a PnL distribution.

        Args:
            pnl: One-dimensional profit-and-loss samples.

        Returns:
            Mean, standard deviation, 95/99 percent VaR, and 95/99 percent CVaR.
        """

        pnl_array = np.asarray(pnl, dtype=np.float64)
        if pnl_array.ndim != 1:
            raise ValueError("pnl must be one-dimensional.")
        if pnl_array.size == 0:
            raise ValueError("pnl must not be empty.")

        loss = -pnl_array
        var_95 = float(np.quantile(loss, 0.95))
        var_99 = float(np.quantile(loss, 0.99))
        cvar_95 = float(loss[loss >= var_95].mean())
        cvar_99 = float(loss[loss >= var_99].mean())
        return RiskMetrics(
            mean=float(pnl_array.mean()),
            standard_deviation=float(pnl_array.std(ddof=1)),
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
        )


@dataclass(frozen=True)
class SimulationResult:
    """Output bundle for the demonstration simulation.

    Args:
        paths: Simulated correlated multi-asset paths.
        portfolio_pnl: Terminal portfolio profit-and-loss samples.
        asian_call_price: Discounted arithmetic Asian call option price.
        risk_metrics: Distributional risk summary for terminal portfolio PnL.
        config: Simulation configuration used to produce the result.
    """

    paths: FloatArray
    portfolio_pnl: FloatArray
    asian_call_price: float
    risk_metrics: RiskMetrics
    config: SimulationConfig


def _format_metrics(metrics: RiskMetrics) -> str:
    rows = [f"{name:<22}: {value:>14,.6f}" for name, value in metrics.as_dict().items()]
    return "\n".join(rows)


def run_demo_simulation(
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int = 42,
    strike: float = 100.0,
) -> SimulationResult:
    """Run the representative 3-asset Sobol QMC simulation.

    Args:
        n_paths: Number of Monte Carlo paths.
        n_steps: Number of time steps.
        seed: Random seed for Sobol scrambling and PRNG components.
        strike: Asian call strike on the first asset.

    Returns:
        Simulation paths, option price, PnL distribution, and risk metrics.
    """

    config = SimulationConfig(
        n_paths=n_paths,
        n_steps=n_steps,
        maturity=1.0,
        seed=seed,
        sampling_method=SamplingMethod.SOBOL,
        antithetic=True,
    )
    spots = np.array([100.0, 95.0, 105.0])
    drifts = np.array([0.05, 0.045, 0.055])
    volatilities = np.array([0.20, 0.18, 0.22])
    correlation = np.array(
        [
            [1.00, 0.35, 0.20],
            [0.35, 1.00, 0.40],
            [0.20, 0.40, 1.00],
        ]
    )
    covariance = np.outer(volatilities, volatilities) * correlation

    process = MultiAssetGeometricBrownianMotion(
        initial_values=spots,
        mu=drifts,
        covariance_matrix=covariance,
        config=config,
    )
    paths = process.simulate()

    pricer = OptionPricer(risk_free_rate=0.04)
    asian_call_price = pricer.arithmetic_asian_option(
        paths=paths,
        strike=strike,
        maturity=config.maturity,
        option_type=OptionType.CALL,
        asset_index=0,
    )

    weights = np.array([0.40, 0.35, 0.25])
    initial_portfolio_value = float(spots @ weights)
    terminal_portfolio_value = OptionPricer.portfolio_terminal_values(paths, weights)
    portfolio_pnl = terminal_portfolio_value - initial_portfolio_value
    metrics = RiskAnalyzer.summarize(portfolio_pnl)
    return SimulationResult(
        paths=paths,
        portfolio_pnl=portfolio_pnl,
        asian_call_price=asian_call_price,
        risk_metrics=metrics,
        config=config,
    )


def create_distribution_figure(portfolio_pnl: FloatArray, metrics: RiskMetrics):
    """Create a Matplotlib histogram figure for the simulated PnL distribution.

    Args:
        portfolio_pnl: One-dimensional terminal portfolio PnL distribution.
        metrics: Risk metrics used to annotate VaR thresholds.

    Returns:
        A Matplotlib ``Figure`` object.
    """

    from matplotlib.figure import Figure

    figure = Figure(figsize=(10.0, 5.8), dpi=110)
    axis = figure.add_subplot(111)
    axis.hist(portfolio_pnl, bins=90, density=True, alpha=0.78, color="#2f6f8f", edgecolor="#ffffff")
    axis.axvline(float(np.mean(portfolio_pnl)), color="#1f2933", linewidth=2.0, label="Mean PnL")
    axis.axvline(-metrics.var_95, color="#d97706", linestyle="--", linewidth=2.0, label="95% VaR")
    axis.axvline(-metrics.var_99, color="#b91c1c", linestyle="--", linewidth=2.0, label="99% VaR")
    axis.set_title("Terminal Portfolio PnL Distribution")
    axis.set_xlabel("Profit and Loss")
    axis.set_ylabel("Probability Density")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right")
    figure.tight_layout()
    return figure


def save_distribution_graph(
    portfolio_pnl: FloatArray,
    metrics: RiskMetrics,
    output_path: Path,
) -> Path:
    """Save the terminal PnL distribution graph to disk.

    Args:
        portfolio_pnl: One-dimensional terminal portfolio PnL distribution.
        metrics: Risk metrics used to annotate VaR thresholds.
        output_path: Destination image path.

    Returns:
        The saved image path.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = create_distribution_figure(portfolio_pnl, metrics)
    figure.savefig(output_path, bbox_inches="tight")
    return output_path


class MonteCarloDashboard:
    """Tkinter GUI for running the demo simulation and visualizing risk output."""

    def __init__(self) -> None:
        """Create the dashboard widgets and default state."""

        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self._tk = tk
        self._ttk = ttk
        self._figure_canvas_type = FigureCanvasTkAgg
        self.root = tk.Tk()
        self.root.title("Monte Carlo Quant Dashboard")
        self.root.geometry("1180x760")

        self.paths_var = tk.StringVar(value="100000")
        self.steps_var = tk.StringVar(value="252")
        self.seed_var = tk.StringVar(value="42")
        self.strike_var = tk.StringVar(value="100.0")
        self.status_var = tk.StringVar(value="Ready")
        self.metrics_var = tk.StringVar(value="Run a simulation to populate metrics.")
        self.canvas = None
        self._build_layout()

    def run(self) -> None:
        """Start the Tkinter event loop."""

        self.root.mainloop()

    def _build_layout(self) -> None:
        frame = self._ttk.Frame(self.root, padding=14)
        frame.pack(fill="both", expand=True)

        controls = self._ttk.Frame(frame)
        controls.pack(fill="x")

        self._add_labeled_entry(controls, "Paths", self.paths_var, 0)
        self._add_labeled_entry(controls, "Steps", self.steps_var, 1)
        self._add_labeled_entry(controls, "Seed", self.seed_var, 2)
        self._add_labeled_entry(controls, "Strike", self.strike_var, 3)

        run_button = self._ttk.Button(controls, text="Run Simulation", command=self._run_from_gui)
        run_button.grid(row=0, column=8, padx=(16, 0), sticky="ew")

        self._ttk.Label(frame, textvariable=self.status_var).pack(anchor="w", pady=(10, 4))
        self._ttk.Label(frame, textvariable=self.metrics_var, font=("Consolas", 10), justify="left").pack(
            anchor="w",
            pady=(0, 8),
        )

        plot_frame = self._ttk.Frame(frame)
        plot_frame.pack(fill="both", expand=True)
        self.plot_frame = plot_frame

    def _add_labeled_entry(self, parent, label: str, variable, column: int) -> None:
        self._ttk.Label(parent, text=label).grid(row=0, column=column * 2, padx=(0, 6), sticky="w")
        self._ttk.Entry(parent, textvariable=variable, width=12).grid(
            row=0,
            column=column * 2 + 1,
            padx=(0, 10),
            sticky="ew",
        )

    def _run_from_gui(self) -> None:
        try:
            n_paths = int(self.paths_var.get())
            n_steps = int(self.steps_var.get())
            seed = int(self.seed_var.get())
            strike = float(self.strike_var.get())
            self.status_var.set("Running simulation...")
            self.root.update_idletasks()
            result = run_demo_simulation(n_paths=n_paths, n_steps=n_steps, seed=seed, strike=strike)
        except Exception as exc:  # pragma: no cover - GUI defensive boundary.
            self.status_var.set(f"Error: {exc}")
            return

        self.metrics_var.set(
            f"Arithmetic Asian Call : {result.asian_call_price:,.6f}\n"
            f"{_format_metrics(result.risk_metrics)}"
        )
        self._draw_plot(result)
        self.status_var.set("Complete")

    def _draw_plot(self, result: SimulationResult) -> None:
        figure = create_distribution_figure(result.portfolio_pnl, result.risk_metrics)
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.canvas = self._figure_canvas_type(figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)


def _figure_to_base64_png(portfolio_pnl: FloatArray, metrics: RiskMetrics) -> str:
    """Encode a distribution figure as a base64 PNG for browser display."""

    buffer = BytesIO()
    figure = create_distribution_figure(portfolio_pnl, metrics)
    figure.savefig(buffer, format="png", bbox_inches="tight")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _render_web_dashboard(
    *,
    n_paths: int,
    n_steps: int,
    seed: int,
    strike: float,
    result: Optional[SimulationResult] = None,
    error: Optional[str] = None,
) -> bytes:
    """Render the browser dashboard HTML."""

    metrics_html = ""
    chart_html = ""
    error_html = f"<p class='error'>{error}</p>" if error else ""
    if result is not None:
        encoded_plot = _figure_to_base64_png(result.portfolio_pnl, result.risk_metrics)
        metrics_rows = "\n".join(
            f"<tr><th>{name}</th><td>{value:,.6f}</td></tr>"
            for name, value in result.risk_metrics.as_dict().items()
        )
        metrics_html = f"""
        <section class="panel">
          <h2>Results</h2>
          <p class="price">Arithmetic Asian Call: <strong>{result.asian_call_price:,.6f}</strong></p>
          <table>{metrics_rows}</table>
        </section>
        """
        chart_html = f"""
        <section class="plot">
          <img alt="Terminal portfolio PnL distribution" src="data:image/png;base64,{encoded_plot}">
        </section>
        """

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Monte Carlo Quant Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17212b;
      --muted: #5d6b78;
      --line: #d8dee6;
      --surface: #ffffff;
      --band: #f5f7fa;
      --accent: #2f6f8f;
      --danger: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--band);
    }}
    header {{
      padding: 22px 28px;
      background: #1f2933;
      color: #ffffff;
    }}
    h1, h2 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 18px; margin-bottom: 12px; }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 18px auto 32px;
      display: grid;
      gap: 16px;
    }}
    form, .panel, .plot {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    form {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr)) auto;
      gap: 12px;
      align-items: end;
    }}
    label {{
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 13px;
    }}
    input {{
      width: 100%;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font-size: 14px;
    }}
    button {{
      height: 38px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      font-weight: 600;
      cursor: pointer;
      padding: 0 16px;
    }}
    .error {{ color: var(--danger); font-weight: 600; }}
    .price {{ margin: 0 0 12px; }}
    table {{
      border-collapse: collapse;
      min-width: min(480px, 100%);
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: right;
    }}
    th {{
      text-align: left;
      color: var(--muted);
      font-weight: 600;
    }}
    img {{
      display: block;
      width: 100%;
      max-height: 620px;
      object-fit: contain;
    }}
    @media (max-width: 780px) {{
      form {{ grid-template-columns: 1fr 1fr; }}
      button {{ grid-column: 1 / -1; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Monte Carlo Quant Dashboard</h1>
  </header>
  <main>
    <form action="/run" method="get">
      <label>Paths <input name="paths" type="number" min="1" value="{n_paths}"></label>
      <label>Steps <input name="steps" type="number" min="1" value="{n_steps}"></label>
      <label>Seed <input name="seed" type="number" value="{seed}"></label>
      <label>Strike <input name="strike" type="number" step="0.01" value="{strike}"></label>
      <button type="submit">Run Simulation</button>
    </form>
    {error_html}
    {metrics_html}
    {chart_html}
  </main>
</body>
</html>"""
    return html.encode("utf-8")


def launch_web_dashboard(host: str = "127.0.0.1", port: int = 8050) -> None:
    """Launch a local browser dashboard using only the Python standard library."""

    class DashboardRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler.
            parsed_url = urlparse(self.path)
            query = parse_qs(parsed_url.query)
            n_paths = int(query.get("paths", ["100000"])[0])
            n_steps = int(query.get("steps", ["252"])[0])
            seed = int(query.get("seed", ["42"])[0])
            strike = float(query.get("strike", ["100.0"])[0])
            result = None
            error = None

            if parsed_url.path == "/run":
                try:
                    result = run_demo_simulation(
                        n_paths=n_paths,
                        n_steps=n_steps,
                        seed=seed,
                        strike=strike,
                    )
                except Exception as exc:  # pragma: no cover - HTTP defensive boundary.
                    error = str(exc)
            elif parsed_url.path not in {"/", "/favicon.ico"}:
                self.send_error(404)
                return

            if parsed_url.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return

            body = _render_web_dashboard(
                n_paths=n_paths,
                n_steps=n_steps,
                seed=seed,
                strike=strike,
                result=result,
                error=error,
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = HTTPServer((host, port), DashboardRequestHandler)
    url = f"http://{host}:{port}"
    print(f"Browser GUI running at {url}")
    print("Press Ctrl+C to stop the dashboard.")
    webbrowser.open(url)
    server.serve_forever()


def main() -> None:
    """Run the demo simulation, save the distribution graph, and optionally open GUI."""

    parser = argparse.ArgumentParser(description="Monte Carlo simulation framework demo.")
    parser.add_argument("--paths", type=int, default=100_000, help="Number of Monte Carlo paths.")
    parser.add_argument("--steps", type=int, default=252, help="Number of time steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--strike", type=float, default=100.0, help="Asian option strike.")
    parser.add_argument("--plot", type=Path, default=Path(__file__).with_name("portfolio_pnl_distribution.png"))
    parser.add_argument("--gui", action="store_true", help="Launch the Tkinter GUI after the command-line demo.")
    args = parser.parse_args()

    result = run_demo_simulation(
        n_paths=args.paths,
        n_steps=args.steps,
        seed=args.seed,
        strike=args.strike,
    )
    graph_path = save_distribution_graph(result.portfolio_pnl, result.risk_metrics, args.plot)

    print("Monte Carlo Simulation Summary")
    print("==============================")
    print(f"Paths                 : {result.config.n_paths:,}")
    print(f"Steps                 : {result.config.n_steps:,}")
    print(f"Sampling              : {result.config.sampling_method.value.upper()} + Antithetic")
    print(f"Arithmetic Asian Call : {result.asian_call_price:,.6f}")
    print(f"Distribution Graph    : {graph_path}")
    print("\nPortfolio Risk Metrics")
    print("----------------------")
    print(_format_metrics(result.risk_metrics))

    if args.gui:
        try:
            MonteCarloDashboard().run()
        except ModuleNotFoundError as exc:
            if exc.name != "tkinter":
                raise
            print("\nTkinter is not installed in this Python distribution.")
            print("Falling back to the browser-based GUI; no pip install is required.")
            launch_web_dashboard()


if __name__ == "__main__":
    main()
