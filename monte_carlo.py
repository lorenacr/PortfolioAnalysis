"""
Monte Carlo Utilities for Portfolio Analysis Framework
=====================================================

Implements Monte Carlo simulations based on trade reordering (random permutation
of realized trade PnL) to estimate distributional risk metrics.

Primary use-case:
    - Estimate drawdown-at-confidence (e.g., 95th percentile of max drawdown)
      under random trade ordering.

Parallelism:
    - Always splits ``n_sims`` across ``MC_PARALLEL_WORKERS`` threads (from
      ``custom_config``), capped by ``n_sims``; each worker uses an independent RNG
      stream via ``SeedSequence``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np

from custom_config import MC_PARALLEL_WORKERS


def _split_n_sims(n_sims: int, n_workers: int) -> List[int]:
    """
    Split n_sims into n_workers non-negative integers that sum to n_sims.

    Args:
        n_sims: Total number of simulations.
        n_workers: Desired number of slices (capped by n_sims).

    Returns:
        List of slice sizes, length min(n_workers, n_sims), summing to n_sims.
    """
    n_workers = max(1, min(int(n_workers), int(n_sims)))
    base = n_sims // n_workers
    rem = n_sims % n_workers
    return [base + (1 if i < rem else 0) for i in range(n_workers)]


def _simulate_max_drawdown_slice(
    pnl: np.ndarray,
    n_sims_slice: int,
    rng: np.random.Generator,
    initial_capital: float,
    batch_size: int,
) -> np.ndarray:
    """
    Run n_sims_slice independent trade-reordering simulations and return max drawdown per sim.

    Args:
        pnl: 1D float64 trade PnL (no NaNs).
        n_sims_slice: Number of simulations in this slice.
        rng: NumPy Generator (unique per worker thread).
        initial_capital: Starting equity.
        batch_size: Inner batch size for vectorized permutations.

    Returns:
        1D array of length n_sims_slice with max drawdown per simulation.
    """
    n_trades = int(pnl.size)
    max_dd = np.empty(int(n_sims_slice), dtype=np.float64)
    sims_done = 0

    while sims_done < n_sims_slice:
        b = int(min(batch_size, n_sims_slice - sims_done))

        keys = rng.random((b, n_trades), dtype=np.float32)
        perm_idx = np.argsort(keys, axis=1)
        pnl_perm = pnl[perm_idx]

        equity = float(initial_capital) + np.cumsum(pnl_perm, axis=1)
        peaks = np.maximum.accumulate(equity, axis=1)

        safe_peaks = np.where(peaks == 0.0, np.finfo(np.float64).eps, peaks)
        dd = (safe_peaks - equity) / safe_peaks

        max_dd[sims_done : sims_done + b] = dd.max(axis=1)
        sims_done += b

    return max_dd


def calculate_trade_reordering_drawdown_metrics(
    trade_pnl: np.ndarray,
    n_sims: int = 5000,
    confidence: float = 0.95,
    initial_capital: float = 10000.0,
    seed: Optional[int] = None,
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    Estimate drawdown distribution under randomized trade ordering.

    This simulation preserves the multiset of realized trade outcomes but
    randomizes their sequence. For each simulation:
        - Randomly permute the trade PnL sequence (without replacement).
        - Build an equity curve: initial_capital + cumulative sum of PnL.
        - Compute max drawdown as (peak - equity) / peak over the path.

    Simulations are always parallelized: ``n_sims`` is split across
    ``MC_PARALLEL_WORKERS`` threads (from ``custom_config``), each with a distinct
    RNG stream derived from ``seed`` and the worker index.

    Args:
        trade_pnl: 1D array-like of per-trade PnL values.
        n_sims: Number of Monte Carlo simulations (permutations).
        confidence: Quantile level for tail metric (0.95 => 95th percentile).
        initial_capital: Starting equity used to compute drawdowns.
        seed: Optional RNG seed for reproducibility.
        batch_size: Number of simulations processed per batch to control memory.

    Returns:
        Dictionary of Monte Carlo drawdown metrics:
            - mc_max_dd_p95: confidence-quantile of max drawdown distribution
            - mc_max_dd_mean: mean of simulated max drawdowns
            - mc_max_dd_median: median of simulated max drawdowns
    """
    try:
        pnl = np.asarray(trade_pnl, dtype=np.float64).reshape(-1)
        pnl = pnl[~np.isnan(pnl)]

        if n_sims <= 0 or pnl.size < 2:
            return {"mc_max_dd_p95": 0.0, "mc_max_dd_mean": 0.0, "mc_max_dd_median": 0.0}

        confidence = float(confidence)
        if not (0.0 < confidence < 1.0):
            raise ValueError("confidence must be in (0, 1)")

        parallel_workers = max(1, int(MC_PARALLEL_WORKERS))
        slice_sizes = _split_n_sims(int(n_sims), parallel_workers)
        base_seed = 0 if seed is None else int(seed)

        def _run_slice(worker_idx: int, n_slice: int) -> np.ndarray:
            ss = np.random.SeedSequence([base_seed, worker_idx])
            wrng = np.random.default_rng(ss)
            return _simulate_max_drawdown_slice(
                pnl, n_slice, wrng, float(initial_capital), int(batch_size)
            )

        with ThreadPoolExecutor(max_workers=len(slice_sizes)) as executor:
            futures = {
                executor.submit(_run_slice, i, sz): i
                for i, sz in enumerate(slice_sizes)
                if sz > 0
            }
            # Preserve worker order so concatenation matches a deterministic layout
            ordered: List[Optional[np.ndarray]] = [None] * len(slice_sizes)
            for fut in as_completed(futures):
                idx = futures[fut]
                ordered[idx] = fut.result()
            chunks = [c for c in ordered if c is not None]

        max_dd = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)

        p95 = float(np.quantile(max_dd, confidence))
        return {
            "mc_max_dd_p95": p95,
            "mc_max_dd_mean": float(max_dd.mean()),
            "mc_max_dd_median": float(np.median(max_dd)),
        }

    except Exception:
        # Fail closed: do not block portfolio analysis if MC fails
        return {"mc_max_dd_p95": 0.0, "mc_max_dd_mean": 0.0, "mc_max_dd_median": 0.0}
