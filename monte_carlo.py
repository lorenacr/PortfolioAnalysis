"""
Monte Carlo Utilities for Portfolio Analysis Framework
=====================================================

Implements Monte Carlo simulations based on trade reordering (random permutation
of realized trade PnL) to estimate distributional risk metrics.

Primary use-case:
    - Estimate drawdown-at-confidence (e.g., 95th percentile of max drawdown)
      under random trade ordering.

Implementation:
    - Each simulation uses ``np.random.shuffle`` on a copy of trade PnL (Fisher–Yates,
      O(n) per permutation), compiled via Numba for speed—replacing the previous
      vectorized ``argsort`` approach (O(n log n) per row).

Parallelism:
    - Always splits ``n_sims`` across ``MC_PARALLEL_WORKERS`` threads (from
      ``custom_config``), capped by ``n_sims``; each worker uses a distinct integer
      seed derived from ``SeedSequence`` for reproducibility.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
from numba import njit

from custom_config import MC_PARALLEL_WORKERS


@njit(cache=True)
def _simulate_shuffle_slice_numba(
    pnl: np.ndarray,
    n_sims_slice: int,
    initial_capital: float,
    seed: int,
) -> np.ndarray:
    """
    Run n_sims_slice simulations: shuffle trade PnL (O(n) per sim), equity path, max DD.

    Args:
        pnl: 1D float64 trade PnL (contiguous, no NaNs).
        n_sims_slice: Number of simulations in this worker slice.
        initial_capital: Starting equity.
        seed: NumPy legacy RNG seed (unique per worker for independent streams).

    Returns:
        1D float64 array of max drawdown per simulation (same convention as before).
    """
    n = pnl.shape[0]
    max_dd_out = np.empty(n_sims_slice, dtype=np.float64)
    buf = np.empty(n, dtype=np.float64)
    eps = 1e-15

    np.random.seed(seed)

    for s in range(n_sims_slice):
        for i in range(n):
            buf[i] = pnl[i]
        np.random.shuffle(buf)

        eq = initial_capital
        peak = initial_capital
        max_d = 0.0
        for i in range(n):
            eq = eq + buf[i]
            if eq > peak:
                peak = eq
            sp = peak
            if sp == 0.0:
                sp = eps
            d = (peak - eq) / sp
            if d > max_d:
                max_d = d
        max_dd_out[s] = max_d

    return max_dd_out


def _worker_seed_int(base_seed: int, worker_idx: int) -> int:
    """Derive a deterministic 32-bit seed for Numba's np.random.seed per worker."""
    word = np.random.SeedSequence([base_seed, worker_idx]).generate_state(1, dtype=np.uint32)[0]
    # Ensure positive int for broad NumPy/Numba compatibility
    return int(word) & 0x7FFFFFFF


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
        - Randomly permute the trade PnL sequence (without replacement) via shuffle.
        - Build an equity curve: initial_capital + cumulative sum of PnL.
        - Compute max drawdown as (peak - equity) / peak over the path.

    Simulations are always parallelized: ``n_sims`` is split across
    ``MC_PARALLEL_WORKERS`` threads (from ``custom_config``), each with a distinct
    RNG seed derived from ``seed`` and the worker index.

    Args:
        trade_pnl: 1D array-like of per-trade PnL values.
        n_sims: Number of Monte Carlo simulations (permutations).
        confidence: Quantile level for tail metric (0.95 => 95th percentile).
        initial_capital: Starting equity used to compute drawdowns.
        seed: Optional RNG seed for reproducibility.
        batch_size: Retained for API compatibility; ignored (shuffle path has no batching).

    Returns:
        Dictionary of Monte Carlo drawdown metrics:
            - mc_max_dd_p95: confidence-quantile of max drawdown distribution
            - mc_max_dd_mean: mean of simulated max drawdowns
            - mc_max_dd_median: median of simulated max drawdowns
    """
    del batch_size  # API compatibility; Numba shuffle loop does not batch

    try:
        pnl = np.asarray(trade_pnl, dtype=np.float64).reshape(-1)
        pnl = pnl[~np.isnan(pnl)]
        if not pnl.flags.c_contiguous:
            pnl = np.ascontiguousarray(pnl)

        if n_sims <= 0 or pnl.size < 2:
            return {"mc_max_dd_p95": 0.0, "mc_max_dd_mean": 0.0, "mc_max_dd_median": 0.0}

        confidence = float(confidence)
        if not (0.0 < confidence < 1.0):
            raise ValueError("confidence must be in (0, 1)")

        parallel_workers = max(1, int(MC_PARALLEL_WORKERS))
        slice_sizes = _split_n_sims(int(n_sims), parallel_workers)
        base_seed = 0 if seed is None else int(seed)

        def _run_slice(worker_idx: int, n_slice: int) -> np.ndarray:
            wseed = _worker_seed_int(base_seed, worker_idx)
            return _simulate_shuffle_slice_numba(
                pnl, int(n_slice), float(initial_capital), wseed
            )

        with ThreadPoolExecutor(max_workers=len(slice_sizes)) as executor:
            futures = {
                executor.submit(_run_slice, i, sz): i
                for i, sz in enumerate(slice_sizes)
                if sz > 0
            }
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
