#!/usr/bin/env python3
"""
KA Geometry Metrics - Proper Definitions and Computation

This module provides standardized computation of KA (Koszul-Alon) geometry metrics
for analyzing Jacobian structure in neural networks.

METRIC DEFINITIONS:
==================

1. PARTICIPATION RATIO (PR) - Two versions:

   a) Standard Physics PR = L1² / (n * L2²)
      - Range: [1/n, 1]
      - PR = 1: perfectly uniform (all values equal)
      - PR = 1/n: maximally concentrated (one dominant value)
      - HIGHER = MORE UNIFORM

   b) This Codebase PR = L2 / L1  (INVERTED!)
      - Range: [1/sqrt(n), 1]
      - PR ≈ 1/sqrt(n): uniform distribution
      - PR = 1: maximally concentrated
      - HIGHER = MORE CONCENTRATED

2. ROTATION RATIO (RR):
   RR = max(|original k-minors|) / mean(max(|rotated k-minors|))

   - RR > 1: ANISOTROPIC (direction-dependent, structured)
   - RR = 1: ISOTROPIC (rotation-invariant, random-like)
   - RR < 1: Less structured than random rotations

   The rotation is applied to ROWS of the Jacobian (output space).

3. KL DIVERGENCE:
   KL(p || uniform) where p is the normalized distribution of |k-minors|

   - KL = 0: perfectly uniform
   - KL > 0: deviation from uniform
   - HIGHER = LESS UNIFORM

4. k-MINORS:
   k×k determinants of submatrices sampled from the Jacobian.
   - k=1: Just column norms (simplest)
   - k=2: 2×2 determinants (captures pairwise relationships)
   - Higher k: More complex geometric structure

USAGE:
======
    from ka_metrics import KAMetricsComputer

    computer = KAMetricsComputer(device='cuda')

    # Compute metrics on a Jacobian matrix
    metrics = computer.compute_all_metrics(jacobian, k=2)

    # metrics contains: pr_standard, pr_codebase, rotation_ratio, kl_divergence
"""

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

# =============================================================================
# EFFICIENT JACOBIAN COMPUTATION
# =============================================================================


def compute_jacobian_efficient(
    func: Callable[[Tensor], Tensor], input_tensor: Tensor, method: str = "jacrev"
) -> Tensor:
    """
    Compute Jacobian efficiently using torch.func (GPU-accelerated).

    THIS IS THE CANONICAL WAY TO COMPUTE JACOBIANS IN THIS CODEBASE.
    Do NOT use row-by-row backward passes - that's 768x slower!

    Args:
        func: Function mapping input [d_in] -> output [d_out]
        input_tensor: Input tensor [d_in] (1D, no batch dim)
        method: 'jacrev' (reverse-mode, good for d_out > d_in) or
                'jacfwd' (forward-mode, good for d_in > d_out)

    Returns:
        Jacobian matrix [d_out, d_in]

    Example:
        >>> def mlp_forward(x):
        ...     return model.mlp(x.unsqueeze(0).unsqueeze(0))[0, 0, :]
        >>> jac = compute_jacobian_efficient(mlp_forward, hidden_state)  # [768, 768]
    """
    if method == "jacrev":
        jacobian_fn = torch.func.jacrev(func)
    elif method == "jacfwd":
        jacobian_fn = torch.func.jacfwd(func)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'jacrev' or 'jacfwd'.")

    return jacobian_fn(input_tensor)


class KAMetricsComputer:
    """Compute KA geometry metrics for Jacobian matrices."""

    def __init__(self, device: str = "cuda", seed: int = 42):
        self.device = device
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    # =========================================================================
    # k-MINOR COMPUTATION
    # =========================================================================

    def compute_kminors(
        self, jacobian: Tensor, k: int, num_samples: int = 500, col_indices: list[int] | None = None
    ) -> Tensor:
        """
        Compute k×k minors (determinants) by random sampling.

        Args:
            jacobian: [n_rows, n_cols] Jacobian matrix
            k: Size of minors (k×k determinants)
            num_samples: Number of minors to sample
            col_indices: Optional subset of columns to sample from

        Returns:
            Tensor of minor values [num_samples]
        """
        n_rows, n_cols = jacobian.shape

        if col_indices is None:
            col_indices = list(range(n_cols))

        row_indices = list(range(n_rows))

        minors = []
        for _ in range(num_samples):
            rows = np.random.choice(row_indices, k, replace=False)
            cols = np.random.choice(col_indices, k, replace=False)

            submatrix = jacobian[rows][:, cols]
            minor = torch.det(submatrix)
            minors.append(minor.item())

        return torch.tensor(minors, device=self.device)

    # =========================================================================
    # PARTICIPATION RATIO
    # =========================================================================

    def compute_pr(self, values: Tensor, eps: float = 1e-12) -> float:
        """
        Participation Ratio: L2 / L1 (THIS IS THE CODEBASE STANDARD)

        From spatial_scale_analysis.py:
            l2_norm = torch.norm(minors_abs, p=2)
            l1_norm = torch.norm(minors_abs, p=1)
            return (l2_norm / (l1_norm + eps)).item()

        Interpretation:
        - Range: [1/sqrt(n), 1]
        - HIGHER = MORE CONCENTRATED (few large values dominate)
        - LOWER = MORE UNIFORM (values spread evenly)

        Example:
        - Uniform [1,1,1,...,1] (n=100): PR = 10/100 = 0.1
        - Concentrated [1,0,0,...,0]: PR = 1/1 = 1.0
        """
        values_abs = torch.abs(values)
        values_abs = torch.nan_to_num(values_abs, nan=0.0, posinf=0.0, neginf=0.0)

        l1 = torch.norm(values_abs, p=1)
        l2 = torch.norm(values_abs, p=2)

        if l1 < eps or torch.isnan(l1) or torch.isnan(l2):
            return float("nan")

        return (l2 / (l1 + eps)).item()

    # =========================================================================
    # ROTATION RATIO
    # =========================================================================

    def compute_rotation_ratio(
        self,
        jacobian: Tensor,
        k: int,
        num_samples: int = 300,
        num_rotations: int = 5,
        col_indices: list[int] | None = None,
    ) -> float:
        """
        Rotation Ratio: max(|original minors|) / mean(max(|rotated minors|))

        - RR > 1: ANISOTROPIC (structured)
        - RR = 1: ISOTROPIC (random-like)

        Rotation is applied to rows (output space) via random orthogonal matrix.
        """
        n_rows, n_cols = jacobian.shape
        device = jacobian.device

        if col_indices is None:
            col_indices = list(range(n_cols))

        # Compute original minors
        original_minors = self.compute_kminors(jacobian, k, num_samples, col_indices)
        original_abs = torch.abs(original_minors)
        original_abs = torch.nan_to_num(original_abs, nan=0.0, posinf=0.0, neginf=0.0)
        original_max = original_abs.max()

        if original_max == 0 or torch.isnan(original_max):
            return float("nan")

        # Compute rotated minors
        rotated_maxes = []
        for _ in range(num_rotations):
            # Random orthogonal matrix (rotation in output space)
            Q, _ = torch.linalg.qr(torch.randn(n_rows, n_rows, device=device))
            jac_rotated = Q @ jacobian

            rotated_minors = self.compute_kminors(jac_rotated, k, num_samples, col_indices)
            rotated_abs = torch.abs(rotated_minors)
            rotated_abs = torch.nan_to_num(rotated_abs, nan=0.0, posinf=0.0, neginf=0.0)
            rotated_maxes.append(rotated_abs.max())

        rotated_mean = torch.stack(rotated_maxes).mean()

        if rotated_mean == 0 or torch.isnan(rotated_mean):
            return float("nan")

        return (original_max / rotated_mean).item()

    # =========================================================================
    # KL DIVERGENCE
    # =========================================================================

    def compute_kl_divergence(self, values: Tensor, eps: float = 1e-12) -> float:
        """
        KL divergence from uniform distribution.

        - KL = 0: perfectly uniform
        - HIGHER = LESS UNIFORM
        """
        values_abs = torch.abs(values)
        values_abs = torch.nan_to_num(values_abs, nan=0.0, posinf=0.0, neginf=0.0)

        if values_abs.sum() < eps:
            return float("nan")

        # Normalize to probability distribution
        p = values_abs / values_abs.sum()
        q = torch.ones_like(p) / len(p)  # Uniform

        # KL(p || q)
        kl = torch.sum(p * torch.log((p + eps) / q))
        return kl.item()

    # =========================================================================
    # COMBINED COMPUTATION
    # =========================================================================

    def compute_all_metrics(
        self,
        jacobian: Tensor,
        k: int = 2,
        num_samples: int = 500,
        num_rotations: int = 5,
        col_indices: list[int] | None = None,
    ) -> dict[str, float]:
        """
        Compute all KA metrics for a Jacobian matrix.

        Args:
            jacobian: [n_rows, n_cols] Jacobian matrix
            k: Size of minors
            num_samples: Number of minors to sample
            num_rotations: Number of rotations for RR
            col_indices: Optional subset of columns

        Returns:
            Dictionary with all metrics
        """
        # Ensure jacobian is on correct device
        jacobian = jacobian.to(self.device)

        # Compute k-minors
        minors = self.compute_kminors(jacobian, k, num_samples, col_indices)

        # Compute all metrics
        return {
            "pr": self.compute_pr(minors),  # L2/L1 - higher = more concentrated
            "rotation_ratio": self.compute_rotation_ratio(
                jacobian, k, num_samples, num_rotations, col_indices
            ),
            "kl_divergence": self.compute_kl_divergence(minors),
            "k": k,
            "num_samples": num_samples,
        }

    # =========================================================================
    # COLUMN NORM METRICS (k=1 special case)
    # =========================================================================

    def compute_column_norms(self, jacobian: Tensor) -> Tensor:
        """Compute L2 norm of each column."""
        return torch.norm(jacobian, dim=0)

    def compute_column_metrics(self, jacobian: Tensor) -> dict[str, float]:
        """Compute metrics based on column norms (k=1 equivalent)."""
        col_norms = self.compute_column_norms(jacobian)

        return {
            "pr_colnorms": self.compute_pr(col_norms),
            "kl_colnorms": self.compute_kl_divergence(col_norms),
            "mean_colnorm": col_norms.mean().item(),
            "std_colnorm": col_norms.std().item(),
            "max_colnorm": col_norms.max().item(),
            "min_colnorm": col_norms.min().item(),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def compute_ka_metrics(jacobian: Tensor, k: int = 2, device: str = "cuda") -> dict[str, float]:
    """
    Convenience function to compute all KA metrics.

    Args:
        jacobian: Jacobian matrix [n_rows, n_cols]
        k: Minor size (default 2)
        device: Computation device

    Returns:
        Dictionary with all metrics
    """
    computer = KAMetricsComputer(device=device)
    return computer.compute_all_metrics(jacobian, k=k)


def compare_trained_vs_random(
    jacobian_trained: Tensor, jacobian_random: Tensor, k: int = 2, device: str = "cuda"
) -> dict[str, dict[str, float]]:
    """
    Compare KA metrics between trained and random Jacobians.

    Returns:
        Dictionary with 'trained', 'random', and 'ratio' metrics
    """
    computer = KAMetricsComputer(device=device)

    metrics_trained = computer.compute_all_metrics(jacobian_trained, k=k)
    metrics_random = computer.compute_all_metrics(jacobian_random, k=k)

    # Compute ratios
    ratios = {}
    for key in ["pr_standard", "pr_codebase", "rotation_ratio", "kl_divergence"]:
        t, r = metrics_trained[key], metrics_random[key]
        if r != 0 and not np.isnan(r) and not np.isnan(t):
            ratios[f"{key}_ratio"] = t / r
        else:
            ratios[f"{key}_ratio"] = float("nan")

    return {"trained": metrics_trained, "random": metrics_random, "ratios": ratios}


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing KA Metrics...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    computer = KAMetricsComputer(device=device)

    # Create test Jacobian
    jacobian = torch.randn(100, 200, device=device)

    # Test all metrics
    metrics = computer.compute_all_metrics(jacobian, k=2)

    print("\nMetrics for random Jacobian:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Test column metrics
    col_metrics = computer.compute_column_metrics(jacobian)
    print("\nColumn-based metrics:")
    for key, value in col_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nTest complete!")
