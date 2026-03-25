"""
KAG Core - Kolmogorov-Arnold Geometry Analysis Toolkit

Core calculation modules for KAG-based neural network analysis.

Modules:
- ka_metrics: Block Jacobian metrics (PR, Rotation Ratio, KL)
- grounding_jacobian: Grounding Jacobian for LLM hallucination detection
- helpers: Utilities for loading/analyzing saved metrics

Quick Start:
============

# Block Jacobians (ViT, MLP analysis)
from kag_core.ka_metrics import KAMetricsComputer, compute_jacobian_efficient

computer = KAMetricsComputer(device='cuda')
jacobian = compute_jacobian_efficient(mlp_forward, hidden_state)
metrics = computer.compute_all_metrics(jacobian, k=2)
# metrics['pr'], metrics['rotation_ratio'], metrics['kl_divergence']

# Grounding Jacobians (LLM hallucination detection)
from kag_core.grounding_jacobian import GroundingAnalyzer

analyzer = GroundingAnalyzer(model_name="google/gemma-3-1b-it")
metrics = analyzer.analyze_qa_pair(question, answer, context, compute_svd=True)
# metrics.svd_pr, metrics.evidence_sensitivity
"""

from .grounding_jacobian import (
    GroundingAnalyzer,
    GroundingMetrics,
)
from .ka_metrics import (
    KAMetricsComputer,
    compute_jacobian_efficient,
    compute_ka_metrics,
)

__all__ = [
    "KAMetricsComputer",
    "compute_jacobian_efficient",
    "compute_ka_metrics",
    "GroundingAnalyzer",
    "GroundingMetrics",
]
