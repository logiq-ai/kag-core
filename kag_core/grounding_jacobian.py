#!/usr/bin/env python3
"""
Grounding Jacobian Analysis for Hallucination Detection

Measures how much model outputs depend on context tokens, rather than
measuring internal block-to-block Jacobian geometry.

Key insight: Hallucination = ungrounded generation. We should measure whether
the model's output DEPENDS ON the context/evidence rather than measuring
internal block dynamics.

Metrics computed:
1. Context Grounding Jacobian: J = d(logits_t) / d(embeddings_context_tokens)
2. Token-wise Gradient Mass: ||d(logit_top)/d(e_i)||_2 for each context token i
3. Context PR: Participation ratio over context token gradient norms
4. SVD-based PR: Effective rank of grounding Jacobian

Usage:
    from grounding_jacobian import GroundingAnalyzer

    analyzer = GroundingAnalyzer(model_name="google/gemma-3-1b-it")

    # Analyze a QA pair
    metrics = analyzer.analyze_qa_pair(
        question="What is the dosage?",
        answer="50mg twice daily",
        context="The recommended dose is 50mg taken twice daily with food."
    )

    print(f"Context PR: {metrics['context_pr']:.4f}")
    print(f"Evidence sensitivity: {metrics['evidence_sensitivity']:.4f}")
"""

import gc
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GroundingMetrics:
    """Grounding analysis metrics for a single QA pair."""

    # Core grounding metrics (per fact-bearing token, then pooled)
    context_pr: float  # Participation ratio over context token gradient norms
    context_pr_max: float  # Max PR across fact-bearing tokens
    context_pr_mean: float  # Mean PR across fact-bearing tokens

    # Gradient mass metrics
    gradient_mass_concentration: float  # How concentrated is gradient mass on top tokens
    gradient_mass_top5: float  # Fraction of gradient mass in top 5 context tokens

    # SVD-based metrics
    svd_pr: float  # Effective rank of grounding Jacobian (mean over tokens)
    svd_pr_max: float  # Max effective rank
    svd_pr_pos1to5: float = float("nan")  # Mean over positions 1-5 only (skip position 0)
    svd_pr_top3_mean: float = float("nan")  # Mean of top 3 highest SVD PR values (adaptive)
    svd_pr_top20pct_mean: float = float("nan")  # Mean of top 20% highest SVD PR values

    # Evidence sensitivity (optional, from ablation)
    evidence_sensitivity: float | None = None

    # Metadata
    n_context_tokens: int = 0
    n_answer_tokens: int = 0
    n_fact_bearing_tokens: int = 0
    n_non_whitespace_tokens: int = 0  # Tokens that aren't whitespace/punctuation

    # Raw data for debugging
    per_token_context_pr: list[float] | None = None
    per_token_svd_pr: list[float] | None = None
    per_token_is_content: list[bool] | None = None  # True if not whitespace/punctuation

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


class GroundingAnalyzer:
    """
    Compute grounding Jacobians for hallucination detection.

    Measures d(output_logits) / d(context_embeddings) to detect
    whether the model is actually using context to generate answers.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-1b-it",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        top_k_logits: int = 10,  # Only compute gradients for top-k logits
        max_context_tokens: int = 256,  # Limit context size for memory
        max_total_tokens: int = 512,  # Maximum total sequence length
    ):
        """
        Initialize the grounding analyzer.

        Args:
            model_name: HuggingFace model name
            device: Device to run on
            dtype: Model dtype (bfloat16 recommended for memory)
            top_k_logits: Number of top logits to compute gradients for
            max_context_tokens: Maximum context tokens to analyze (for memory)
            max_total_tokens: Maximum total sequence length
        """
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        self.top_k_logits = top_k_logits
        self.max_context_tokens = max_context_tokens
        self.max_total_tokens = max_total_tokens

        # Load model and tokenizer
        print(f"Loading {model_name} for grounding analysis...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

        # Get model config
        self.config = self.model.config
        if hasattr(self.config, "text_config"):
            text_config = self.config.text_config
            self.hidden_size = text_config.hidden_size
            self.vocab_size = text_config.vocab_size
        else:
            self.hidden_size = self.config.hidden_size
            self.vocab_size = self.config.vocab_size

        print(f"Model loaded: hidden_size={self.hidden_size}, vocab_size={self.vocab_size}")

    def _tokenize_with_positions(
        self,
        question: str,
        answer: str,
        context: str | None = None,
    ) -> dict:
        """
        Tokenize QA pair and track context/answer token positions.

        Truncates context if necessary to stay within memory limits.

        Returns:
            Dict with:
                - input_ids: Full tokenized sequence
                - context_start, context_end: Token indices for context
                - answer_start, answer_end: Token indices for full answer segment (including "\nA: ")
                - answer_content_start: Token index where actual answer text begins (after "\nA: ")
                - answer_prefix_len: Number of tokens in the "\nA: " prefix
                - truncated: Whether context was truncated
        """
        # Build prompt components - separate the answer prefix from content
        answer_prefix = "\nA: "
        if context:
            prompt_parts = [
                f"Context: {context}",
                f"\n\nQ: {question}",
                answer_prefix,
                answer,
            ]
        else:
            prompt_parts = [
                f"Q: {question}",
                answer_prefix,
                answer,
            ]

        # Tokenize each part separately to get positions
        tokenized_parts = []
        for part in prompt_parts:
            tokens = self.tokenizer.encode(part, add_special_tokens=False)
            tokenized_parts.append(tokens)

        # Check if we need to truncate context
        truncated = False
        if context and len(tokenized_parts[0]) > self.max_context_tokens:
            tokenized_parts[0] = tokenized_parts[0][: self.max_context_tokens]
            truncated = True

        # Check total length
        total_len = sum(len(p) for p in tokenized_parts) + 1  # +1 for BOS
        if total_len > self.max_total_tokens:
            # Truncate context further if needed
            if context:
                excess = total_len - self.max_total_tokens
                new_context_len = max(10, len(tokenized_parts[0]) - excess)
                tokenized_parts[0] = tokenized_parts[0][:new_context_len]
                truncated = True

        # Combine with BOS token
        full_tokens = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id else []

        # Track positions
        positions = {}
        current_pos = len(full_tokens)

        if context:
            positions["context_start"] = current_pos
            full_tokens.extend(tokenized_parts[0])  # Context
            positions["context_end"] = len(full_tokens)

            full_tokens.extend(tokenized_parts[1])  # Question

            positions["answer_start"] = len(full_tokens)
            full_tokens.extend(tokenized_parts[2])  # Answer prefix "\nA: "
            positions["answer_content_start"] = len(full_tokens)
            positions["answer_prefix_len"] = len(tokenized_parts[2])
            full_tokens.extend(tokenized_parts[3])  # Actual answer content
            positions["answer_end"] = len(full_tokens)
        else:
            positions["context_start"] = None
            positions["context_end"] = None

            full_tokens.extend(tokenized_parts[0])  # Question

            positions["answer_start"] = len(full_tokens)
            full_tokens.extend(tokenized_parts[1])  # Answer prefix "\nA: "
            positions["answer_content_start"] = len(full_tokens)
            positions["answer_prefix_len"] = len(tokenized_parts[1])
            full_tokens.extend(tokenized_parts[2])  # Actual answer content
            positions["answer_end"] = len(full_tokens)

        return {
            "input_ids": torch.tensor([full_tokens], device=self.device),
            **positions,
            "full_text": "".join(prompt_parts),
            "truncated": truncated,
        }

    def _clear_memory(self):
        """Clear GPU memory cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_token_grounding_jacobian(
        self,
        input_ids: torch.Tensor,
        target_token_idx: int,
        context_start: int,
        context_end: int,
    ) -> torch.Tensor:
        """
        Compute grounding Jacobian for a single generated token.

        Computes J = d(logits[target_token_idx, top_k]) / d(embeddings[context_tokens])

        Args:
            input_ids: Full input sequence [1, seq_len]
            target_token_idx: Index of the token to analyze
            context_start: Start index of context tokens
            context_end: End index of context tokens

        Returns:
            Gradient mass tensor [n_context_tokens] - L2 norm of gradient per context token
        """
        seq_len = input_ids.shape[1]
        n_context = context_end - context_start

        # Check sequence length limit
        if seq_len > self.max_total_tokens:
            raise ValueError(f"Sequence too long: {seq_len} > {self.max_total_tokens}")

        # Get embeddings with gradient tracking
        embed_layer = self.model.get_input_embeddings()

        gradient_mass = None
        try:
            # Create embeddings and enable gradient
            with torch.enable_grad():
                embeddings = embed_layer(input_ids).clone().detach()

                # Only track gradients for context tokens
                context_embeds = embeddings[:, context_start:context_end, :].clone()
                context_embeds.requires_grad_(True)

                # Reconstruct full embeddings
                full_embeds = torch.cat(
                    [
                        embeddings[:, :context_start, :],
                        context_embeds,
                        embeddings[:, context_end:, :],
                    ],
                    dim=1,
                )

                # Forward pass to get logits at target position
                outputs = self.model(
                    inputs_embeds=full_embeds,
                    output_hidden_states=False,
                    return_dict=True,
                )

                # Get logits at target position
                logits = outputs.logits[0, target_token_idx - 1, :]  # -1 because causal

                # Get top-k logits
                top_values, top_indices = torch.topk(logits, self.top_k_logits)

                # Sum top logits (we want gradients of the prediction confidence)
                target_logit = top_values.sum()

                # Backward pass
                target_logit.backward()

                # Get gradient w.r.t. context embeddings
                # grad shape: [1, n_context, hidden_size]
                grad = context_embeds.grad

                if grad is None:
                    gradient_mass = torch.zeros(n_context, device=self.device)
                else:
                    # Compute L2 norm per context token
                    # Shape: [n_context]
                    gradient_mass = torch.norm(grad[0], dim=1).detach().clone()

                # Explicit cleanup
                del embeddings, context_embeds, full_embeds, outputs, logits
                del top_values, top_indices, target_logit

        finally:
            # Always clear cache after gradient computation
            self._clear_memory()

        return (
            gradient_mass
            if gradient_mass is not None
            else torch.zeros(n_context, device=self.device)
        )

    def compute_context_pr(self, gradient_mass: torch.Tensor) -> float:
        """
        Compute participation ratio over context token gradient norms.

        PR = L2 / L1 = sqrt(sum ||g_i||^2) / sum ||g_i||

        Higher PR = more concentrated on few tokens
        Lower PR = more diffuse/uniform dependence

        Args:
            gradient_mass: L2 norms of gradients per context token [n_context]

        Returns:
            Participation ratio (higher = more concentrated)
        """
        if len(gradient_mass) == 0:
            return float("nan")

        mass = gradient_mass.float()
        l1 = mass.sum().item()
        l2 = torch.sqrt((mass**2).sum()).item()

        if l1 == 0:
            return float("nan")

        pr = l2 / l1
        return pr

    def compute_svd_pr(self, jacobian: torch.Tensor) -> float:
        """
        Compute SVD-based participation ratio (effective rank).

        PR = L2 / L1 = sqrt(sum sigma_i^2) / sum sigma_i

        Higher = more concentrated singular values (lower effective rank)
        Lower = more uniform singular values (higher effective rank)

        Args:
            jacobian: Grounding Jacobian matrix

        Returns:
            Participation ratio of singular values
        """
        if jacobian.numel() == 0:
            return float("nan")

        try:
            # SVD - use float32 for numerical stability
            jac_f32 = jacobian.float()
            U, S, Vh = torch.linalg.svd(jac_f32, full_matrices=False)

            # PR of singular values: L2/L1
            l1 = S.sum().item()
            l2 = torch.sqrt((S**2).sum()).item()

            if l1 == 0:
                return float("nan")

            return l2 / l1
        except Exception as e:
            warnings.warn(f"SVD failed: {e}")
            return float("nan")

    def compute_full_grounding_jacobian(
        self,
        input_ids: torch.Tensor,
        target_token_idx: int,
        context_start: int,
        context_end: int,
    ) -> torch.Tensor:
        """
        Compute full grounding Jacobian matrix.

        J[i, j] = d(logit_i) / d(embedding_j)

        This is expensive - use compute_token_grounding_jacobian for
        just the gradient mass.

        Returns:
            Jacobian [top_k_logits, n_context * hidden_size]
        """
        n_context = context_end - context_start
        embed_layer = self.model.get_input_embeddings()

        jacobian_rows = []

        with torch.enable_grad():
            embeddings = embed_layer(input_ids).clone().detach()
            context_embeds = embeddings[:, context_start:context_end, :].clone()
            context_embeds.requires_grad_(True)

            full_embeds = torch.cat(
                [
                    embeddings[:, :context_start, :],
                    context_embeds,
                    embeddings[:, context_end:, :],
                ],
                dim=1,
            )

            outputs = self.model(
                inputs_embeds=full_embeds,
                output_hidden_states=False,
                return_dict=True,
            )

            logits = outputs.logits[0, target_token_idx - 1, :]
            top_values, top_indices = torch.topk(logits, self.top_k_logits)

            # Compute gradient for each top logit
            for i in range(self.top_k_logits):
                if context_embeds.grad is not None:
                    context_embeds.grad.zero_()

                top_values[i].backward(retain_graph=True)

                if context_embeds.grad is not None:
                    # Flatten gradient: [n_context * hidden_size]
                    grad_flat = context_embeds.grad[0].reshape(-1).clone()
                    jacobian_rows.append(grad_flat)
                else:
                    jacobian_rows.append(
                        torch.zeros(n_context * self.hidden_size, device=self.device)
                    )

        # Stack to get Jacobian [top_k, n_context * hidden_size]
        jacobian = torch.stack(jacobian_rows, dim=0)
        return jacobian.detach()

    def compute_batch_grounding_jacobians(
        self,
        input_ids: torch.Tensor,
        target_token_indices: list[int],
        context_start: int,
        context_end: int,
        compute_svd: bool = True,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor | None]]:
        """
        Compute grounding Jacobians for multiple tokens with a SINGLE forward pass.

        This is much faster than calling compute_token_grounding_jacobian and
        compute_full_grounding_jacobian separately for each token.

        Args:
            input_ids: Full input sequence [1, seq_len]
            target_token_indices: List of token indices to analyze
            context_start: Start index of context tokens
            context_end: End index of context tokens
            compute_svd: Whether to compute full Jacobian for SVD PR

        Returns:
            Dict mapping token_idx -> (gradient_mass, full_jacobian or None)
            - gradient_mass: [n_context] tensor of L2 gradient norms per context token
            - full_jacobian: [top_k, n_context * hidden_size] if compute_svd, else None
        """
        embed_layer = self.model.get_input_embeddings()
        results = {}

        try:
            with torch.enable_grad():
                # Single forward pass for all tokens
                embeddings = embed_layer(input_ids).clone().detach()
                context_embeds = embeddings[:, context_start:context_end, :].clone()
                context_embeds.requires_grad_(True)

                full_embeds = torch.cat(
                    [
                        embeddings[:, :context_start, :],
                        context_embeds,
                        embeddings[:, context_end:, :],
                    ],
                    dim=1,
                )

                # ONE forward pass
                outputs = self.model(
                    inputs_embeds=full_embeds,
                    output_hidden_states=False,
                    return_dict=True,
                )

                # Process each target token
                for token_idx in target_token_indices:
                    # Get logits at target position (-1 because causal)
                    logits = outputs.logits[0, token_idx - 1, :]
                    top_values, top_indices = torch.topk(logits, self.top_k_logits)

                    # --- Compute gradient mass (sum of top-k logits) ---
                    target_logit = top_values.sum()

                    # Use torch.autograd.grad instead of backward for cleaner handling
                    grad = torch.autograd.grad(
                        target_logit,
                        context_embeds,
                        retain_graph=True,  # Need graph for SVD computation
                        create_graph=False,
                    )[0]

                    # L2 norm per context token [n_context]
                    gradient_mass = torch.norm(grad[0], dim=1).detach().clone()

                    # --- Compute full Jacobian for SVD (if requested) ---
                    full_jacobian = None
                    if compute_svd:
                        jacobian_rows = []

                        # Vectorized: compute gradients for all top-k logits at once
                        for i in range(self.top_k_logits):
                            grad_i = torch.autograd.grad(
                                top_values[i],
                                context_embeds,
                                retain_graph=True,
                                create_graph=False,
                            )[0]
                            grad_flat = grad_i[0].reshape(-1).clone()
                            jacobian_rows.append(grad_flat)

                        full_jacobian = torch.stack(jacobian_rows, dim=0).detach()

                    results[token_idx] = (gradient_mass, full_jacobian)

                # Clean up
                del embeddings, context_embeds, full_embeds, outputs

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # OOM - fall back to sequential computation
                warnings.warn("OOM in batch computation, falling back to sequential")
                self._clear_memory()
                for token_idx in target_token_indices:
                    grad_mass = self.compute_token_grounding_jacobian(
                        input_ids, token_idx, context_start, context_end
                    )
                    full_jac = None
                    if compute_svd:
                        full_jac = self.compute_full_grounding_jacobian(
                            input_ids, token_idx, context_start, context_end
                        )
                    results[token_idx] = (grad_mass, full_jac)
            else:
                raise
        finally:
            self._clear_memory()

        return results

    def compute_evidence_sensitivity(
        self,
        question: str,
        answer: str,
        context: str,
        mask_fraction: float = 0.5,
        n_masks: int = 3,
    ) -> float:
        """
        Compute evidence sensitivity by masking context and measuring logit change.

        Grounded answers should be sensitive to evidence removal.
        Hallucinations should be stable (weren't using evidence anyway).

        Args:
            question: Question text
            answer: Answer text
            context: Context/evidence text
            mask_fraction: Fraction of context tokens to mask
            n_masks: Number of random masks to average over

        Returns:
            Average logit change when masking context (higher = more grounded)
        """
        # Get baseline logits
        tokenized = self._tokenize_with_positions(question, answer, context)
        input_ids = tokenized["input_ids"]
        # Use answer_content_start to skip the "\nA: " prefix
        answer_content_start = tokenized["answer_content_start"]
        answer_end = tokenized["answer_end"]
        context_start = tokenized["context_start"]
        context_end = tokenized["context_end"]

        n_context = context_end - context_start
        n_to_mask = max(1, int(n_context * mask_fraction))

        # Baseline forward pass
        with torch.no_grad():
            baseline_out = self.model(input_ids, return_dict=True)
            # Get logits for answer content tokens only (not the "\nA: " prefix)
            # Logits at position i-1 predict token at position i
            baseline_logits = baseline_out.logits[0, answer_content_start - 1 : answer_end - 1, :]
            # Get top logit per position
            baseline_top_logits = baseline_logits.max(dim=-1).values

        # Masked forward passes
        logit_changes = []
        embed_layer = self.model.get_input_embeddings()

        for _ in range(n_masks):
            # Random mask positions
            mask_indices = torch.randperm(n_context)[:n_to_mask] + context_start

            with torch.no_grad():
                embeddings = embed_layer(input_ids).clone()
                # Zero out masked context tokens
                embeddings[:, mask_indices, :] = 0

                masked_out = self.model(inputs_embeds=embeddings, return_dict=True)
                masked_logits = masked_out.logits[0, answer_content_start - 1 : answer_end - 1, :]
                masked_top_logits = masked_logits.max(dim=-1).values

            # Compute change
            change = (baseline_top_logits - masked_top_logits).abs().mean().item()
            logit_changes.append(change)

        return np.mean(logit_changes)

    def analyze_qa_pair(
        self,
        question: str,
        answer: str,
        context: str | None = None,
        fact_bearing_tokens: list[int] | None = None,
        compute_svd: bool = True,
        compute_sensitivity: bool = False,
    ) -> GroundingMetrics:
        """
        Analyze grounding for a QA pair.

        Args:
            question: Question text
            answer: Answer text
            context: Optional context/evidence text
            fact_bearing_tokens: Optional list of answer token indices to analyze
                                If None, analyzes all answer tokens
            compute_svd: Whether to compute SVD-based PR (slower)
            compute_sensitivity: Whether to compute evidence sensitivity (slowest)

        Returns:
            GroundingMetrics with all computed metrics
        """
        if context is None:
            # No context = no grounding to measure
            return GroundingMetrics(
                context_pr=float("nan"),
                context_pr_max=float("nan"),
                context_pr_mean=float("nan"),
                gradient_mass_concentration=float("nan"),
                gradient_mass_top5=float("nan"),
                svd_pr=float("nan"),
                svd_pr_max=float("nan"),
                n_context_tokens=0,
            )

        # Tokenize with position tracking
        tokenized = self._tokenize_with_positions(question, answer, context)
        input_ids = tokenized["input_ids"]
        context_start = tokenized["context_start"]
        context_end = tokenized["context_end"]
        answer_end = tokenized["answer_end"]
        # answer_content_start is where the actual answer text begins (after "\nA: " prefix)
        answer_content_start = tokenized["answer_content_start"]

        n_context = context_end - context_start
        # n_answer counts only the actual answer content tokens (excluding "\nA: " prefix)
        n_answer = answer_end - answer_content_start

        # Determine which answer tokens to analyze
        if fact_bearing_tokens is not None:
            # Use provided indices (relative to answer content, not the full segment)
            # fact_bearing_tokens are indices into the raw answer text
            tokens_to_analyze = [
                answer_content_start + i
                for i in fact_bearing_tokens
                if answer_content_start + i < answer_end
            ]
        else:
            # Analyze all answer content tokens (excluding "\nA: " prefix)
            tokens_to_analyze = list(range(answer_content_start, answer_end))

        # Limit number of tokens to analyze for memory efficiency
        max_tokens_to_analyze = 10
        if len(tokens_to_analyze) > max_tokens_to_analyze:
            # Sample evenly spaced tokens
            step = len(tokens_to_analyze) // max_tokens_to_analyze
            tokens_to_analyze = tokens_to_analyze[::step][:max_tokens_to_analyze]

        if not tokens_to_analyze:
            return GroundingMetrics(
                context_pr=float("nan"),
                context_pr_max=float("nan"),
                context_pr_mean=float("nan"),
                gradient_mass_concentration=float("nan"),
                gradient_mass_top5=float("nan"),
                svd_pr=float("nan"),
                svd_pr_max=float("nan"),
                n_context_tokens=n_context,
                n_answer_tokens=n_answer,
                n_fact_bearing_tokens=0,
            )

        # Compute metrics for each token
        per_token_pr = []
        per_token_svd_pr = []
        per_token_is_content = []  # True if not whitespace/punctuation
        all_gradient_mass = []

        # Decode tokens to check for whitespace/punctuation (fast, CPU only)
        full_token_ids = input_ids[0].tolist()
        for token_idx in tokens_to_analyze:
            token_text = self.tokenizer.decode([full_token_ids[token_idx]]).strip()
            is_content = len(token_text) > 0 and not all(
                c in " \t\n\r.,;:!?()[]{}\"'" for c in token_text
            )
            per_token_is_content.append(is_content)

        # BATCHED COMPUTATION: Single forward pass for all tokens
        try:
            batch_results = self.compute_batch_grounding_jacobians(
                input_ids,
                tokens_to_analyze,
                context_start,
                context_end,
                compute_svd=compute_svd,
            )

            # Extract results in order
            for token_idx in tokens_to_analyze:
                if token_idx not in batch_results:
                    continue

                grad_mass, full_jac = batch_results[token_idx]
                all_gradient_mass.append(grad_mass)

                # Context PR for this token
                pr = self.compute_context_pr(grad_mass)
                per_token_pr.append(pr)

                # SVD-based PR (if computed)
                if compute_svd and full_jac is not None:
                    svd_pr = self.compute_svd_pr(full_jac)
                    per_token_svd_pr.append(svd_pr)

        except Exception as e:
            warnings.warn(f"Batch computation failed: {e}, falling back to sequential")
            # Fall back to sequential computation
            for token_idx in tokens_to_analyze:
                try:
                    grad_mass = self.compute_token_grounding_jacobian(
                        input_ids, token_idx, context_start, context_end
                    )
                    all_gradient_mass.append(grad_mass)
                    pr = self.compute_context_pr(grad_mass)
                    per_token_pr.append(pr)
                    if compute_svd:
                        full_jac = self.compute_full_grounding_jacobian(
                            input_ids, token_idx, context_start, context_end
                        )
                        svd_pr = self.compute_svd_pr(full_jac)
                        per_token_svd_pr.append(svd_pr)
                except Exception as e2:
                    warnings.warn(f"Failed to analyze token {token_idx}: {e2}")
                    continue

        if not per_token_pr:
            return GroundingMetrics(
                context_pr=float("nan"),
                context_pr_max=float("nan"),
                context_pr_mean=float("nan"),
                gradient_mass_concentration=float("nan"),
                gradient_mass_top5=float("nan"),
                svd_pr=float("nan"),
                svd_pr_max=float("nan"),
                n_context_tokens=n_context,
                n_answer_tokens=n_answer,
                n_fact_bearing_tokens=len(tokens_to_analyze),
            )

        # Aggregate gradient mass
        total_grad_mass = torch.stack(all_gradient_mass).mean(dim=0)

        # Gradient mass concentration metrics
        sorted_mass, _ = torch.sort(total_grad_mass, descending=True)
        total_mass = sorted_mass.sum().item()
        if total_mass > 0:
            top5_mass = sorted_mass[:5].sum().item() / total_mass
            # Gini-like concentration
            cumsum = sorted_mass.cumsum(0) / total_mass
            concentration = 1.0 - 2 * cumsum.mean().item()
        else:
            top5_mass = 0.0
            concentration = 0.0

        # Compute evidence sensitivity if requested
        evidence_sens = None
        if compute_sensitivity:
            evidence_sens = self.compute_evidence_sensitivity(question, answer, context)

        # Pool PR metrics using max and mean
        pr_values = [p for p in per_token_pr if not np.isnan(p)]
        svd_values = [p for p in per_token_svd_pr if not np.isnan(p)]

        # Compute svd_pr_pos1to5: pool over positions 1-5 only (skip position 0)
        # Also filter to content tokens (non-whitespace/punctuation)
        svd_pos1to5_values = []
        for i, (svd_val, is_content) in enumerate(zip(per_token_svd_pr, per_token_is_content)):
            # Position 1-5 means indices 1,2,3,4,5 in the tokens_to_analyze list
            if 1 <= i <= 5 and is_content and not np.isnan(svd_val):
                svd_pos1to5_values.append(svd_val)

        svd_pr_pos1to5 = np.mean(svd_pos1to5_values) if svd_pos1to5_values else float("nan")

        # ADAPTIVE POOLING: Top-k mean (finds signal wherever it is strongest)
        # Filter to content tokens only for adaptive pooling
        svd_content_values = [
            v for v, is_c in zip(per_token_svd_pr, per_token_is_content) if is_c and not np.isnan(v)
        ]

        # Top-3 mean: mean of 3 highest SVD PR values
        if len(svd_content_values) >= 3:
            sorted_svd = sorted(svd_content_values, reverse=True)
            svd_pr_top3_mean = np.mean(sorted_svd[:3])
        else:
            svd_pr_top3_mean = np.mean(svd_content_values) if svd_content_values else float("nan")

        # Top-20% mean: mean of top 20% highest SVD PR values
        if len(svd_content_values) >= 5:  # Need at least 5 to make 20% meaningful
            sorted_svd = sorted(svd_content_values, reverse=True)
            n_top = max(1, len(sorted_svd) // 5)  # 20% = 1/5
            svd_pr_top20pct_mean = np.mean(sorted_svd[:n_top])
        else:
            svd_pr_top20pct_mean = (
                np.mean(svd_content_values) if svd_content_values else float("nan")
            )

        # Count non-whitespace tokens
        n_content = sum(per_token_is_content)

        return GroundingMetrics(
            context_pr=np.max(pr_values) if pr_values else float("nan"),  # Use max as primary
            context_pr_max=np.max(pr_values) if pr_values else float("nan"),
            context_pr_mean=np.mean(pr_values) if pr_values else float("nan"),
            gradient_mass_concentration=concentration,
            gradient_mass_top5=top5_mass,
            svd_pr=np.mean(svd_values) if svd_values else float("nan"),
            svd_pr_max=np.max(svd_values) if svd_values else float("nan"),
            svd_pr_pos1to5=svd_pr_pos1to5,
            svd_pr_top3_mean=svd_pr_top3_mean,
            svd_pr_top20pct_mean=svd_pr_top20pct_mean,
            evidence_sensitivity=evidence_sens,
            n_context_tokens=n_context,
            n_answer_tokens=n_answer,
            n_fact_bearing_tokens=len(tokens_to_analyze),
            n_non_whitespace_tokens=n_content,
            per_token_context_pr=per_token_pr,
            per_token_svd_pr=per_token_svd_pr if compute_svd else None,
            per_token_is_content=per_token_is_content,
        )


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("Testing GroundingAnalyzer...")

    # Test with a simple example
    analyzer = GroundingAnalyzer(model_name="google/gemma-3-1b-it")

    # Test case 1: Grounded answer (context contains the information)
    metrics_grounded = analyzer.analyze_qa_pair(
        question="What is the recommended dosage?",
        answer="50mg twice daily with food",
        context="The medication should be taken at 50mg twice daily with food. "
        "Do not exceed 100mg per day. Store at room temperature.",
        compute_svd=True,
        compute_sensitivity=True,
    )

    print("\n=== GROUNDED ANSWER ===")
    print(f"Context PR (max): {metrics_grounded.context_pr_max:.4f}")
    print(f"Context PR (mean): {metrics_grounded.context_pr_mean:.4f}")
    print(f"Gradient mass top5: {metrics_grounded.gradient_mass_top5:.4f}")
    print(f"SVD PR: {metrics_grounded.svd_pr:.4f}")
    print(f"Evidence sensitivity: {metrics_grounded.evidence_sensitivity:.4f}")

    # Test case 2: Potentially ungrounded (answer not in context)
    metrics_ungrounded = analyzer.analyze_qa_pair(
        question="What is the recommended dosage?",
        answer="75mg three times daily without food",  # Made up!
        context="The medication should be taken at 50mg twice daily with food. "
        "Do not exceed 100mg per day. Store at room temperature.",
        compute_svd=True,
        compute_sensitivity=True,
    )

    print("\n=== UNGROUNDED ANSWER ===")
    print(f"Context PR (max): {metrics_ungrounded.context_pr_max:.4f}")
    print(f"Context PR (mean): {metrics_ungrounded.context_pr_mean:.4f}")
    print(f"Gradient mass top5: {metrics_ungrounded.gradient_mass_top5:.4f}")
    print(f"SVD PR: {metrics_ungrounded.svd_pr:.4f}")
    print(f"Evidence sensitivity: {metrics_ungrounded.evidence_sensitivity:.4f}")

    print("\n=== COMPARISON ===")
    print(
        f"Context PR diff: {metrics_ungrounded.context_pr_max - metrics_grounded.context_pr_max:.4f}"
    )
    print(
        f"Sensitivity diff: {metrics_grounded.evidence_sensitivity - metrics_ungrounded.evidence_sensitivity:.4f}"
    )
