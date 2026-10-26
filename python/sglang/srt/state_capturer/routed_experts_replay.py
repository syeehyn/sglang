"""Routed-experts replay.

With --enable-routed-experts-replay, a return_routed_experts request that is
retracted keeps the routed experts the engine computed for its tokens, and its
re-prefill reuses them. The recomputed KV and the returned routed experts then
follow the experts that sampled each token, which R3 (rollout routing replay)
replays in training. Rows it cannot replay or hold within
--routed-experts-replay-max-tokens are recomputed and counted; no request is
refused. Held rows live in routed_experts_rows.py.

Every decision that changes a forward (hold or drop a snapshot, cut a radix
match) is agreed across TP ranks, so all ranks force the same rows. A request
whose extend forces rows stays out of the radix tree until its routed experts
are collected, so no insert swaps its forced KV for cached KV.
"""

from __future__ import annotations

import contextlib
import logging
import zlib
from typing import TYPE_CHECKING, Any, Iterator, Optional, Sequence

import msgspec
import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.mem_cache.common import match_kv_cache
from sglang.srt.runtime_context import (
    get_exec,
    get_forward,
    get_parallel,
    get_resources,
    get_schedule,
)
from sglang.srt.state_capturer.routed_experts_rows import (
    ForcingBatch,
    HeldRoutes,
    RouteLayout,
    RouteSegment,
    build_forcing_batch,
)
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from array import array

    from sglang.srt.eplb.expert_location import ExpertLocationMetadata
    from sglang.srt.layers.moe.topk import TopKConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
    from sglang.srt.state_capturer.routed_experts import RoutedExpertsCapturer

logger = logging.getLogger(__name__)

_NO_ROWS = HeldRoutes()


class ReplayState(msgspec.Struct, kw_only=True):
    """A return_routed_experts request's held rows and replay counters, kept
    across retracts."""

    held: HeldRoutes = _NO_ROWS
    forced_tokens: int = 0
    retract_snapshots: int = 0
    snapshot_dropped: int = 0
    prefix_mismatch_tokens: int = 0
    # Matched tokens the latest radix match gave up; counted when the extend runs.
    pending_mismatch_tokens: int = 0
    # The latest extend's prefix_len, and len(output_ids) when it was scheduled:
    # until output_ids grows, that extend may be in flight (overlap scheduling),
    # with only the rows below its prefix written.
    extend_prefix_len: int = 0
    extend_output_len: int = 0
    # Whether replay set the request's skip_radix_cache_insert.
    barred_from_tree: bool = False

    def meta_info(self) -> dict:
        return {
            "forced_tokens": self.forced_tokens,
            "retract_snapshots": self.retract_snapshots,
            "snapshot_dropped": self.snapshot_dropped,
            "prefix_mismatch_tokens": self.prefix_mismatch_tokens,
        }


class RoutedExpertsReplay:
    """The scheduler steps that hold a released request's rows and force them."""

    def __init__(
        self,
        *,
        layout: RouteLayout,
        host_table: torch.Tensor,
        req_to_token: torch.Tensor,
        max_tokens: int,
        pin_memory: bool,
        tp_cpu_group: Any = None,
        tp_rank: int = 0,
        overlap_schedule: bool = False,
    ):
        self.layout = layout
        self.layer_ordinals = {
            layer_id: i for i, layer_id in enumerate(layout.moe_layer_ids)
        }
        self.dropped_snapshots = 0
        self.max_tokens = max_tokens
        # Requests holding rows, by id(). Every TP rank holds and releases the
        # same rows in the same scheduler steps, so their total is rank-invariant.
        self._holders: dict[int, Req] = {}
        self._host_table = host_table
        self._pin_memory = pin_memory
        # None on a single TP rank.
        self._tp_cpu_group = tp_cpu_group
        self._tp_rank = tp_rank
        self._overlap_schedule = overlap_schedule
        self._req_to_token = req_to_token
        # Under overlap scheduling, the latest extend's unfinished chunk as
        # (req, start, end): the next round stashes it into the radix tree
        # before its result writes its rows.
        self._inflight_chunk: Optional[tuple[Req, int, int]] = None

    @staticmethod
    def create(
        *,
        model: torch.nn.Module,
        capturer: RoutedExpertsCapturer,
        expert_location: Optional[ExpertLocationMetadata],
        req_to_token: torch.Tensor,
    ) -> RoutedExpertsReplay:
        """Build replay for the loaded model.

        Raises ValueError if the model's gates cannot be forced.
        """
        # Imported here so that importing this module (schedule_batch does)
        # does not load the MoE layers.
        from sglang.srt.layers.moe.topk import TopK

        gates = [
            m
            for m in model.modules()
            if isinstance(m, TopK) and m.topk_config.allow_routed_experts_capture
        ]
        problems = _gate_problems(
            [g.topk_config for g in gates], top_k=capturer.topk_size
        )
        if expert_location is None:
            problems.append("models without expert location metadata")
        if problems:
            raise ValueError(
                "--enable-routed-experts-replay does not support " + "; ".join(problems)
            )
        layout = RouteLayout(
            moe_layer_ids=tuple(sorted({g.layer_id for g in gates})),
            top_k=capturer.topk_size,
            num_logical_experts=expert_location.num_logical_experts,
        )
        parallel = get_parallel()
        return RoutedExpertsReplay(
            layout=layout,
            host_table=capturer.host_cache.buffer,
            req_to_token=req_to_token,
            max_tokens=get_exec().features.routed_experts_replay_max_tokens,
            pin_memory=is_pin_memory_available(),
            tp_cpu_group=parallel.tp_group.cpu_group if parallel.tp_size > 1 else None,
            tp_rank=parallel.tp_rank,
            overlap_schedule=not get_schedule().disable_overlap_schedule,
        )

    def snapshot_on_retract(self, req: Req) -> None:
        """Hold the rows a retracted request computed past its held rows.

        Runs before the retract frees the request's KV slots, which index the
        capture table. Rows past the prefix of an extend that may be in flight are
        not held; rows that would exceed the held-token budget, do not look
        captured or differ across TP ranks are dropped on every rank. Both are
        recomputed unforced.
        """
        state = req.routed_experts_replay_state
        if (
            state is None
            or not req.kv.holds_kv
            or req.to_finish is not None
            # Its re-prefill restarts the output from the prompt embeddings.
            or req.input_embeds is not None
        ):
            return
        # Skip the last kept token's row: under overlap scheduling its forward may
        # still be in flight, with its capture not written yet.
        end = min(
            req.kv.kv_committed_len, len(req.origin_input_ids) + len(req.output_ids) - 1
        )
        if len(req.output_ids) == state.extend_output_len:
            # The extend may be in flight; rows below its prefix are radix hits
            # or earlier extends, already written.
            end = min(end, state.extend_prefix_len)
        start = state.held.end
        if end <= start:
            return
        # Decided on lengths only, so every rank drops the same snapshots without
        # an all-reduce.
        if self._held_tokens() + end - start > self.max_tokens:
            self._drop(
                req, state, start, end, "over --routed-experts-replay-max-tokens"
            )
            return
        slots = self._req_to_token[req.kv.req_pool_idx, start:end].to(
            "cpu", torch.int64
        )
        try:
            rows = self.layout.gather(self._host_table, slots)
            valid = _rows_look_captured(rows)
        except ValueError:
            rows, valid = None, False
        digest = zlib.crc32(rows) if valid else 0
        # One MIN all-reduce gives the min and max digest; they match iff every
        # rank gathered the same rows.
        all_valid, low, neg_high = self._agree_min([int(valid), digest, -digest])
        if not all_valid or low != -neg_high:
            reason = "rows differ across TP ranks" if all_valid else "invalid rows"
            self._drop(req, state, start, end, reason)
            return
        state.held = state.held.extended(RouteSegment(start=start, routes=rows))
        state.retract_snapshots += 1
        self._holders[id(req)] = req

    def release(self, req: Req) -> None:
        """Free the request's held rows and their budget, and let its KV into the
        radix tree again.

        Runs when its routed experts are collected or it is aborted; a request
        that finishes uncollected is released by the next budget check.
        """
        state = req.routed_experts_replay_state
        if state is None:
            return
        self._holders.pop(id(req), None)
        state.held = _NO_ROWS
        if state.barred_from_tree:
            req.skip_radix_cache_insert = False
            state.barred_from_tree = False

    def match_kv_cache(
        self,
        req: Req,
        tree_cache: BasePrefixCache,
        token_ids: array,
        *,
        cow_mamba: bool,
        max_prefix_len: Optional[int],
    ) -> MatchResult:
        """``match_kv_cache``, cut before the first cached row whose experts differ
        from the request's held rows or may not be written yet.

        After a retract, the KV a request matches may have been recomputed by
        another request with other routes; the cut part is then recomputed and
        forced. Under overlap scheduling, the tree can hold a chunk whose forward
        is still in flight; its slots hold earlier rows, often this request's own
        from pages it freed, so the match is also cut there. The probe match does
        no copy-on-write, so the final match copies Mamba state for the kept
        prefix only.
        """
        state = req.routed_experts_replay_state
        if state is None or state.held.end == 0 or req.kv.holds_kv:
            return match_kv_cache(
                req,
                tree_cache,
                token_ids,
                cow_mamba=cow_mamba,
                max_prefix_len=max_prefix_len,
            )
        state.pending_mismatch_tokens = 0
        probe = match_kv_cache(
            req, tree_cache, token_ids, max_prefix_len=max_prefix_len
        )
        matched = req.prefix_len
        checked = min(matched, state.held.end)
        cut = None
        if checked > 0:
            slots = tree_cache.prefix_device_indices(req)[:checked].to(
                "cpu", torch.int64
            )
            first = min(
                self.layout.first_differing_row(self._host_table, slots, state.held),
                self._first_inflight_slot(slots),
            )
            (first,) = self._agree_min([first])
            if first < checked:
                cut = first
        if cut is None and not cow_mamba:
            return probe
        if cut is not None:
            max_prefix_len = cut if max_prefix_len is None else min(max_prefix_len, cut)
        result = match_kv_cache(
            req,
            tree_cache,
            token_ids,
            cow_mamba=cow_mamba,
            max_prefix_len=max_prefix_len,
        )
        state.pending_mismatch_tokens = matched - req.prefix_len
        return result

    def build_batch(
        self,
        *,
        reqs: Sequence[Req],
        prefix_lens: Sequence[int],
        extend_lens: Sequence[int],
    ) -> Optional[ForcingBatch]:
        """An extend batch's forced rows; counts each request's forced rows.

        A request whose extend forces rows is kept out of the radix tree: a
        checkpoint insert would free its forced KV where the tree caches the same
        tokens, and point it at KV computed with other routes. Under overlap
        scheduling, also records the batch's unfinished chunk for match_kv_cache.
        """
        held = []
        self._inflight_chunk = None
        for req, prefix_len, extend_len in zip(
            reqs, prefix_lens, extend_lens, strict=True
        ):
            end = prefix_len + extend_len
            if self._overlap_schedule and end < len(req.full_untruncated_fill_ids):
                self._inflight_chunk = (req, prefix_len, end)
            state = req.routed_experts_replay_state
            if state is None:
                if not req.return_routed_experts:
                    held.append(_NO_ROWS)
                    continue
                state = req.routed_experts_replay_state = ReplayState()
            state.extend_prefix_len = prefix_len
            state.extend_output_len = len(req.output_ids)
            held.append(state.held)
            forced = max(0, min(end, state.held.end) - prefix_len)
            if forced and not req.skip_radix_cache_insert:
                req.skip_radix_cache_insert = True
                state.barred_from_tree = True
            state.forced_tokens += forced
            state.prefix_mismatch_tokens += state.pending_mismatch_tokens
            state.pending_mismatch_tokens = 0
        return build_forcing_batch(
            held=held,
            prefix_lens=prefix_lens,
            extend_lens=extend_lens,
            layout=self.layout,
            pin_memory=self._pin_memory,
        )

    def _held_tokens(self) -> int:
        """Rows held on this rank, after releasing finished requests: a finished
        request never re-prefills, and one whose output was discarded is not
        collected."""
        for req in [r for r in self._holders.values() if r.finished()]:
            self.release(req)
        return sum(
            r.routed_experts_replay_state.held.end for r in self._holders.values()
        )

    def _first_inflight_slot(self, slots: torch.Tensor) -> int:
        """The first ``i`` whose ``slots[i]`` the in-flight chunk computes, or
        ``len(slots)``."""
        if self._inflight_chunk is None:
            return len(slots)
        req, start, end = self._inflight_chunk
        if not req.kv.holds_kv:
            return len(slots)
        computed = self._req_to_token[req.kv.req_pool_idx, start:end].cpu()
        (hits,) = np.isin(slots.numpy(), computed.numpy()).nonzero()
        return int(hits[0]) if len(hits) else len(slots)

    def _drop(
        self, req: Req, state: ReplayState, start: int, end: int, reason: str
    ) -> None:
        state.snapshot_dropped += 1
        self.dropped_snapshots += 1
        if self._tp_rank == 0:
            logger.warning(
                "Routed-experts replay dropped rows [%d, %d) of request %s (%s); "
                "they are recomputed unforced (%d dropped).",
                start,
                end,
                req.rid,
                reason,
                self.dropped_snapshots,
            )

    def _agree_min(self, values: Sequence[int]) -> list[int]:
        """Each value's minimum over the TP ranks: one all-reduce."""
        if self._tp_cpu_group is None:
            return list(values)
        tensor = torch.tensor(values, dtype=torch.int64)
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MIN, group=self._tp_cpu_group
        )
        return tensor.tolist()


def get_global_routed_experts_replay() -> Optional[RoutedExpertsReplay]:
    return get_resources().routed_experts_replay


def set_global_routed_experts_replay(replay: Optional[RoutedExpertsReplay]) -> None:
    get_resources().routed_experts_replay = replay


def replay_meta_info(req: Req) -> dict:
    """The replay counters of a return_routed_experts request."""
    return (req.routed_experts_replay_state or ReplayState()).meta_info()


def _rows_look_captured(rows: np.ndarray) -> bool:
    """False if a row repeats an expert within its top-k, or is all zeros (a slot
    the capture never wrote): such rows are not a gate's choice."""
    if rows.size == 0:
        return True
    if rows.shape[-1] > 1 and (np.diff(np.sort(rows, axis=-1), axis=-1) == 0).any():
        return False
    return bool(rows.reshape(len(rows), -1).any(axis=1).all())


@contextlib.contextmanager
def forcing_forward(batch: ForcingBatch, *, device) -> Iterator[None]:
    """Expose one extend forward's forced rows to every MoE gate it runs."""
    forced = _ForcedForward(
        batch=batch.to_device(device),
        layer_ordinals=get_global_routed_experts_replay().layer_ordinals,
    )
    with get_forward().scoped(routed_experts_forced=forced):
        yield
    forced.check_every_layer_forced()


class _ForcedForward:
    """One extend forward's forced rows on the device, and the gates that took them."""

    def __init__(self, *, batch: ForcingBatch, layer_ordinals: dict[int, int]):
        self._batch = batch
        self._layer_ordinals = layer_ordinals
        self._forced_layers: set[int] = set()

    def force_routing(
        self,
        *,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        router_logits: torch.Tensor,
        layer_id: Optional[int],
        topk_config: TopKConfig,
        packed: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not topk_config.allow_routed_experts_capture:
            return topk_ids, topk_weights
        ordinal = self._layer_ordinals.get(layer_id)
        if ordinal is None or packed:
            raise RuntimeError(
                f"routed-experts replay cannot force the gate of layer {layer_id}: "
                "it is not a captured TopK layer, or it packs its top-k"
            )
        self._forced_layers.add(layer_id)
        return apply_forced_routing(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            router_logits=router_logits,
            forced_ids=self._batch.routes[:, ordinal],
            mask=self._batch.mask,
            scoring_func=topk_config.scoring_func,
            renormalize=topk_config.renormalize,
            routed_scaling_factor=topk_config.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=topk_config.apply_routed_scaling_factor_on_output,
        )

    def check_every_layer_forced(self) -> None:
        # A gate that routes outside select_experts would recompute held rows freely.
        missing = self._layer_ordinals.keys() - self._forced_layers
        if missing:
            raise RuntimeError(
                "routed-experts replay never reached the gates of MoE layers "
                f"{sorted(missing)}"
            )


def _gate_problems(configs: Sequence[TopKConfig], *, top_k: int) -> list[str]:
    problems = set()
    for config in configs:
        num_routed = config.top_k - config.num_fused_shared_experts
        if num_routed != top_k:
            problems.add(
                f"gates whose routed top-k ({num_routed}) differs from the "
                f"captured top-k ({top_k})"
            )
        elif config.sqrtsoftplus_log1p:
            problems.add("sqrtsoftplus gates with log1p")
        elif problem := gate_mirror_problem(
            scoring_func=config.scoring_func,
            has_correction_bias=config.correction_bias is not None,
            has_custom_routing=config.custom_routing_function is not None,
            renormalize=config.renormalize,
            apply_routed_scaling_factor_on_output=config.apply_routed_scaling_factor_on_output,
            num_fused_shared_experts=config.num_fused_shared_experts,
        ):
            problems.add(problem)
    if not configs:
        problems.add("models without TopK MoE gates")
    return sorted(problems)


def gate_mirror_problem(
    *,
    scoring_func: str,
    has_correction_bias: bool,
    has_custom_routing: bool,
    renormalize: bool,
    apply_routed_scaling_factor_on_output: bool,
    num_fused_shared_experts: int,
) -> Optional[str]:
    """Why ``_forced_routing_weights`` cannot reproduce this gate's weights, or None."""
    if has_custom_routing:
        return "custom routing functions"
    if scoring_func == "softmax":
        # The fused gate weighs with the bias; flashinfer_trtllm_routed's raw-logit
        # gate never scales, and weighs raw logits unless it renormalizes.
        if has_correction_bias or apply_routed_scaling_factor_on_output:
            return "softmax gates with a correction bias or routed scaling on output"
        if not renormalize:
            return "softmax gates without renormalize"
    elif scoring_func not in ("sigmoid", "sqrtsoftplus"):
        return f"scoring_func={scoring_func!r}"
    if not renormalize and (
        apply_routed_scaling_factor_on_output or num_fused_shared_experts
    ):
        # Unrenormalized, only the fused gate scales, and a fused shared expert's
        # weight follows the routed experts.
        return "routed scaling or fused shared experts without renormalize"
    return None


def _forced_routing_weights(
    router_logits: torch.Tensor,
    forced_ids: torch.Tensor,
    *,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: Optional[float],
    apply_routed_scaling_factor_on_output: bool,
) -> torch.Tensor:
    """The gate's fp32 weights for ``forced_ids``.

    A correction bias affects only which experts are selected, not their weights.
    """
    logits = router_logits.float()
    if scoring_func == "softmax":
        scores = torch.softmax(logits, dim=-1)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(logits)
    elif scoring_func == "sqrtsoftplus":
        scores = F.softplus(logits).sqrt()
    else:
        raise ValueError(f"replay does not mirror scoring_func={scoring_func!r}")
    weights = scores.gather(1, forced_ids.long())
    if renormalize:
        total = weights.sum(dim=-1, keepdim=True)
        weights = weights / torch.where(total > 0, total, torch.ones_like(total))
        if apply_routed_scaling_factor_on_output and routed_scaling_factor is not None:
            weights = weights * routed_scaling_factor
    return weights


def apply_forced_routing(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    router_logits: torch.Tensor,
    forced_ids: torch.Tensor,
    mask: torch.Tensor,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: Optional[float],
    apply_routed_scaling_factor_on_output: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route masked rows to ``forced_ids``.

    A row whose forced set equals the gate's keeps the gate's ids and weights;
    fused shared-expert columns are untouched.
    """
    if topk_ids.shape[0] != mask.shape[0]:
        raise ValueError(f"{topk_ids.shape[0]} rows but {mask.shape[0]} forcing rows")
    num_routed = forced_ids.shape[1]
    routed_ids, routed_weights = topk_ids[:, :num_routed], topk_weights[:, :num_routed]
    forced = forced_ids.to(routed_ids.dtype)
    same_set = (forced.sort(dim=-1).values == routed_ids.sort(dim=-1).values).all(-1)
    replace = (mask & ~same_set)[:, None]
    weights = _forced_routing_weights(
        router_logits,
        forced,
        scoring_func=scoring_func,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    ).to(routed_weights.dtype)
    ids = torch.where(replace, forced, routed_ids)
    weights = torch.where(replace, weights, routed_weights)
    return (
        torch.cat([ids, topk_ids[:, num_routed:]], dim=1),
        torch.cat([weights, topk_weights[:, num_routed:]], dim=1),
    )
