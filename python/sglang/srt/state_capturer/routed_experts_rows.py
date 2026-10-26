"""Routed-experts rows held for replay, and the rows an extend forward forces.

Row ``i`` holds the experts token ``i`` was routed to in the forward that
produced token ``i + 1``, so ``n`` tokens have ``n - 1`` rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import msgspec
import numpy as np
import torch

_TORCH_DTYPES = {np.dtype(np.uint8): torch.uint8, np.dtype(np.int16): torch.int16}
# Capture-table rows gathered per step, which bounds the int32 staging copy.
_GATHER_CHUNK_TOKENS = 4096


class RouteLayout(msgspec.Struct, frozen=True, kw_only=True):
    """The held layers (MoE only) and the narrowest dtype for their expert ids."""

    moe_layer_ids: tuple[int, ...]
    top_k: int
    num_logical_experts: int

    @property
    def dtype(self) -> np.dtype:
        return _route_dtype(self.num_logical_experts)

    def gather(self, table: torch.Tensor, slots: torch.Tensor) -> np.ndarray:
        """Capture-table ``[slots, layers, top_k]`` rows at ``slots`` as held
        ``[n, moe_layers, top_k]`` rows, staged through int32 in chunks."""
        if table.ndim != 3 or table.shape[2] != self.top_k:
            raise ValueError(f"table {tuple(table.shape)} for a top-{self.top_k} gate")
        out = np.empty(
            (len(slots), len(self.moe_layer_ids), self.top_k), dtype=self.dtype
        )
        all_layers = self.moe_layer_ids == tuple(range(table.shape[1]))
        layers = torch.tensor(self.moe_layer_ids, dtype=torch.int64)
        for start in range(0, len(slots), _GATHER_CHUNK_TOKENS):
            chunk = slots[start : start + _GATHER_CHUNK_TOKENS]
            rows = table[chunk] if all_layers else table[chunk[:, None], layers]
            rows = rows.numpy()
            if rows.size and (rows.min() < 0 or rows.max() >= self.num_logical_experts):
                raise ValueError(f"expert ids outside [0, {self.num_logical_experts})")
            out[start : start + len(chunk)] = rows
        return out

    def first_differing_row(
        self, table: torch.Tensor, slots: torch.Tensor, held: HeldRoutes
    ) -> int:
        """The first ``i`` whose experts at ``slots[i]`` differ from held row ``i``
        as a set, or ``len(slots)`` if none does."""
        for start in range(0, len(slots), _GATHER_CHUNK_TOKENS):
            stop = min(start + _GATHER_CHUNK_TOKENS, len(slots))
            try:
                cached = self.gather(table, slots[start:stop])
            except ValueError:
                return start
            own = np.empty_like(cached)
            held.copy_rows(start, stop, out=own)
            differs = np.any(
                np.sort(cached, axis=-1) != np.sort(own, axis=-1), axis=(1, 2)
            )
            if differs.any():
                return start + int(differs.argmax())
        return len(slots)


class RouteSegment(msgspec.Struct, frozen=True, kw_only=True, eq=False):
    """Read-only rows ``[start, start + n)`` of shape ``[n, moe_layers, top_k]``."""

    start: int
    routes: np.ndarray

    def __post_init__(self) -> None:
        if self.start < 0 or self.routes.ndim != 3:
            raise ValueError(f"segment at {self.start} of shape {self.routes.shape}")
        self.routes.flags.writeable = False

    @property
    def end(self) -> int:
        return self.start + len(self.routes)


class HeldRoutes(msgspec.Struct, frozen=True, kw_only=True, eq=False):
    """Rows ``[0, end)`` as a chain of segments, one per retract snapshot.

    Appending never copies the rows already held.
    """

    segments: tuple[RouteSegment, ...] = ()

    def __post_init__(self) -> None:
        expected = 0
        for segment in self.segments:
            if segment.start != expected:
                raise ValueError(f"segment starts at {segment.start}, not {expected}")
            expected = segment.end

    @property
    def end(self) -> int:
        return self.segments[-1].end if self.segments else 0

    def extended(self, segment: RouteSegment) -> HeldRoutes:
        if len(segment.routes) == 0:
            return self
        return HeldRoutes(segments=self.segments + (segment,))

    def copy_rows(self, start: int, stop: int, out: np.ndarray) -> None:
        """Copy rows ``[start, stop)`` into ``out[: stop - start]``."""
        if not 0 <= start <= stop <= self.end:
            raise ValueError(f"rows [{start}, {stop}) are outside [0, {self.end})")
        for s in self.segments:
            lo, hi = max(start, s.start), min(stop, s.end)
            if lo < hi:
                out[lo - start : hi - start] = s.routes[lo - s.start : hi - s.start]


class ForcingBatch(msgspec.Struct, frozen=True, kw_only=True, eq=False):
    """An extend forward's forced rows: ``mask[t]`` forces token row ``t``.

    ``routes`` is ``[tokens, moe_layers, top_k]`` logical expert ids, zero where the
    mask is off, in the held dtype on the host and int32 on the device.
    """

    mask: torch.Tensor
    routes: torch.Tensor

    def padded_to(self, num_tokens: int) -> ForcingBatch:
        """The batch with unforced rows appended (a mixed batch's decode rows)."""
        extra = num_tokens - self.mask.shape[0]
        if extra < 0:
            raise ValueError(f"cannot pad {self.mask.shape[0]} rows to {num_tokens}")
        if extra == 0:
            return self
        pad = self.routes.new_zeros((extra, *self.routes.shape[1:]))
        return ForcingBatch(
            mask=torch.cat([self.mask, self.mask.new_zeros(extra)]),
            routes=torch.cat([self.routes, pad]),
        )

    def to_device(self, device) -> ForcingBatch:
        return ForcingBatch(
            mask=self.mask.to(device, non_blocking=True),
            routes=self.routes.to(device, non_blocking=True).to(torch.int32),
        )


def build_forcing_batch(
    *,
    held: Sequence[HeldRoutes],
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    layout: RouteLayout,
    pin_memory: bool = False,
) -> Optional[ForcingBatch]:
    """The forced rows of an extend forward, or None if it recomputes no held row.

    Request ``i`` computes positions from ``prefix_lens[i]`` at forward rows from
    ``sum(extend_lens[:i])``; each one below its held end is forced.
    """
    num_tokens = sum(extend_lens)
    mask = routes = None
    offset = 0
    for held_routes, prefix_len, extend_len in zip(
        held, prefix_lens, extend_lens, strict=True
    ):
        stop = min(prefix_len + extend_len, held_routes.end)
        if stop > prefix_len:
            if mask is None:
                mask = torch.zeros(num_tokens, dtype=torch.bool, pin_memory=pin_memory)
                routes = torch.zeros(
                    (num_tokens, len(layout.moe_layer_ids), layout.top_k),
                    dtype=_TORCH_DTYPES[layout.dtype],
                    pin_memory=pin_memory,
                )
            n = stop - prefix_len
            mask[offset : offset + n] = True
            held_routes.copy_rows(
                prefix_len, stop, out=routes.numpy()[offset : offset + n]
            )
        offset += extend_len
    return None if mask is None else ForcingBatch(mask=mask, routes=routes)


def _route_dtype(num_logical_experts: int) -> np.dtype:
    """The narrowest dtype that holds expert ids ``[0, num_logical_experts)``."""
    if 0 < num_logical_experts <= 256:
        return np.dtype(np.uint8)
    if 0 < num_logical_experts <= np.iinfo(np.int16).max + 1:
        return np.dtype(np.int16)
    raise ValueError(f"no expert id dtype for {num_logical_experts} experts")
