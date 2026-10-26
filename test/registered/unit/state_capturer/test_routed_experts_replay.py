"""Routed-experts replay with fake capture tables, and its forced top-k."""

import contextlib
import datetime
import unittest
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.distributed
import torch.multiprocessing

from sglang.srt.environ import envs
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import (
    TopK,
    TopKConfig,
    biased_topk_impl,
    fused_topk_softmax_torch_raw_logits,
    fused_topk_torch_native,
)
from sglang.srt.managers import schedule_batch
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components import batch_result_processor
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import checkpoint_kv_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_forward
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.state_capturer.routed_experts_replay import (
    ReplayState,
    RoutedExpertsReplay,
    _forced_routing_weights,
    _rows_look_captured,
    apply_forced_routing,
    forcing_forward,
    gate_mirror_problem,
    replay_meta_info,
    set_global_routed_experts_replay,
)
from sglang.srt.state_capturer.routed_experts_rows import (
    ForcingBatch,
    HeldRoutes,
    RouteLayout,
    RouteSegment,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

# Layer 0 is dense: its capture column stays 0 and is never held.
LAYOUT = RouteLayout(moe_layer_ids=(1, 2), top_k=2, num_logical_experts=8)
GATE = TopKConfig(top_k=2, renormalize=True, scoring_func="softmax")
# The gate itself, run eagerly: its torch.compile wrapper needs a C++ compiler.
_biased_topk = biased_topk_impl._torchdynamo_orig_callable


class _Engine:
    """A capture table and one softmax top-2 gate per MoE layer, whose logits depend
    on the weight version and the position. Every computed position writes its
    routes at a fresh KV slot, as the capturer does; slots are never reused.

    With a ``tree``, request rows and KV slots come from the tree's pools."""

    def __init__(self, tp_cpu_group=None, tree=None, overlap=False, max_tokens=4096):
        self.host = torch.zeros((512, 3, 2), dtype=torch.int32)
        self.req_to_token = (
            torch.zeros((9, 64), dtype=torch.int64)
            if tree is None
            else tree.req_to_token_pool.req_to_token
        )
        self._allocator = None if tree is None else tree.token_to_kv_pool_allocator
        self.replay = RoutedExpertsReplay(
            layout=LAYOUT,
            host_table=self.host,
            req_to_token=self.req_to_token,
            max_tokens=max_tokens,
            pin_memory=False,
            tp_cpu_group=tp_cpu_group,
            overlap_schedule=overlap,
        )
        self.version = 0
        self._next_slot = 0
        self._next_pool_idx = 1
        self._next_rid = 0

    def gate(self, positions, layer, version=None):
        version = self.version if version is None else version
        generator = torch.Generator().manual_seed(version * 1000 + layer)
        logits = torch.randn((64, 8), generator=generator)[positions]
        weights, ids = torch.softmax(logits, dim=-1).topk(2)
        return ids.int(), weights / weights.sum(-1, keepdim=True), logits

    def request(self, tokens, *, return_routed_experts=True):
        self._next_rid += 1
        return Req(
            f"rid-{self._next_rid}",
            "",
            array("q", tokens),
            SamplingParams(),
            return_routed_experts=return_routed_experts,
        )

    def extend(self, req, start, stop, *, sample=True):
        """A scheduled extend of positions [start, stop), forced where held; its
        result (one sampled token) is processed unless ``sample`` is False."""
        batch = self.replay.build_batch(
            reqs=[req], prefix_lens=[start], extend_lens=[stop - start]
        )
        slots = self._alloc(req, start, stop)
        with (
            contextlib.nullcontext()
            if batch is None
            else forcing_forward(batch, device="cpu")
        ):
            for layer in LAYOUT.moe_layer_ids:
                ids, weights, logits = self.gate(torch.arange(start, stop), layer)
                if (forced := get_forward().routed_experts_forced) is not None:
                    ids, weights = forced.force_routing(
                        topk_ids=ids,
                        topk_weights=weights,
                        router_logits=logits,
                        layer_id=layer,
                        topk_config=GATE,
                        packed=False,
                    )
                self.host[slots, layer] = ids
        req.extend_end = stop
        if sample:
            req.output_ids.append(100 + stop)
        return batch

    def decode(self, req, steps):
        """Decode steps, which compute only new positions and are never forced."""
        for _ in range(steps):
            position = len(req.origin_input_ids) + len(req.output_ids) - 1
            slots = self._alloc(req, position, position + 1)
            for layer in LAYOUT.moe_layer_ids:
                self.host[slots, layer] = self.gate(torch.tensor([position]), layer)[0]
            req.output_ids.append(100 + position)

    def share(self, req, donor, n):
        """Start ``req`` on a radix hit of ``donor``'s first ``n`` KV slots."""
        self.start_on(req, self.req_to_token[donor.kv.req_pool_idx, :n])

    def start_on(self, req, slots):
        """Start ``req`` on a radix hit of ``slots``, written as prepare_for_extend
        writes a matched prefix."""
        req.kv.req_pool_idx = self._next_pool_idx
        self._next_pool_idx += 1
        self.req_to_token[req.kv.req_pool_idx, : len(slots)] = slots
        req.kv.kv_committed_len = len(slots)

    def retract(self, req):
        self.replay.snapshot_on_retract(req)
        req.kv.req_pool_idx = None
        req.kv.kv_committed_len = 0

    def collect(self, req):
        """Rows [0, n - 1) as the result processor collects them at finish."""
        return self.rows(req, 0, len(req.origin_input_ids) + len(req.output_ids) - 1)

    def rows(self, req, start, stop):
        slots = self.req_to_token[req.kv.req_pool_idx, start:stop]
        return LAYOUT.gather(self.host, slots)

    def gate_rows(self, start, stop, version):
        positions = torch.arange(start, stop)
        return np.stack(
            [self.gate(positions, layer, version)[0].numpy() for layer in (1, 2)], 1
        )

    def _alloc(self, req, start, stop):
        if req.kv.req_pool_idx is None:
            req.kv.req_pool_idx = self._next_pool_idx
            self._next_pool_idx += 1
        if self._allocator is None:
            slots = torch.arange(self._next_slot, self._next_slot + stop - start)
            self._next_slot += stop - start
        else:
            slots = self._allocator.alloc(stop - start)
        self.req_to_token[req.kv.req_pool_idx, start:stop] = slots
        req.kv.kv_committed_len = stop
        return slots


def _collect(req):
    """The result processor collecting a finished request's routed experts."""
    capturer = SimpleNamespace(get_topk=lambda **_: None)
    with patch.object(
        batch_result_processor, "get_global_experts_capturer", return_value=capturer
    ):
        SchedulerBatchResultProcessor._maybe_collect_routed_experts(
            SimpleNamespace(req_to_token_pool=None), req
        )


def _abort(req):
    """The scheduler dropping an aborted request's state, e.g. a queued retract."""
    Scheduler._release_aborted_request(
        SimpleNamespace(tree_cache=SimpleNamespace(finish=lambda *_: None)), req
    )


def _abort_on_retract(engine, req):
    """retract_decode aborting a request it cannot requeue (the last one out of
    memory, or a beam group) after releasing its KV; update_running_batch reports
    the abort, and the request never finishes."""

    def retract_decode():
        req.to_finish = FINISH_ABORT("out of memory")
        engine.retract(req)
        return [], 0.5, [req]

    scheduler = Mock(decode_offload_manager=None)
    scheduler._reqs_finishing_inflight.return_value = []
    scheduler.token_to_kv_pool_allocator.available_size.return_value = 0
    scheduler.new_token_ratio_tracker.current = 0.5
    scheduler.tree_cache.req_to_token_pool = SimpleNamespace()
    scheduler.metrics_reporter.enable_metrics = False
    batch = Mock(retract_decode=retract_decode)
    batch.batch_size.return_value = 1
    batch.check_decode_mem.return_value = False
    batch.is_empty.return_value = False
    Scheduler.update_running_batch(scheduler, batch)
    scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()


def _same_sets(a, b):
    return np.array_equal(np.sort(a, axis=-1), np.sort(b, axis=-1))


def _counters(forced, snapshots, dropped=0, mismatch=0):
    return {
        "forced_tokens": forced,
        "retract_snapshots": snapshots,
        "snapshot_dropped": dropped,
        "prefix_mismatch_tokens": mismatch,
    }


class TestRoutedExpertsReplay(CustomTestCase):
    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology())
        self.engine = _Engine()
        set_global_routed_experts_replay(self.engine.replay)

    def test_retract_walk_through(self):
        """A turn retracted mid-decode re-prefills under new weights with its own
        first routes, so the collected rows are the ones that sampled it."""
        engine = self.engine
        req = engine.request(list(range(10, 16)))
        self.assertIsNone(engine.extend(req, 0, 6))  # nothing held yet
        engine.decode(req, 3)
        before = engine.rows(req, 0, 9)

        engine.retract(req)  # 6 prompt + 4 output tokens: rows [0, 9) were kept
        self.assertEqual(req.routed_experts_replay_state.held.end, 9)
        engine.version = 1
        batch = engine.extend(req, 0, 10)
        self.assertEqual(int(batch.mask.sum()), 9)
        self.assertTrue(_same_sets(engine.rows(req, 0, 9), before))
        self.assertFalse(_same_sets(engine.gate_rows(0, 9, version=1), before))
        engine.decode(req, 2)
        self.assertTrue(_same_sets(engine.collect(req)[:9], before))
        self.assertEqual(replay_meta_info(req), _counters(forced=9, snapshots=1))

    def test_rows_read_through_prefix_slots_are_held(self):
        """The snapshot starts at 0: prefix rows another request computed (a radix
        hit) are held with the request's own."""
        engine = self.engine
        donor = engine.request(list(range(20, 28)))
        engine.extend(donor, 0, 8)
        req = engine.request(list(range(20, 28)) + [5, 6])
        engine.share(req, donor, 8)
        engine.extend(req, 8, 10)
        engine.decode(req, 2)
        engine.retract(req)
        held = np.empty((12, 2, 2), dtype=np.uint8)
        req.routed_experts_replay_state.held.copy_rows(0, 12, out=held)
        np.testing.assert_array_equal(held[:8], engine.rows(donor, 0, 8))

        engine.version = 1
        self.assertEqual(int(engine.extend(req, 0, 13).mask.sum()), 12)
        self.assertTrue(_same_sets(engine.rows(req, 0, 12), held))

    def test_snapshot_holds_only_written_rows(self):
        """A retract before the first token holds the rows of its radix hit, which
        an earlier request sampled with; rows of an extend or decode that may be
        in flight are not held."""
        engine = self.engine
        donor = engine.request(list(range(20, 28)))
        engine.extend(donor, 0, 8)
        req = engine.request(list(range(20, 28)) + [5, 6, 7, 8])
        engine.share(req, donor, 8)
        engine.extend(req, 8, 12, sample=False)  # in flight: nothing sampled yet
        engine.retract(req)
        state = req.routed_experts_replay_state
        self.assertEqual((state.held.end, state.retract_snapshots), (8, 1))
        held = np.empty((8, 2, 2), dtype=np.uint8)
        state.held.copy_rows(0, 8, out=held)
        np.testing.assert_array_equal(held, engine.rows(donor, 0, 8))

        engine.extend(req, 0, 12)
        engine.decode(req, 2)  # positions 12 and 13; 3 tokens kept
        in_flight = engine._alloc(req, 14, 15)
        engine.host[in_flight] = 200  # an expert id no gate produces
        engine.retract(req)
        self.assertEqual((state.held.end, state.retract_snapshots), (14, 2))

        # A re-prefill in flight holds only the rows below its prefix.
        state.held = HeldRoutes()
        engine.version = 1
        engine.extend(req, 4, 15, sample=False)
        engine.retract(req)
        self.assertEqual((state.held.end, state.snapshot_dropped), (4, 0))

    def test_repeated_retracts_append_new_rows(self):
        engine = self.engine
        req = engine.request(list(range(6)))
        engine.extend(req, 0, 6)
        engine.decode(req, 2)
        engine.retract(req)  # rows [0, 8)
        engine.version = 1
        engine.extend(req, 0, 9)
        engine.decode(req, 2)
        before = engine.rows(req, 0, 11)
        engine.retract(req)  # rows [8, 11)
        held = req.routed_experts_replay_state.held
        self.assertEqual([(s.start, s.end) for s in held.segments], [(0, 8), (8, 11)])

        engine.version = 2
        self.assertEqual(int(engine.extend(req, 0, 12).mask.sum()), 11)
        self.assertTrue(_same_sets(engine.rows(req, 0, 11), before))
        self.assertEqual(replay_meta_info(req), _counters(forced=19, snapshots=2))

    def test_chunked_re_prefill_forces_every_chunk(self):
        """A re-prefill split into chunks forces each chunk's held rows."""
        engine = self.engine
        req = engine.request(list(range(10, 16)))
        engine.extend(req, 0, 6)
        engine.decode(req, 3)
        before = engine.rows(req, 0, 9)
        engine.retract(req)
        engine.version = 1
        engine.extend(req, 0, 5, sample=False)
        engine.extend(req, 5, 10)
        self.assertTrue(_same_sets(engine.rows(req, 0, 9), before))
        self.assertEqual(replay_meta_info(req), _counters(forced=9, snapshots=1))

    def test_invalid_rows_are_dropped_and_recomputed(self):
        """Rows the capture cannot have written are dropped and counted; the
        request is never refused and its tokens are recomputed unforced."""
        for spoil in ("repeated", "out_of_range"):
            with self.subTest(spoil):
                engine = _Engine()
                set_global_routed_experts_replay(engine.replay)
                req = engine.request(list(range(6)))
                engine.extend(req, 0, 6)
                engine.decode(req, 2)
                slot = engine.req_to_token[req.kv.req_pool_idx, 3]
                engine.host[slot, 2] = 7 if spoil == "repeated" else 200
                engine.retract(req)
                self.assertEqual(req.routed_experts_replay_state.held.end, 0)
                self.assertIsNone(engine.extend(req, 0, 9))
                self.assertEqual(
                    replay_meta_info(req), _counters(forced=0, snapshots=0, dropped=1)
                )
                self.assertEqual(engine.replay.dropped_snapshots, 1)

    def test_snapshots_over_the_budget_are_dropped(self):
        """A retract whose new rows would exceed --routed-experts-replay-max-tokens
        is dropped on lengths alone, with no all-reduce, and counted; rows held
        earlier stay held and forced."""
        engine = _Engine(max_tokens=12)
        set_global_routed_experts_replay(engine.replay)
        first = engine.request(list(range(6)))
        engine.extend(first, 0, 6)
        engine.decode(first, 2)
        engine.retract(first)  # rows [0, 8)
        second = engine.request(list(range(20, 26)))
        engine.extend(second, 0, 6)
        engine.decode(second, 2)
        with patch.object(engine.replay, "_agree_min", side_effect=AssertionError):
            engine.retract(second)  # 8 more rows
        self.assertEqual(
            replay_meta_info(second), _counters(forced=0, snapshots=0, dropped=1)
        )

        engine.extend(first, 0, 9)
        engine.decode(first, 4)
        engine.retract(first)  # rows [8, 13)
        self.assertEqual(first.routed_experts_replay_state.held.end, 8)
        self.assertEqual(
            replay_meta_info(first), _counters(forced=8, snapshots=1, dropped=1)
        )
        self.assertEqual(engine.replay.dropped_snapshots, 2)

    def test_released_rows_free_the_budget(self):
        """Held rows leave the budget when the routed experts are collected, when
        the request is aborted (also by a retract that cannot requeue it), and,
        at the next check, when it finished uncollected (a discarded output)."""

        def finish(req):
            req.finished_reason = FINISH_ABORT()

        def abort_on_retract(req):
            # Re-prefilled and decoding when a later retract aborts it.
            engine.extend(req, 0, 9)
            engine.decode(req, 1)
            _abort_on_retract(engine, req)

        for release in (_collect, _abort, finish, abort_on_retract):
            with self.subTest(release.__name__):
                engine = _Engine(max_tokens=8)
                set_global_routed_experts_replay(engine.replay)
                holder = engine.request(list(range(6)))
                engine.extend(holder, 0, 6)
                engine.decode(holder, 2)
                engine.retract(holder)  # rows [0, 8) fill the budget
                release(holder)
                req = engine.request(list(range(20, 26)))
                engine.extend(req, 0, 6)
                engine.decode(req, 2)
                engine.retract(req)
                self.assertEqual(holder.routed_experts_replay_state.held.end, 0)
                self.assertEqual(
                    replay_meta_info(req), _counters(forced=0, snapshots=1)
                )

    def test_release_req_snapshots_before_freeing_the_slots(self):
        engine = self.engine
        req = engine.request(list(range(6)))
        engine.extend(req, 0, 6)
        engine.decode(req, 2)

        def release_kv_cache(req, *_, **__):
            req.kv.req_pool_idx = None

        with (
            patch.object(
                schedule_batch, "release_kv_cache", side_effect=release_kv_cache
            ),
            patch.object(schedule_batch, "evict_from_tree_cache"),
        ):
            schedule_batch.release_req(
                req=req,
                remaing_req_count=1,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=None,
                tree_cache=None,
                hisparse_coordinator=None,
            )
        self.assertEqual(req.routed_experts_replay_state.held.end, 8)

    def test_rows_look_captured(self):
        rows = np.array([[[1, 2], [3, 0]], [[4, 5], [6, 7]]], dtype=np.uint8)
        self.assertTrue(_rows_look_captured(rows))
        self.assertTrue(_rows_look_captured(rows[:0]))
        rows[1, 0] = [5, 5]
        self.assertFalse(_rows_look_captured(rows))
        # Top-1: an all-zero row is a slot the capture never wrote.
        top1 = np.array([[[0], [3]], [[0], [0]]], dtype=np.uint8)
        self.assertFalse(_rows_look_captured(top1))
        self.assertTrue(_rows_look_captured(top1[:1]))

    def test_requests_that_hold_nothing(self):
        engine = self.engine
        plain = engine.request(list(range(6)), return_routed_experts=False)
        embeds = engine.request(list(range(6)))
        embeds.input_embeds = [[0.0]] * 6
        aborted = engine.request(list(range(6)))
        for req in (plain, embeds, aborted):
            engine.extend(req, 0, 6)
            engine.decode(req, 2)
        aborted.to_finish = SimpleNamespace()
        for req in (plain, embeds, aborted):
            engine.retract(req)
        self.assertIsNone(plain.routed_experts_replay_state)
        self.assertEqual(replay_meta_info(plain), _counters(forced=0, snapshots=0))
        for req in (embeds, aborted):
            self.assertEqual(req.routed_experts_replay_state.held.end, 0)

    def test_unchanged_set_keeps_the_gate_order_and_weights(self):
        """A held row whose experts the gate picks again in another order moves no
        bit, so forcing composes with captures that sort each row."""
        replay = self.engine.replay
        ids, weights, logits = self.engine.gate(torch.arange(4), layer=1)
        req = self.engine.request([1, 2, 3, 4, 5])
        held = ids.flip(-1).numpy().astype(np.uint8)
        req.routed_experts_replay_state = ReplayState(
            held=HeldRoutes(
                segments=(RouteSegment(start=0, routes=np.stack([held, held], 1)),)
            )
        )
        batch = replay.build_batch(reqs=[req], prefix_lens=[0], extend_lens=[4])
        with forcing_forward(batch, device="cpu"):
            forced = get_forward().routed_experts_forced
            for layer in LAYOUT.moe_layer_ids:
                out_ids, out_weights = forced.force_routing(
                    topk_ids=ids,
                    topk_weights=weights,
                    router_logits=logits,
                    layer_id=layer,
                    topk_config=GATE,
                    packed=False,
                )
                self.assertTrue(torch.equal(out_ids, ids))
                self.assertTrue(torch.equal(out_weights, weights))

    def test_select_experts_forces_before_capture_and_remap(self):
        logits = torch.randn((3, 8), generator=torch.Generator().manual_seed(0))
        routes = torch.tensor(
            [[[0, 1]] * 2, [[2, 3]] * 2, [[4, 5]] * 2], dtype=torch.uint8
        )
        batch = ForcingBatch(mask=torch.tensor([True, False, True]), routes=routes)
        config = TopKConfig(top_k=2, renormalize=True, torch_native=True)
        own = topk_module.select_experts(torch.empty(3, 1), logits, config, layer_id=1)
        expected = [[0, 1], sorted(own.topk_ids[1].tolist()), [4, 5]]
        with (
            patch.object(
                topk_module,
                "_post_process_topk_ids",
                wraps=topk_module._post_process_topk_ids,
            ) as post,
            forcing_forward(batch, device="cpu"),
        ):
            for layer in LAYOUT.moe_layer_ids:
                out = topk_module.select_experts(
                    torch.empty(3, 1), logits, config, layer_id=layer
                )
        # Capture and the EPLB remap both run in _post_process_topk_ids.
        self.assertEqual(
            [sorted(r) for r in post.call_args.kwargs["topk_ids"].tolist()], expected
        )
        self.assertEqual([sorted(r) for r in out.topk_ids.tolist()], expected)

    def test_unforced_layer_raises(self):
        req = self.engine.request([1, 2, 3])
        req.routed_experts_replay_state = ReplayState(
            held=HeldRoutes(
                segments=(
                    RouteSegment(start=0, routes=np.zeros((2, 2, 2), dtype=np.uint8)),
                )
            )
        )
        batch = self.engine.replay.build_batch(
            reqs=[req], prefix_lens=[0], extend_lens=[3]
        )
        ids, weights, logits = self.engine.gate(torch.arange(3), layer=1)
        with self.assertRaisesRegex(RuntimeError, r"MoE layers \[2\]"):
            with forcing_forward(batch, device="cpu"):
                get_forward().routed_experts_forced.force_routing(
                    topk_ids=ids,
                    topk_weights=weights,
                    router_logits=logits,
                    layer_id=1,
                    topk_config=GATE,
                    packed=False,
                )


def _pools(page_size):
    """CPU request rows and KV slots, as the scheduler's tree holds them."""
    return dict(
        req_to_token_pool=ReqToTokenPool(
            size=8, max_context_len=64, device="cpu", enable_memory_saver=False
        ),
        token_to_kv_pool_allocator=TokenToKVPoolAllocator(
            size=256,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=MHATokenToKVPool(
                size=256,
                page_size=page_size,
                dtype=torch.bfloat16,
                head_num=1,
                head_dim=8,
                layer_num=1,
                device="cpu",
                enable_memory_saver=False,
            ),
            need_sort=False,
        ),
        page_size=page_size,
    )


def _radix_tree(page_size):
    return RadixCache(CacheInitParams(disable=False, **_pools(page_size)))


def _unified_tree(page_size):
    """The tree the scheduler builds by default."""
    with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
        return UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                tree_components=(ComponentType.FULL,),
                **_pools(page_size),
            )
        )


def _radix_with_held_request(engine, *, differ_at=None, tree=None):
    """A request holding rows [0, 11) of 12 tokens, and a tree (with pages of 2
    tokens) caching its first 10 tokens at other slots whose rows are its held
    rows, except ``differ_at``."""
    tokens = list(range(30, 42))
    req = engine.request(tokens)
    engine.extend(req, 0, 12)
    held = engine.rows(req, 0, 11)
    req.routed_experts_replay_state.held = HeldRoutes(
        segments=(RouteSegment(start=0, routes=held),)
    )
    req.kv.req_pool_idx = None
    cached = torch.arange(300, 310)
    engine.host[cached, 1:3] = torch.tensor(held[:10], dtype=torch.int32)
    if differ_at is not None:
        row = engine.host[cached[differ_at], 1]
        row.copy_((row + 1) % LAYOUT.num_logical_experts)
    if tree is None:
        tree = RadixCache.create_simulated(page_size=2)
    tree.insert(
        InsertParams(key=RadixKey(token_ids=array("q", tokens[:10])), value=cached)
    )
    return req, tree


class TestRadixMatch(CustomTestCase):
    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology())

    def _engine(self, tree=None, overlap=False):
        engine = _Engine(tree=tree, overlap=overlap)
        set_global_routed_experts_replay(engine.replay)
        return engine

    def test_match_is_cut_where_cached_routes_differ(self):
        """A cached prefix recomputed with other routes (e.g. by another request
        after the prompt was evicted) is cut at the page of the first differing
        row; the rest is recomputed and forced."""
        for tree_factory in (RadixCache.create_simulated, _unified_tree):
            for differ_at, prefix_len in ((5, 4), (0, 0)):
                with self.subTest(tree_factory.__name__, differ_at=differ_at):
                    engine = self._engine()
                    req, tree = _radix_with_held_request(
                        engine, differ_at=differ_at, tree=tree_factory(page_size=2)
                    )
                    req.init_next_round_input(tree)
                    self.assertEqual(req.prefix_len, prefix_len)
                    forced = 11 - prefix_len
                    batch = engine.replay.build_batch(
                        reqs=[req],
                        prefix_lens=[prefix_len],
                        extend_lens=[12 - prefix_len],
                    )
                    self.assertEqual(int(batch.mask.sum()), forced)
                    self.assertEqual(
                        replay_meta_info(req),
                        _counters(forced=forced, snapshots=0, mismatch=10 - prefix_len),
                    )

    def test_chunked_re_prefill_counts_each_token_once(self):
        """Forced tokens add up over the chunks; the radix cut counts once."""
        engine = self._engine()
        req, tree = _radix_with_held_request(engine, differ_at=5)
        req.init_next_round_input(tree)
        self.assertEqual(req.prefix_len, 4)
        b1 = engine.replay.build_batch(reqs=[req], prefix_lens=[4], extend_lens=[4])
        b2 = engine.replay.build_batch(reqs=[req], prefix_lens=[8], extend_lens=[4])
        self.assertEqual((int(b1.mask.sum()), int(b2.mask.sum())), (4, 3))
        self.assertEqual(
            replay_meta_info(req), _counters(forced=7, snapshots=0, mismatch=6)
        )

    def test_match_is_cut_at_a_chunk_in_flight(self):
        """Under overlap scheduling, a chunk is stashed into the tree before its
        forward's rows are written; its slots may still hold the held rows (pages
        the retracted request freed), so the match is cut at the chunk."""
        for tree_factory in (_radix_tree, _unified_tree):
            for overlap, prefix_len in ((True, 4), (False, 8)):
                with self.subTest(tree_factory.__name__, overlap=overlap):
                    tree = tree_factory(page_size=2)
                    engine = self._engine(tree, overlap=overlap)
                    tokens = list(range(30, 42))
                    req = engine.request(tokens)
                    engine.extend(req, 0, 12, sample=False)
                    held = engine.rows(req, 0, 11)
                    req.routed_experts_replay_state.held = HeldRoutes(
                        segments=(RouteSegment(start=0, routes=held),)
                    )
                    req.kv.req_pool_idx = None

                    sibling = engine.request(tokens)
                    sibling.init_next_round_input(tree)
                    engine.start_on(sibling, tree.prefix_device_indices(sibling))
                    engine.extend(sibling, 0, 4, sample=False)
                    checkpoint_kv_cache(sibling, tree)
                    # The next chunk's forward is in flight: its rows are not
                    # written, and its slots still hold the held rows.
                    engine.replay.build_batch(
                        reqs=[sibling], prefix_lens=[4], extend_lens=[4]
                    )
                    slots = engine._alloc(sibling, 4, 8)
                    engine.host[slots, 1:3] = torch.tensor(held[4:8], dtype=torch.int32)
                    sibling.extend_end = 8
                    checkpoint_kv_cache(sibling, tree)

                    req.init_next_round_input(tree)
                    self.assertEqual(req.prefix_len, prefix_len)
                    engine.replay.build_batch(
                        reqs=[req],
                        prefix_lens=[prefix_len],
                        extend_lens=[12 - prefix_len],
                    )
                    self.assertEqual(
                        replay_meta_info(req),
                        _counters(
                            forced=11 - prefix_len, snapshots=0, mismatch=8 - prefix_len
                        ),
                    )

    def test_matching_routes_keep_the_match(self):
        for tree_factory in (RadixCache.create_simulated, _unified_tree):
            with self.subTest(tree_factory.__name__):
                engine = self._engine()
                req, tree = _radix_with_held_request(
                    engine, tree=tree_factory(page_size=2)
                )
                req.init_next_round_input(tree)
                self.assertEqual(req.prefix_len, 10)
                engine.replay.build_batch(reqs=[req], prefix_lens=[10], extend_lens=[2])
                state = req.routed_experts_replay_state
                self.assertEqual(state.prefix_mismatch_tokens, 0)

                plain, tree = _radix_with_held_request(
                    engine, differ_at=0, tree=tree_factory(page_size=2)
                )
                plain.routed_experts_replay_state = None
                plain.init_next_round_input(tree)
                self.assertEqual(plain.prefix_len, 10)

    def test_mamba_copy_on_write_runs_once_on_the_kept_prefix(self):
        """The probe match does no copy-on-write; the final match copies Mamba
        state for the kept prefix only."""
        for differ_at, limit, prefix_len in ((None, 12, 10), (5, 5, 4)):
            with self.subTest(differ_at=differ_at):
                engine = self._engine()
                req, tree = _radix_with_held_request(engine, differ_at=differ_at)
                with patch.object(tree, "match_prefix", wraps=tree.match_prefix) as spy:
                    req.init_next_round_input(tree, cow_mamba=True)
                self.assertEqual(
                    [
                        (c.args[0].cow_mamba, c.args[0].key.limit)
                        for c in spy.call_args_list
                    ],
                    [(False, 12), (True, limit)],
                )
                self.assertEqual(req.prefix_len, prefix_len)

    def test_forced_kv_stays_out_of_the_tree_until_collected(self):
        """The checkpoint after a forced extend keeps the request's own KV, which
        an insert would swap for the cached KV with other routes; the request
        rejoins the tree once its routed experts are collected."""
        for tree_factory in (_radix_tree, _unified_tree):
            with self.subTest(tree_factory.__name__):
                tree = tree_factory(page_size=2)
                engine = self._engine(tree)
                req, _ = _radix_with_held_request(engine, differ_at=5, tree=tree)
                state = req.routed_experts_replay_state
                held = np.empty((11, 2, 2), dtype=np.uint8)
                state.held.copy_rows(0, 11, out=held)
                req.init_next_round_input(tree)
                self.assertEqual(req.prefix_len, 4)
                engine.start_on(req, tree.prefix_device_indices(req))

                engine.version = 1
                engine.extend(req, 4, 12)
                self.assertTrue(req.skip_radix_cache_insert)
                own = engine.req_to_token[req.kv.req_pool_idx, 4:12].clone()
                checkpoint_kv_cache(req, tree)
                self.assertTrue(
                    torch.equal(engine.req_to_token[req.kv.req_pool_idx, 4:12], own)
                )
                self.assertTrue(_same_sets(engine.rows(req, 0, 11), held))

                _collect(req)
                self.assertFalse(req.skip_radix_cache_insert)


def _run_tp_rank(rank: int, init_file: str) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        with published_topology():
            engine = _Engine(tp_cpu_group=torch.distributed.group.WORLD)
            set_global_routed_experts_replay(engine.replay)
            # Identical valid rows are held on both ranks. A corrupt or unreadable
            # row on one rank drops the snapshot on both ranks, and valid rows
            # that differ across ranks are dropped too.
            for spoil in (None, "corrupt", "different", "out_of_range"):
                req = engine.request(list(range(6)))
                engine.extend(req, 0, 6)
                engine.decode(req, 2)
                if rank == 1 and spoil is not None:
                    row = engine.host[engine.req_to_token[req.kv.req_pool_idx, 2], 2]
                    unused = [e for e in range(8) if e not in row.tolist()]
                    row[0] = {
                        "corrupt": row[1],
                        "different": unused[0],
                        "out_of_range": 200,
                    }[spoil]
                engine.retract(req)
                state = req.routed_experts_replay_state
                expected = (8, 0) if spoil is None else (0, 1)
                got = (state.held.end, state.snapshot_dropped)
                assert got == expected, (rank, spoil, got)
                if spoil is None:
                    assert int(engine.extend(req, 0, 9).mask.sum()) == 8, rank
            # Both ranks cut a radix match at the first row that differs on any rank.
            req, tree = _radix_with_held_request(engine, differ_at=7 if rank else 3)
            req.init_next_round_input(tree)
            assert req.prefix_len == 2, (rank, req.prefix_len)
    finally:
        torch.distributed.destroy_process_group()


class TestTpAgreement(CustomTestCase):
    def test_ranks_agree_on_every_decision(self):
        with TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _run_tp_rank,
                args=(str(Path(directory) / "gloo-init"),),
                nprocs=2,
                join=True,
            )


class TestCreate(CustomTestCase):
    def test_refuses_gates_it_cannot_force(self):
        """Replay starts only for gates whose weights it reproduces, and its layout
        keeps the captured TopK layers of the loaded model."""
        capturer = SimpleNamespace(
            topk_size=2, host_cache=SimpleNamespace(buffer=torch.zeros((8, 4, 2)))
        )
        expert_location = SimpleNamespace(num_logical_experts=16)

        def create(*gate_kwargs):
            model = torch.nn.ModuleList(
                TopK(top_k=2, layer_id=layer_id, **kwargs)
                for layer_id, kwargs in zip((1, 3), gate_kwargs)
            )
            return RoutedExpertsReplay.create(
                model=model,
                capturer=capturer,
                expert_location=expert_location,
                req_to_token=torch.zeros((2, 8), dtype=torch.int32),
            )

        with published_topology(routed_experts_replay_max_tokens=64):
            replay = create({}, {})
            self.assertEqual(
                replay.layout,
                RouteLayout(moe_layer_ids=(1, 3), top_k=2, num_logical_experts=16),
            )
            self.assertEqual(replay.max_tokens, 64)
            bias = {"correction_bias": torch.zeros(16)}
            with self.assertRaisesRegex(ValueError, "softmax gates with a correction"):
                create({}, bias)


class TestForcedRouting(CustomTestCase):
    def test_forced_weights_are_the_gates_own_for_its_choice(self):
        """For each gate config that gate_mirror_problem accepts, the forced
        weights equal the gate's own weights for the same expert set."""
        torch.manual_seed(0)
        logits = torch.randn(16, 64)
        hidden = torch.empty(16, 1)
        bias = torch.randn(64)
        gates = [
            # (scoring_func, renormalize, routed scaling on output, gate output)
            (
                "softmax",
                True,
                None,
                fused_topk_torch_native(
                    hidden_states=hidden, gating_output=logits, topk=8, renormalize=True
                ),
            ),
            # flashinfer_trtllm_routed's softmax over the chosen raw logits.
            (
                "softmax",
                True,
                None,
                fused_topk_softmax_torch_raw_logits(
                    hidden_states=hidden, gating_output=logits, topk=8, renormalize=True
                ),
            ),
            (
                "sigmoid",
                False,
                None,
                fused_topk_torch_native(
                    hidden_states=hidden,
                    gating_output=logits,
                    topk=8,
                    renormalize=False,
                    scoring_func="sigmoid",
                ),
            ),
            # A fused shared expert in the last column, which forcing leaves alone.
            (
                "sigmoid",
                True,
                2.5,
                _biased_topk(
                    hidden_states=hidden,
                    gating_output=logits,
                    correction_bias=bias,
                    topk=9,
                    renormalize=True,
                    num_fused_shared_experts=1,
                    routed_scaling_factor=2.5,
                    apply_routed_scaling_factor_on_output=True,
                ),
            ),
            *(
                (
                    "sqrtsoftplus",
                    renormalize,
                    None,
                    _biased_topk(
                        hidden_states=hidden,
                        gating_output=logits,
                        correction_bias=bias,
                        topk=8,
                        renormalize=renormalize,
                        scoring_func="sqrtsoftplus",
                    ),
                )
                for renormalize in (True, False)
            ),
        ]
        for scoring_func, renormalize, scaling, (weights, ids) in gates:
            with self.subTest(scoring_func=scoring_func, renormalize=renormalize):
                got = _forced_routing_weights(
                    logits,
                    ids[:, :8],
                    scoring_func=scoring_func,
                    renormalize=renormalize,
                    routed_scaling_factor=scaling,
                    apply_routed_scaling_factor_on_output=scaling is not None,
                )
                torch.testing.assert_close(got, weights[:, :8])

    def test_apply_touches_only_masked_rows_whose_expert_set_changed(self):
        torch.manual_seed(1)
        logits = torch.randn(6, 32)
        weights, ids = fused_topk_torch_native(
            hidden_states=logits, gating_output=logits, topk=4, renormalize=True
        )
        ids = ids.int()
        # A fused shared-expert column after the routed ones.
        topk_ids = torch.cat([ids, torch.full((6, 1), 32, dtype=torch.int32)], dim=1)
        topk_weights = torch.cat([weights, torch.full((6, 1), 0.4)], dim=1)

        forced = ids.clone()
        forced[1] = torch.tensor([0, 1, 2, 3])  # changed and masked
        forced[2] = forced[2].flip(0)  # same set, other order: untouched
        forced[4] = torch.tensor([5, 6, 7, 8])  # changed but not masked
        mask = torch.tensor([True, True, True, False, False, True])
        new_ids, new_weights = apply_forced_routing(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            router_logits=logits,
            forced_ids=forced,
            mask=mask,
            scoring_func="softmax",
            renormalize=True,
            routed_scaling_factor=None,
            apply_routed_scaling_factor_on_output=False,
        )
        for row in (0, 2, 3, 4, 5):
            self.assertTrue(torch.equal(new_ids[row], topk_ids[row]))
            self.assertTrue(torch.equal(new_weights[row], topk_weights[row]))
        self.assertEqual(new_ids[1, :4].tolist(), [0, 1, 2, 3])
        probs = logits.softmax(-1)[1, :4]
        torch.testing.assert_close(new_weights[1, :4], probs / probs.sum())
        self.assertTrue(torch.equal(new_ids[:, 4], topk_ids[:, 4]))
        self.assertTrue(torch.equal(new_weights[:, 4], topk_weights[:, 4]))
        self.assertEqual(topk_ids[1, :4].tolist(), ids[1].tolist(), "inputs unchanged")

        with self.assertRaisesRegex(ValueError, "forcing rows"):
            apply_forced_routing(
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                router_logits=logits,
                forced_ids=forced,
                mask=mask[:5],
                scoring_func="softmax",
                renormalize=True,
                routed_scaling_factor=None,
                apply_routed_scaling_factor_on_output=False,
            )

    def test_gate_mirror_problem_refuses_unmirrored_weights(self):
        def problem(**overrides):
            gate = dict(
                scoring_func="sigmoid",
                has_correction_bias=True,
                has_custom_routing=False,
                renormalize=True,
                apply_routed_scaling_factor_on_output=True,
                num_fused_shared_experts=1,
            )
            return gate_mirror_problem(**{**gate, **overrides})

        self.assertIsNone(problem())
        self.assertIsNone(
            problem(
                scoring_func="softmax",
                has_correction_bias=False,
                apply_routed_scaling_factor_on_output=False,
            )
        )
        self.assertIsNone(
            problem(
                renormalize=False,
                apply_routed_scaling_factor_on_output=False,
                num_fused_shared_experts=0,
            )
        )
        refused = (
            dict(has_custom_routing=True),
            dict(scoring_func="tanh"),
            dict(scoring_func="softmax", apply_routed_scaling_factor_on_output=False),
            dict(scoring_func="softmax", has_correction_bias=False),
            dict(
                scoring_func="softmax",
                has_correction_bias=False,
                apply_routed_scaling_factor_on_output=False,
                renormalize=False,
            ),
            dict(renormalize=False, num_fused_shared_experts=0),
            dict(renormalize=False, apply_routed_scaling_factor_on_output=False),
        )
        for overrides in refused:
            with self.subTest(**overrides):
                self.assertIsNotNone(problem(**overrides))


if __name__ == "__main__":
    unittest.main()
