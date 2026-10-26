"""Held rows of routed-experts replay and the rows an extend forward forces."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.state_capturer import routed_experts_rows
from sglang.srt.state_capturer.routed_experts_rows import (
    HeldRoutes,
    RouteLayout,
    RouteSegment,
    _route_dtype,
    build_forcing_batch,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

LAYOUT = RouteLayout(moe_layer_ids=(1, 2), top_k=3, num_logical_experts=200)


def _segment(start, n):
    ids = np.arange(start * 6, (start + n) * 6) % 200
    return RouteSegment(start=start, routes=ids.reshape(n, 2, 3).astype(np.uint8))


def _held(*lengths):
    held, start = HeldRoutes(), 0
    for n in lengths:
        held = held.extended(_segment(start, n))
        start += n
    return held


def _all_routes(held):
    return np.concatenate([s.routes for s in held.segments])


def _table(held, num_slots=32):
    """A capture table holding ``held``'s rows at slots ``[10, 10 + n)``."""
    table = torch.zeros((num_slots, 4, 3), dtype=torch.int32)
    table[10 : 10 + held.end, 1:3] = torch.from_numpy(_all_routes(held)).int()
    return table, torch.arange(10, 10 + held.end)


class TestRouteLayout(CustomTestCase):
    def test_dtype_is_the_narrowest_that_holds_every_id(self):
        self.assertEqual(_route_dtype(8), np.uint8)
        self.assertEqual(_route_dtype(256), np.uint8)
        self.assertEqual(_route_dtype(257), np.int16)
        self.assertEqual(_route_dtype(32768), np.int16)
        for bad in (0, 32769):
            with self.assertRaises(ValueError):
                _route_dtype(bad)

    def test_gather_keeps_moe_columns_in_chunks(self):
        table = torch.zeros((16, 4, 3), dtype=torch.int32)  # layers 0 and 3 are dense
        table[:, 1:3] = torch.arange(16 * 6).reshape(16, 2, 3) % 6 + 250
        slots = torch.tensor([9, 2, 2, 15, 0, 7, 4])
        for experts, dtype in ((256, np.uint8), (257, np.int16)):
            layout = RouteLayout(
                moe_layer_ids=(1, 2), top_k=3, num_logical_experts=experts
            )
            with patch.object(routed_experts_rows, "_GATHER_CHUNK_TOKENS", 3):
                rows = layout.gather(table, slots)
            self.assertEqual(rows.dtype, dtype)
            np.testing.assert_array_equal(rows, table[slots, 1:3].numpy())
        every_layer = RouteLayout(
            moe_layer_ids=(0, 1, 2, 3), top_k=3, num_logical_experts=256
        )
        np.testing.assert_array_equal(
            every_layer.gather(table, slots), table[slots].numpy()
        )
        self.assertEqual(LAYOUT.gather(table, slots[:0]).shape, (0, 2, 3))

    def test_gather_refuses_ids_it_cannot_hold(self):
        layout = RouteLayout(moe_layer_ids=(0,), top_k=1, num_logical_experts=255)
        for bad in (255, -1):
            with self.assertRaisesRegex(ValueError, "outside"):
                layout.gather(torch.full((2, 1, 1), bad), torch.arange(2))
        with self.assertRaisesRegex(ValueError, "top-1"):
            layout.gather(torch.zeros((2, 1, 2), dtype=torch.int32), torch.arange(2))

    def test_first_differing_row_compares_expert_sets(self):
        held = _held(4, 5)
        table, slots = _table(held)
        self.assertEqual(LAYOUT.first_differing_row(table, slots, held), 9)
        # The same experts in another order are the same routing.
        table[12, 1] = table[12, 1].flip(0)
        self.assertEqual(LAYOUT.first_differing_row(table, slots, held), 9)
        with patch.object(routed_experts_rows, "_GATHER_CHUNK_TOKENS", 4):
            table[16, 2, 0] = 199
            self.assertEqual(LAYOUT.first_differing_row(table, slots, held), 6)
            # Ids no gate can produce make their whole chunk differ.
            table[17, 1, 0] = 200
            self.assertEqual(LAYOUT.first_differing_row(table, slots, held), 4)
            table[13, 1, 1] = 199
            self.assertEqual(LAYOUT.first_differing_row(table, slots, held), 3)


class TestHeldRoutes(CustomTestCase):
    def test_chain_and_copy_across_segments(self):
        held = _held(5, 3, 4)
        self.assertEqual(held.end, 12)
        with self.assertRaisesRegex(ValueError, "not 5"):
            HeldRoutes(segments=(_segment(0, 5), _segment(6, 2)))
        self.assertIs(held.extended(_segment(12, 0)), held)

        out = np.full((6, 2, 3), 99, dtype=np.uint8)
        held.copy_rows(3, 9, out=out)
        np.testing.assert_array_equal(out, _all_routes(held)[3:9])
        with self.assertRaisesRegex(ValueError, "outside"):
            held.copy_rows(10, 13, out=out)

    def test_segments_are_read_only(self):
        segment = _segment(0, 2)
        with self.assertRaises(ValueError):
            segment.routes[0, 0, 0] = 1


class TestForcingBatch(CustomTestCase):
    def test_build_places_rows_at_their_forward_rows(self):
        held = [_held(10), _held(6), HeldRoutes()]
        batch = build_forcing_batch(
            held=held, prefix_lens=[8, 0, 0], extend_lens=[5, 4, 3], layout=LAYOUT
        )
        self.assertEqual((int(batch.mask.sum()), len(batch.mask)), (6, 12))
        self.assertEqual(
            batch.mask.tolist(), [True, True] + [False] * 3 + [True] * 4 + [False] * 3
        )
        self.assertEqual(batch.routes.dtype, torch.uint8)
        np.testing.assert_array_equal(
            batch.routes[:2].numpy(), _all_routes(held[0])[8:10]
        )
        np.testing.assert_array_equal(
            batch.routes[5:9].numpy(), _all_routes(held[1])[:4]
        )
        self.assertFalse(batch.routes[2:5].any() or batch.routes[9:].any())

        device = batch.padded_to(15).to_device("cpu")
        self.assertEqual(len(device.mask), 15)
        self.assertFalse(device.mask[12:].any())
        self.assertEqual(device.routes.dtype, torch.int32)
        self.assertTrue(torch.equal(device.routes[:12], batch.routes.int()))

        self.assertIsNone(
            build_forcing_batch(
                held=held, prefix_lens=[10, 6, 0], extend_lens=[2, 2, 2], layout=LAYOUT
            )
        )

    def test_wide_ids_survive_the_device_copy(self):
        layout = RouteLayout(moe_layer_ids=(0,), top_k=2, num_logical_experts=512)
        routes = layout.gather(
            torch.tensor([[[511, 300]], [[0, 256]]], dtype=torch.int32),
            torch.arange(2),
        )
        held = HeldRoutes(segments=(RouteSegment(start=0, routes=routes),))
        batch = build_forcing_batch(
            held=[held], prefix_lens=[0], extend_lens=[2], layout=layout
        ).to_device("cpu")
        self.assertEqual(batch.routes[:, 0].tolist(), [[511, 300], [0, 256]])


if __name__ == "__main__":
    unittest.main()
