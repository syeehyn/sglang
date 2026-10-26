import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from sglang.srt.state_capturer.routed_experts import (
    extract_routed_experts_from_meta_info,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=480, stage="extra-a", runner_config="2-gpu-large")

MODEL = DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST
_QWEN3_30B_A3B_NUM_LAYERS = 48
_QWEN3_30B_A3B_TOPK = 8
PROMPT = (
    "Write a short, careful explanation of why the sum of the first n odd numbers "
    "equals n squared, and then check the claim for n = 1, 2, 3 and 4."
)
# Output tokens of the earlier request whose KV the turn starts on.
HISTORY_TOKENS = 64
# The turn is retracted once this many of its tokens have streamed out.
RETRACT_AFTER_TOKENS = 128
# Below this share of differing rows, an unforced recompute under the perturbed
# routers cannot tell replay from a no-op.
MIN_CONTROL_FLIP_FRAC = 0.01
ZERO_COUNTERS = {
    "forced_tokens": 0,
    "retract_snapshots": 0,
    "snapshot_dropped": 0,
    "prefix_mismatch_tokens": 0,
}


def _perturbed_routers(out_dir: Path, *, seed: int, scale: float = 0.2) -> Path:
    """A checkpoint holding only the MoE routers, each scaled by 1 + scale * N(0, 1);
    /update_weights_from_disk loads the tensors it finds and keeps the rest."""
    model_dir = Path(snapshot_download(MODEL, allow_patterns=["*.safetensors"]))
    generator = torch.Generator().manual_seed(seed)
    routers = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                if name.endswith(".mlp.gate.weight"):
                    weight = f.get_tensor(name)
                    noise = torch.randn(weight.shape, generator=generator)
                    routers[name] = (weight.float() * (1 + scale * noise)).to(
                        weight.dtype
                    )
    assert len(routers) == _QWEN3_30B_A3B_NUM_LAYERS, len(routers)
    out_dir.mkdir(parents=True)
    save_file(routers, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    return out_dir


def _sorted_rows(routes: np.ndarray) -> np.ndarray:
    return np.sort(routes, axis=-1)


def _differing_row_frac(a: np.ndarray, b: np.ndarray) -> float:
    return float((_sorted_rows(a) != _sorted_rows(b)).any(axis=-1).mean())


class TestRoutedExpertsReplayRetract(CustomTestCase):
    """A turn retracted mid-decode re-prefills with the routed experts the engine
    first computed for its tokens, under new router weights or on a radix hit.

    Each turn starts on a radix hit of an earlier request, whose returned rows are
    the reference for those positions: rows are compared with rows the engine
    returned, never with a fresh prefill, since SGLang prefill is not run-to-run
    exact. Rows are compared as sets, since a forced position whose experts the
    gate picks again keeps the gate's order. The triton attention backend with
    page size 1 runs on Hopper and Blackwell alike, and lets the radix hit cover
    every token of the earlier request. One request runs at a time, so a request
    sent while a turn runs is scheduled before the retracted turn resumes.
    """

    @classmethod
    def setUpClass(cls):
        workdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(workdir.cleanup)
        cls.perturbed = _perturbed_routers(Path(workdir.name) / "routers", seed=1)
        cls.perturbed_again = _perturbed_routers(
            Path(workdir.name) / "routers-again", seed=2
        )
        process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                2,
                "--enable-return-routed-experts",
                "--enable-routed-experts-replay",
                "--moe-runner-backend",
                "triton",
                "--attention-backend",
                "triton",
                "--page-size",
                1,
                "--max-running-requests",
                1,
            ],
        )
        cls.addClassCleanup(terminate_and_kill_process_tree, process)

    def test_retracted_turn_reuses_its_first_routes(self):
        info = requests.get(f"{DEFAULT_URL_FOR_TEST}/server_info", timeout=60).json()
        self.assertTrue(info["enable_routed_experts_replay"])

        history = self._generate(self._tokenize(PROMPT), HISTORY_TOKENS)
        self.assertEqual(history["meta_info"]["routed_experts_replay"], ZERO_COUNTERS)
        history_ids = self._tokenize(PROMPT) + history["output_ids"]
        history_routes = self._routes(history)
        self.assertEqual(history_routes.shape[0], len(history_ids) - 1)

        # The router update flushes the radix cache, as an RL weight sync does, so
        # the re-prefill recomputes every kept token.
        turn, retracted_at, _ = self._generate_retracted(
            history_ids, 2048, model_path=self.perturbed
        )
        self.assertIsNotNone(
            retracted_at,
            f"the turn finished before {RETRACT_AFTER_TOKENS} tokens streamed out",
        )
        stats = turn["meta_info"]["routed_experts_replay"]
        self.assertEqual(
            {k: stats[k] for k in ("retract_snapshots", "snapshot_dropped")},
            {"retract_snapshots": 1, "snapshot_dropped": 0},
            stats,
        )
        self.assertEqual(stats["prefix_mismatch_tokens"], 0, stats)
        forced = stats["forced_tokens"]
        self.assertGreaterEqual(forced, len(history_ids) - 1 + retracted_at, stats)
        turn_ids = history_ids + turn["output_ids"]
        turn_routes = self._routes(turn)
        self.assertEqual(turn_routes.shape[0], len(turn_ids) - 1)

        # Rows the turn first read through the radix hit keep the earlier
        # request's experts after the recompute under the new routers.
        np.testing.assert_array_equal(
            _sorted_rows(turn_routes[: len(history_ids) - 1]),
            _sorted_rows(history_routes),
        )

        # An unforced recompute of the same tokens under the new routers flips
        # rows; its own radix namespace keeps it off the turn's cached KV.
        control = self._generate(turn_ids, 1, extra_key="replay-control")
        control = self._routes(control)[:forced]
        flips = _differing_row_frac(control, turn_routes[:forced])
        self.assertGreater(flips, MIN_CONTROL_FLIP_FRAC, "perturbation too small")

        # Retracted with no flush, the next turn keeps its radix hit over the
        # finished turn (the cached rows are its held rows) and forces only the
        # rows it decoded.
        turn2, retracted_at2, _ = self._generate_retracted(turn_ids, 512)
        self.assertIsNotNone(retracted_at2)
        stats2 = turn2["meta_info"]["routed_experts_replay"]
        self.assertEqual(
            {**stats2, "forced_tokens": 0},
            {**ZERO_COUNTERS, "retract_snapshots": 1},
            stats2,
        )
        # The tree holds the prompt, so only decoded tokens' rows are forced: all
        # but the last kept token's.
        self.assertGreaterEqual(stats2["forced_tokens"], retracted_at2 - 1, stats2)
        self.assertLess(stats2["forced_tokens"], len(turn_ids) - 1, stats2)
        np.testing.assert_array_equal(
            _sorted_rows(self._routes(turn2)[: len(turn_ids) - 1]),
            _sorted_rows(turn_routes),
        )

    def test_retracted_turn_cuts_a_prefix_recomputed_by_another_request(self):
        """A turn retracted across a router update, whose prompt another request
        recomputes under the new routers before the turn resumes, cuts that radix
        hit and recomputes its prompt with its own first routes."""
        namespace = "replay-sibling"
        reference = self._generate(
            self._tokenize(PROMPT), HISTORY_TOKENS, extra_key=namespace
        )
        reference_ids = self._tokenize(PROMPT) + reference["output_ids"]
        reference_routes = self._routes(reference)

        turn, retracted_at, sibling = self._generate_retracted(
            reference_ids,
            1024,
            model_path=self.perturbed_again,
            extra_key=namespace,
            sibling=True,
        )
        self.assertIsNotNone(retracted_at)
        sibling_routes = self._routes(sibling)[: len(reference_ids) - 1]
        self.assertGreater(_differing_row_frac(sibling_routes, reference_routes), 0)

        stats = turn["meta_info"]["routed_experts_replay"]
        self.assertEqual(
            {k: stats[k] for k in ("retract_snapshots", "snapshot_dropped")},
            {"retract_snapshots": 1, "snapshot_dropped": 0},
            stats,
        )
        self.assertGreater(stats["prefix_mismatch_tokens"], 0, stats)
        self.assertGreaterEqual(stats["forced_tokens"], retracted_at, stats)
        np.testing.assert_array_equal(
            _sorted_rows(self._routes(turn)[: len(reference_ids) - 1]),
            _sorted_rows(reference_routes),
        )

    def _generate_retracted(
        self,
        input_ids,
        max_new_tokens,
        *,
        model_path=None,
        extra_key=None,
        sibling=False,
    ):
        """Stream a turn and retract it once RETRACT_AFTER_TOKENS tokens are out.

        With ``model_path``, the routers change (which flushes the radix cache)
        before it resumes. With ``sibling``, a one-token request for the same
        input is queued first, so it runs before the turn resumes. Returns the
        turn's final output, where it was retracted (None if it finished first)
        and the sibling's output.
        """
        payload = {
            **self._payload(input_ids, max_new_tokens, extra_key),
            "stream": True,
        }
        retracted_at = sibling_out = None
        with (
            ThreadPoolExecutor(max_workers=1) as pool,
            requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json=payload,
                stream=True,
                timeout=600,
            ) as response,
        ):
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                out = json.loads(line[len(b"data: ") :])
                meta = out["meta_info"]
                if (
                    retracted_at is None
                    and meta["finish_reason"] is None
                    and meta["completion_tokens"] >= RETRACT_AFTER_TOKENS
                ):
                    retracted_at = meta["completion_tokens"]
                    if sibling:
                        sibling_out = pool.submit(
                            self._generate, input_ids, 1, extra_key
                        )
                        self._wait_until_queued()
                    self._post("/pause_generation", {"mode": "retract"})
                    if model_path is not None:
                        self._post(
                            "/update_weights_from_disk", {"model_path": str(model_path)}
                        )
                    self._post("/continue_generation", {})
        if sibling_out is not None:
            sibling_out = sibling_out.result()
        return out, retracted_at, sibling_out

    def _wait_until_queued(self, timeout: float = 60):
        """Wait until a request waits behind the running turn."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            loads = requests.get(
                f"{DEFAULT_URL_FOR_TEST}/v1/loads",
                params={"include": "core"},
                timeout=60,
            ).json()["loads"]
            if sum(load["num_waiting_reqs"] for load in loads) > 0:
                return
            time.sleep(0.05)
        self.fail("the sibling request never reached the waiting queue")

    def _tokenize(self, text: str) -> list:
        return self._post("/tokenize", {"prompt": text})["tokens"]

    def _payload(self, input_ids, max_new_tokens, extra_key=None):
        return {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_routed_experts": True,
            "extra_key": extra_key,
        }

    def _generate(self, input_ids, max_new_tokens, extra_key=None):
        return self._post(
            "/generate", self._payload(input_ids, max_new_tokens, extra_key)
        )

    def _post(self, path: str, body: dict) -> dict:
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}{path}", json=body, timeout=600
        )
        response.raise_for_status()
        return response.json()

    def _routes(self, output: dict) -> np.ndarray:
        return extract_routed_experts_from_meta_info(output).reshape(
            -1, _QWEN3_30B_A3B_NUM_LAYERS, _QWEN3_30B_A3B_TOPK
        )


if __name__ == "__main__":
    unittest.main()
