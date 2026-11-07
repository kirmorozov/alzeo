#!/usr/bin/env python3
"""flash-local-dflash2 is the DFlash2 capture plugin, on PYTHONPATH with no pip install.

The lane mounts it read-only at /opt/df2-plugin, and vLLM's general-plugin
loader must find it through importlib.metadata: a plain package directory
plus a .dist-info holding METADATA and an entry_points.txt with a
vllm.general_plugins group. Without a GPU only what a GPU does not settle
is checked here: the plugin's files compile, the entry point file parses and
resolves to register() from a temporary sys.path, and loading it imports no
vllm. The capture arithmetic (the hyper-connection stream mean into a fixed
buffer, the SGLang reference patch's semantics) runs on a tiny tensor when
torch is there.
"""
import configparser
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
PLUGIN = REPO / "flash-local-dflash2"
DIST_INFO = PLUGIN / "flash_local_dflash2-0.0.0.dist-info"
ENTRY_POINT_NAME = "flash_local_dflash2"
ENTRY_POINT_VALUE = "flash_local_dflash2.plugin:register"


def torch_or_none():
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


TORCH = torch_or_none()


class PluginTree(unittest.TestCase):
    def setUp(self):
        self.t = pathlib.Path(tempfile.mkdtemp(prefix="df2-capture-"))

    def tearDown(self):
        shutil.rmtree(self.t, ignore_errors=True)

    def test_plugin_files_compile(self):
        # The lane copies this tree as is, so every file must parse, checked
        # without writing bytecode into the repo.
        files = sorted(PLUGIN.rglob("*.py"))
        self.assertTrue(files)
        for f in files:
            compile(f.read_text(), str(f), "exec")

    def test_entry_points_file_declares_the_vllm_group(self):
        # Parsed as the importlib.metadata writer writes it: an INI file with
        # one section per group, one option per plugin.
        parsed = configparser.ConfigParser()
        parsed.read_string((DIST_INFO / "entry_points.txt").read_text())
        self.assertIn("vllm.general_plugins", parsed.sections())
        self.assertEqual(
            parsed["vllm.general_plugins"][ENTRY_POINT_NAME], ENTRY_POINT_VALUE
        )
        metadata = (DIST_INFO / "METADATA").read_text().splitlines()
        self.assertIn("Name: flash-local-dflash2", metadata)
        self.assertIn("Version: 0.0.0", metadata)
        # the dist-info directory is named after the normalized metadata
        # name: importlib.metadata matches it that way on sys.path
        self.assertEqual(DIST_INFO.name, "flash_local_dflash2-0.0.0.dist-info")

    def test_entry_point_resolves_from_a_temporary_sys_path(self):
        # What the lane's mount gives PYTHONPATH: the package plus the
        # dist-info, nothing else, in a fresh interpreter that must find the
        # plugin by name and load it without importing vllm.
        shutil.copytree(PLUGIN, self.t, dirs_exist_ok=True)
        # The braces of the subprocess script are doubled: only the two
        # constants are filled in here.
        script = f"""
import importlib.metadata, sys
eps = importlib.metadata.entry_points(group="vllm.general_plugins")
mine = [e for e in eps if e.name == {ENTRY_POINT_NAME!r}]
assert len(mine) == 1, f"expected one plugin, found {{[e.value for e in mine]}}"
ep = mine[0]
assert ep.value == {ENTRY_POINT_VALUE!r}, ep.value
fn = ep.load()
assert callable(fn) and fn.__name__ == "register", fn
assert fn.__module__ == "flash_local_dflash2.plugin"
assert "vllm" not in sys.modules, "loading the entry point must not import vllm"
assert importlib.metadata.distribution("flash-local-dflash2").version == "0.0.0"
print("ok")
"""
        env = dict(os.environ, PYTHONPATH=str(self.t))
        r = subprocess.run(
            [sys.executable, "-P", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(r.returncode, 0, f"{r.stdout}\n{r.stderr}")
        self.assertEqual(r.stdout.strip(), "ok")


@unittest.skipUnless(
    TORCH is not None, "needs the reference class of machine (its torch)"
)
class HcMeanCapture(unittest.TestCase):
    """The capture the wrapped layers do, without CUDA and without vLLM
    (flash_local_dflash2.capture needs torch alone): the deferred
    hyper-connection combine materializes into a [tokens, hc*hidden]
    multi-stream state, the capture averages the streams, and copies the
    current tokens' rows into one fixed buffer slot, the way the model's own
    MTP buffer is filled. The wrapper hands the triple on unchanged and
    answers for the layer it wraps."""

    @classmethod
    def setUpClass(cls):
        # imported from the repo's folder, which is what the lane copies: no bytecode written
        # into it by the test
        cls.bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(PLUGIN))
        import flash_local_dflash2.capture as capture
        cls.capture = capture

    @classmethod
    def tearDownClass(cls):
        sys.dont_write_bytecode = cls.bytecode
        sys.path.remove(str(PLUGIN))
        sys.modules.pop("flash_local_dflash2.capture", None)
        sys.modules.pop("flash_local_dflash2", None)

    def test_stream_mean_into_a_fixed_buffer(self):
        t = TORCH
        hc, hidden, tokens, max_tokens = 4, 8, 3, 16
        combined = t.arange(
            tokens * hc * hidden, dtype=t.float32
        ).view(tokens, hc * hidden)
        buffer = t.full((2, max_tokens, hidden), float("nan"))
        slot = 1
        view_before = buffer[slot, :tokens].data_ptr()
        self.capture.capture_hc_mean(combined, buffer, slot, hc, hidden)
        aux = buffer[slot, :tokens]
        self.assertEqual(tuple(aux.shape), (tokens, hidden))
        # hand check: feature j of token i averages stream s at s*hidden + j
        for i in range(tokens):
            for j in range(hidden):
                expect = sum(
                    float(combined[i, s * hidden + j]) for s in range(hc)
                ) / hc
                self.assertAlmostEqual(aux[i, j].item(), expect)
        # the fixed-buffer discipline: only the current tokens' rows are
        # written, into storage that keeps its address, so a CUDA graph
        # replayed at the same shape overwrites the same bytes
        self.assertTrue(t.isnan(buffer[slot, tokens:]).all())
        self.assertTrue(t.isnan(buffer[0]).all())
        self.assertEqual(buffer[slot, :tokens].data_ptr(), view_before)
        # a width that is not hc streams of hidden, or more rows than the
        # buffer holds, is a wrong capture, refused rather than reshaped
        with self.assertRaises(AssertionError):
            self.capture.capture_hc_mean(combined[:, :-1], buffer, slot, hc, hidden)
        with self.assertRaises(AssertionError):
            self.capture.capture_hc_mean(
                combined.repeat(max_tokens, 1), buffer, slot, hc, hidden
            )

    def test_the_wrapper_hands_the_triple_on_and_answers_for_its_layer(self):
        t = TORCH
        hc, hidden, tokens, max_tokens = 4, 8, 3, 16
        calls = []

        class Combine:
            # the image's GatedResidual.combine: a pure function of the triple
            def combine(self, hidden_states, block_output, injection):
                calls.append((hidden_states, block_output, injection))
                return hidden_states + block_output.repeat(1, hc) * injection[:, :1]

        class Layer(t.nn.Module):
            layer_type = "linear_attention"
            layer_idx = 5

            def __init__(self):
                super().__init__()
                self.mlp_hyper_connection = Combine()
                self.linear_attn = t.nn.Identity()

            def forward(self, hidden_states, prev_block_output, prev_injection,
                        positions, *, input_ids, query_start_loc, ngram_context,
                        output_indices=None):
                self.seen = dict(prev_block_output=prev_block_output,
                                 prev_injection=prev_injection, positions=positions,
                                 input_ids=input_ids, query_start_loc=query_start_loc,
                                 ngram_context=ngram_context, output_indices=output_indices)
                return hidden_states * 2, t.ones(tokens, hidden), t.full((tokens, hc), 0.5)

        layer = Layer()
        buffer = t.full((5, max_tokens, hidden), float("nan"))
        wrapped = self.capture.DFlashCaptureLayer(layer, buffer, 2, hc, hidden)
        # the decoder layer's own attributes, read through the wrapper
        self.assertEqual(wrapped.layer_type, "linear_attention")
        self.assertEqual(wrapped.layer_idx, 5)
        self.assertIs(wrapped.linear_attn, layer.linear_attn)
        self.assertIs(wrapped.mlp_hyper_connection, layer.mlp_hyper_connection)
        self.assertIs(wrapped.layer, layer)
        with self.assertRaises(AttributeError):
            wrapped.no_such_attribute
        # the wrapped layer stays in the module walk, under the wrapper
        self.assertIn(layer, list(wrapped.modules()))
        x = t.arange(tokens * hc * hidden, dtype=t.float32).view(tokens, hc * hidden)
        positions = t.arange(tokens)
        out = wrapped(x, None, None, positions, input_ids=positions,
                      query_start_loc=None, ngram_context=None)
        self.assertEqual(layer.seen["positions"].data_ptr(), positions.data_ptr())
        self.assertIsNone(layer.seen["prev_block_output"])
        # the triple goes on as the layer returned it: the next layer does its
        # own combine, the capture's is an extra, pure one
        self.assertEqual(len(out), 3)
        self.assertTrue(t.equal(out[0], x * 2))
        self.assertTrue(t.equal(out[1], t.ones(tokens, hidden)))
        self.assertEqual(len(calls), 1)
        self.assertTrue(t.equal(calls[0][0], out[0]))
        self.assertIs(calls[0][1], out[1])
        self.assertIs(calls[0][2], out[2])
        # and its mean of the four streams sits in the wrapper's slot alone
        combined = layer.mlp_hyper_connection.combine(*out)
        want = combined.view(tokens, hc, hidden).mean(dim=1)
        self.assertTrue(t.equal(buffer[2, :tokens], want))
        self.assertTrue(t.isnan(buffer[2, tokens:]).all())
        for slot in (0, 1, 3, 4):
            self.assertTrue(t.isnan(buffer[slot]).all())


if __name__ == "__main__":
    unittest.main()
