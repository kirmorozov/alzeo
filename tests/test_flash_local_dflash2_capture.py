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
    """The capture arithmetic the wrapped layers do, without CUDA: the
    deferred hyper-connection combine materializes into a [tokens, hc*hidden]
    multi-stream state, the capture averages the streams, and copies the
    current tokens' rows into one fixed buffer slot, the way the model's own
    MTP buffer is filled."""

    def test_stream_mean_into_a_fixed_buffer(self):
        t = TORCH
        hc, hidden, tokens, max_tokens = 4, 8, 3, 16
        combined = t.arange(
            tokens * hc * hidden, dtype=t.float32
        ).view(tokens, hc * hidden)
        aux = combined.unflatten(-1, (hc, hidden)).mean(dim=-2)
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
        buffer = t.full((2, max_tokens, hidden), float("nan"))
        slot = 1
        view = buffer[slot, :tokens]
        view.copy_(aux)
        self.assertTrue(t.equal(buffer[slot, :tokens], aux))
        self.assertTrue(t.isnan(buffer[slot, tokens:]).all())
        self.assertTrue(t.isnan(buffer[0]).all())
        self.assertEqual(view.data_ptr(), buffer[slot, :tokens].data_ptr())


if __name__ == "__main__":
    unittest.main()
