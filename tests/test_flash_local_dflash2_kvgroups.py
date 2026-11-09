#!/usr/bin/env python3
"""The plugin's kvgroups hook: one KV cache group per DFlash2 drafter layer.

The dflash2 boot of 2026-11-08 passed every earlier gate and died at vLLM's memory check:
35.77 GiB of KV cache for one request at the 262,144-token window, against 18.75 GiB
available. The image keeps the drafter's five sliding-window layers in one cache group
while each target layer has its own, and sizes every block of the shared pool by the
widest group: five drafter pages of 6,160,384 bytes, 30,801,920 per block, in which each
target group stores 3,272,704 bytes. The hook splits the drafter group one layer per
group, so a block is one drafter page. Checked here on stubs of the vLLM module the hook
wraps, without vLLM: the pure split, the wrapper's gate on the dflash method, the
replacement in every module that holds the function, idempotence, and that the hook
imports nothing of vLLM when it loads.
"""
import importlib
import logging
import pathlib
import sys
import types
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
PLUGIN = REPO / "flash-local-dflash2"

TARGET_LAYERS = 48
TARGET_PAGE = 3_272_704
DRAFT_PAGE = 6_160_384


class Spec:
    def __init__(self, page_size_bytes):
        self.page_size_bytes = page_size_bytes


class Group:
    """The shape of vLLM's KVCacheGroupSpec that the hook touches."""

    def __init__(self, layer_names, kv_cache_spec):
        self.layer_names = list(layer_names)
        self.kv_cache_spec = kv_cache_spec


def layer_index(name):
    return int(name.split(".")[2])


def image_groups():
    """What the image's grouping makes for this lane: 12 attention and 36 GDN layers of the
    target, one group each; the drafter's 5 layers in one group."""
    groups = [Group([f"model.layers.{i}.attn"], Spec(TARGET_PAGE)) for i in range(TARGET_LAYERS)]
    draft = Spec(DRAFT_PAGE)
    groups.append(
        Group([f"model.layers.{TARGET_LAYERS + i}.self_attn.attn" for i in range(5)], draft)
    )
    return groups


def bytes_per_block(groups):
    return max(sum(g.kv_cache_spec.page_size_bytes for _ in g.layer_names) for g in groups)


def partition(vllm_config, kv_cache_spec):
    target, draft = {}, {}
    for name, spec in kv_cache_spec.items():
        (draft if layer_index(name) >= TARGET_LAYERS else target)[name] = spec
    return target, draft


def all_specs(groups):
    return {name: g.kv_cache_spec for g in groups for name in g.layer_names}


class Config:
    def __init__(self, method):
        self.speculative_config = None if method is None else types.SimpleNamespace(method=method)


def stub_kv_cache_utils(groups):
    """A stand-in for vllm.v1.core.kv_cache_utils with the names the hook uses, and a second
    module that imported get_kv_cache_groups the way the CUDA-graph planner does."""
    calls = []

    def get_kv_cache_groups(vllm_config, kv_cache_spec):
        calls.append(sorted(kv_cache_spec))
        return list(groups)

    kcu = types.ModuleType("vllm.v1.core.kv_cache_utils")
    kcu.get_kv_cache_groups = get_kv_cache_groups
    kcu._partition_parallel_draft_specs = partition
    kcu.KVCacheGroupSpec = Group
    kcu._get_per_layer_spec = lambda group, name: group.kv_cache_spec
    kcu._get_kv_cache_bytes_per_block = bytes_per_block
    kcu.calls = calls

    other = types.ModuleType("vllm.v1.worker.gpu.cudagraph_utils")
    other.get_kv_cache_groups = get_kv_cache_groups
    return kcu, other


def load_hook():
    for name in [n for n in sys.modules if n == "flash_local_dflash2" or n.startswith("flash_local_dflash2.")]:
        sys.modules.pop(name)
    if str(PLUGIN) not in sys.path:
        sys.path.insert(0, str(PLUGIN))
    return importlib.import_module("flash_local_dflash2.kvgroups")


class PureSplit(unittest.TestCase):
    def setUp(self):
        self.kvgroups = load_hook()

    def split(self, groups):
        return self.kvgroups.split_draft_groups(
            groups,
            lambda name: layer_index(name) >= TARGET_LAYERS,
            Group,
            lambda group, name: group.kv_cache_spec,
        )

    def test_the_drafter_group_becomes_one_group_per_layer_and_the_target_is_untouched(self):
        groups = image_groups()
        out, split = self.split(groups)
        self.assertEqual(split, 1)
        self.assertEqual(len(out), TARGET_LAYERS + 5)
        self.assertEqual(out[:TARGET_LAYERS], groups[:TARGET_LAYERS])
        for i, group in enumerate(out[TARGET_LAYERS:]):
            self.assertEqual(group.layer_names, [f"model.layers.{TARGET_LAYERS + i}.self_attn.attn"])
            self.assertIs(group.kv_cache_spec, groups[-1].kv_cache_spec)

    def test_the_pool_block_shrinks_from_five_drafter_pages_to_one(self):
        groups = image_groups()
        self.assertEqual(bytes_per_block(groups), 5 * DRAFT_PAGE)
        out, _ = self.split(groups)
        self.assertEqual(bytes_per_block(out), DRAFT_PAGE)

    def test_a_group_with_a_target_layer_in_it_is_kept_whole(self):
        mixed = Group(["model.layers.47.attn", "model.layers.48.self_attn.attn"], Spec(DRAFT_PAGE))
        out, split = self.split([mixed])
        self.assertEqual(split, 0)
        self.assertEqual(out, [mixed])

    def test_a_drafter_group_of_one_layer_is_already_split(self):
        single = Group(["model.layers.50.self_attn.attn"], Spec(DRAFT_PAGE))
        out, split = self.split([single])
        self.assertEqual(split, 0)
        self.assertEqual(out, [single])


class InstalledHook(unittest.TestCase):
    def setUp(self):
        self.added = []
        self.groups = image_groups()
        self.kcu, self.other = stub_kv_cache_utils(self.groups)
        for name, module in (
            ("vllm", types.ModuleType("vllm")),
            ("vllm.v1", types.ModuleType("vllm.v1")),
            ("vllm.v1.core", types.ModuleType("vllm.v1.core")),
            ("vllm.v1.core.kv_cache_utils", self.kcu),
            ("vllm.v1.worker.gpu.cudagraph_utils", self.other),
        ):
            self.assertNotIn(name, sys.modules, f"{name} is already imported; the stub would hide it")
            sys.modules[name] = module
            self.added.append(name)
        self.kvgroups = load_hook()
        self.kvgroups.install()

    def tearDown(self):
        for name in self.added:
            sys.modules.pop(name, None)
        for name in [n for n in sys.modules if n.startswith("flash_local_dflash2")]:
            sys.modules.pop(name, None)

    def test_every_module_holding_the_function_gets_the_wrapper(self):
        self.assertTrue(getattr(self.kcu.get_kv_cache_groups, "_df2_kvgroups", False))
        self.assertIs(self.other.get_kv_cache_groups, self.kcu.get_kv_cache_groups)

    def test_under_dflash_the_drafter_group_is_split(self):
        with self.assertLogs("vllm.flash_local_dflash2.kvgroups", level=logging.INFO) as logs:
            out = self.kcu.get_kv_cache_groups(Config("dflash"), all_specs(self.groups))
        self.assertEqual(len(out), TARGET_LAYERS + 5)
        self.assertEqual(bytes_per_block(out), DRAFT_PAGE)
        self.assertEqual(self.kcu.calls, [sorted(all_specs(self.groups))], "called through once")
        self.assertIn(f"{5 * DRAFT_PAGE} -> {DRAFT_PAGE}", logs.output[0])

    def test_under_mtp_or_without_speculation_nothing_changes(self):
        for config in (Config("mtp"), Config(None)):
            out = self.kcu.get_kv_cache_groups(config, all_specs(self.groups))
            self.assertEqual(out, self.groups)

    def test_a_second_install_does_not_wrap_the_wrapper(self):
        wrapped = self.kcu.get_kv_cache_groups
        self.kvgroups._installed = False
        self.kvgroups.install()
        self.assertIs(self.kcu.get_kv_cache_groups, wrapped)
        out = self.kcu.get_kv_cache_groups(Config("dflash"), all_specs(self.groups))
        self.assertEqual(len(self.kcu.calls), 1)
        self.assertEqual(len(out), TARGET_LAYERS + 5)


class LoadsWithoutVllm(unittest.TestCase):
    def test_importing_the_hook_imports_no_vllm(self):
        before = {n for n in sys.modules if n == "vllm" or n.startswith("vllm.")}
        load_hook()
        after = {n for n in sys.modules if n == "vllm" or n.startswith("vllm.")}
        self.assertEqual(after, before)

    def test_register_installs_the_hook_after_the_diagnostics(self):
        source = (PLUGIN / "flash_local_dflash2" / "plugin.py").read_text()
        self.assertIn("from .kvgroups import install as install_kv_groups", source)
        self.assertLess(source.index("install_kv_diagnostics()"), source.index("install_kv_groups()"))


if __name__ == "__main__":
    unittest.main()
