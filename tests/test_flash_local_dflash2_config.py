#!/usr/bin/env python3
"""flash_local_dflash2.config, the hook that lets this target be drafted at all.

The pinned image's hook for the target's arch refuses every speculative method but its own
MTP and n-gram proposers, and it runs while the engine's config is built, so the dflash2
target died there before its capture subclass was ever resolved (the reference box,
2026-11-07). The lane opens that whitelist from outside the image, on PYTHONPATH, and what
this checks is how narrowly: the image's own hook decides everything the image checks, the
step-over is its method whitelist alone, only for the one method the lane stitches in, only
for the lane's target arches, and it fails closed when the image's words move.

No vLLM and no GPU: the hook is a subclass of whatever class the image's map holds, so the
image's hook is stood in for by a class with the shape the pinned one has, checks first and
whitelist last, and the engine's config by its one field the hook reads.
"""
import importlib
import pathlib
import sys
import types
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
PLUGIN = REPO / "flash-local-dflash2"
sys.dont_write_bytecode = True  # the lane copies this folder as the repo has it
sys.path.insert(0, str(PLUGIN))
from flash_local_dflash2 import config as df2_config  # noqa: E402  (the plugin's own module)

# The image's own sentences, word for word as the pinned image writes them
# (vllm/model_executor/models/config.py of eugr/spark-vllm-b12x@03a95654), because matching
# the first is the whole difference between stepping over a policy and swallowing a fault.
GATE = ("Qwen4Exp speculative decoding supports only its native MTP checkpoint and linear "
        "n-gram proposers")
PLE_PP = ("Qwen4Exp N-gram PLE embedding requires pipeline_parallel_size=1 because non-first "
          "pipeline ranks do not receive the raw input_ids it needs. Please run with PP=1.")
IMAGE_METHODS = {"mtp", "ngram", "ngram_gpu"}


def image_hook(raises=None, methods=IMAGE_METHODS):
    """A stand-in for the image's hook: its checks, then the whitelist, in that order, the
    way the pinned one is written. Each call records the method it was asked about."""
    calls = []

    class Qwen4ExpForConditionalGenerationConfig:
        @staticmethod
        def verify_and_update_config(vllm_config):
            spec = getattr(vllm_config, "speculative_config", None)
            calls.append(getattr(spec, "method", None))
            if raises is not None:
                raise raises
            if spec is not None and spec.method not in methods:
                raise NotImplementedError(GATE)

    Qwen4ExpForConditionalGenerationConfig.calls = calls
    return Qwen4ExpForConditionalGenerationConfig


def vllm_config(method):
    """Only what the hook reads: the speculative config's method, or no speculative config."""
    return types.SimpleNamespace(speculative_config=None if method is None
                                 else types.SimpleNamespace(method=method))


def the_map():
    """The image's map as the pinned image has it for this arch, plus arches that are not
    the lane's business and must come out of install() as they went in."""
    return {
        "Qwen4ExpForConditionalGeneration": image_hook(),
        "Qwen3_8FlashNextForConditionalGeneration": image_hook(),
        "Qwen4ExpMTP": image_hook(),
        "DeepseekV4ForCausalLM": image_hook(),
    }


class TheGateHook(unittest.TestCase):
    def test_importing_the_plugin_needs_no_vllm(self):
        # the entry point is loaded in every process that loads plugins; vLLM's model and
        # config code is only ever imported inside register()
        self.assertNotIn("vllm", sys.modules)

    def test_only_the_lane_targets_arches_are_repointed(self):
        m = the_map()
        before = dict(m)
        changed = df2_config.install(m)
        self.assertEqual(changed, list(df2_config.TARGET_ARCHES))
        for arch in df2_config.TARGET_ARCHES:
            self.assertTrue(issubclass(m[arch], before[arch]),
                            "the image's hook has to stay the one that decides")
        for arch in set(m) - set(df2_config.TARGET_ARCHES):
            self.assertIs(m[arch], before[arch], f"{arch} is not this lane's arch")

    def test_the_image_runs_its_own_checks_under_a_dflash_draft(self):
        for arch in df2_config.TARGET_ARCHES:
            m = the_map()
            base = m[arch]
            df2_config.install(m)
            m[arch].verify_and_update_config(vllm_config("dflash"))
            self.assertEqual(base.calls, ["dflash"], "the image's hook was never asked")

    def test_the_methods_the_image_allows_still_answer_as_the_image_answers(self):
        for method in sorted(IMAGE_METHODS):
            m = the_map()
            df2_config.install(m)
            self.assertIsNone(m["Qwen4ExpForConditionalGeneration"].verify_and_update_config(
                vllm_config(method)))

    def test_a_method_the_lane_does_not_stitch_in_is_still_refused(self):
        # fail closed: another proposer must not find the gate open because this plugin
        # happens to be on the PYTHONPATH of the process that reads the map
        for method in ("eagle3", "eagle", "medusa", "dspark", "mtp_bad"):
            m = the_map()
            df2_config.install(m)
            with self.subTest(method=method):
                with self.assertRaises(NotImplementedError) as ctx:
                    m["Qwen4ExpForConditionalGeneration"].verify_and_update_config(
                        vllm_config(method))
                self.assertIn("supports only", str(ctx.exception))

    def test_a_refusal_that_is_not_the_whitelist_goes_out(self):
        # the PLE/PP refusal comes from the same hook, above the whitelist: a dflash draft
        # must not silence it, or the engine loads its weights to die at the first forward
        m = the_map()
        base = image_hook(raises=NotImplementedError(PLE_PP))
        m["Qwen4ExpForConditionalGeneration"] = df2_config._hook_with_the_gate_opened(
            base, df2_config.DFLASH_METHOD)
        with self.assertRaises(NotImplementedError) as ctx:
            m["Qwen4ExpForConditionalGeneration"].verify_and_update_config(vllm_config("dflash"))
        self.assertIn("pipeline_parallel_size=1", str(ctx.exception))

    def test_a_reworded_gate_is_not_swallowed(self):
        # the safe failure when the image rewords its refusal: the lane does not recognise
        # it, so the operator reads the refusal again instead of booting into a hole
        base = image_hook(raises=NotImplementedError("Qwen4Exp drafts with nothing else"))
        hook = df2_config._hook_with_the_gate_opened(base, df2_config.DFLASH_METHOD)
        with self.assertRaises(NotImplementedError):
            hook.verify_and_update_config(vllm_config("dflash"))

    def test_no_speculative_config_is_the_images_affair(self):
        m = the_map()
        df2_config.install(m)
        self.assertIsNone(
            m["Qwen4ExpForConditionalGeneration"].verify_and_update_config(vllm_config(None)))

    def test_registering_twice_wraps_the_hook_once(self):
        # vLLM loads the general plugins more than once in a process, and the engine core
        # and each worker both call register(); a hook wrapped around itself is a hook whose
        # base is a hook
        m = the_map()
        base = m["Qwen4ExpForConditionalGeneration"]
        df2_config.install(m)
        once = m["Qwen4ExpForConditionalGeneration"]
        self.assertEqual(df2_config.install(m), [])
        self.assertIs(m["Qwen4ExpForConditionalGeneration"], once)
        m["Qwen4ExpForConditionalGeneration"].verify_and_update_config(vllm_config("dflash"))
        self.assertEqual(base.calls, ["dflash"], "the image's hook ran twice")

    def test_the_gate_is_recognised_by_its_words_alone(self):
        self.assertTrue(df2_config.opens_the_method_gate(NotImplementedError(GATE)))
        self.assertFalse(df2_config.opens_the_method_gate(NotImplementedError(PLE_PP)))
        self.assertFalse(df2_config.opens_the_method_gate(ValueError(GATE)),
                         "a ValueError of the same words is a different fault")

    def test_the_method_it_opens_is_the_one_the_image_names_for_this_drafter(self):
        # the image's own name for the DFlash family (config/speculative.py's
        # DFlashModelTypes); what makes a draft DFlash2 is its arch, not its method
        self.assertEqual(df2_config.DFLASH_METHOD, "dflash")


class WhatThePluginRegisters(unittest.TestCase):
    """register(), the entry point the lane's mount puts on PYTHONPATH: it has to hand vLLM
    both halves of the target, the capture subclass and the hook beside it, in the map vLLM
    itself reads. vLLM is stood in for module by module, so the imports inside register()
    are the ones the test drives."""

    def setUp(self):
        self.registered = []

        class ModelRegistry:
            @classmethod
            def register_model(cls, arch, cls_ref):
                self.registered.append((arch, cls_ref))

        self.map = the_map()
        self.saved = {}
        parts = {
            "vllm": {},
            "vllm.model_executor": {},
            "vllm.model_executor.models": {"ModelRegistry": ModelRegistry},
            "vllm.model_executor.models.config": {"MODELS_CONFIG_MAP": self.map},
        }
        for name, attrs in parts.items():
            self.saved[name] = sys.modules.pop(name, None)
            mod = types.ModuleType(name)
            for attr, value in attrs.items():
                setattr(mod, attr, value)
            sys.modules[name] = mod
        # the plugin itself is the thing under test: gone from the cache so that
        # register() is imported fresh, and never stood in for
        self.saved["flash_local_dflash2.plugin"] = sys.modules.pop("flash_local_dflash2.plugin", None)

    def tearDown(self):
        for name, mod in self.saved.items():
            sys.modules.pop(name, None)
            if mod is not None:
                sys.modules[name] = mod

    def register(self):
        return importlib.import_module("flash_local_dflash2.plugin")

    def test_register_gives_vllm_the_target_and_opens_its_gate(self):
        self.register().register()
        self.assertIn(("Qwen4ExpForConditionalGeneration",
                       "flash_local_dflash2.model:DFlashQwen4ExpForConditionalGeneration"),
                      self.registered, "the capture subclass was not handed over")
        for arch in df2_config.TARGET_ARCHES:
            hook = self.map[arch]
            hook.verify_and_update_config(vllm_config("dflash"))  # the refusal is gone
            with self.assertRaises(NotImplementedError):
                hook.verify_and_update_config(vllm_config("eagle3"))

    def test_the_map_vllm_reads_is_the_one_that_changed(self):
        # vllm/config/vllm.py reads MODELS_CONFIG_MAP by name when it verifies: a map bound
        # to something else here would leave the image's refusal in force
        self.register().register()
        import vllm.model_executor.models.config as image_config
        self.assertIs(self.map, image_config.MODELS_CONFIG_MAP)

    def test_the_arch_it_registers_is_the_one_the_target_names(self):
        # both halves name one target: the arch the checkpoint's config.json carries is the
        # arch the capture subclass is registered under, and the one the hook opens
        self.register().register()
        self.assertEqual([arch for arch, _ in self.registered], [df2_config.TARGET_ARCHES[0]])
        self.assertIn(df2_config.TARGET_ARCHES[0], self.map)


if __name__ == "__main__":
    unittest.main()
