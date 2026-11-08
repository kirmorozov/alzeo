#!/usr/bin/env python3
"""The DFlash2 target hands the runner a real tensor, not the image's None.

vLLM feeds a drafter ``last_hidden_states``, and when the target answers to
``get_mtp_target_hidden_states`` it overrides that with what the method returns
and slices it with no None check. The image's Qwen4Exp does answer to that name
- MTP's pre-final-mixer multi-stream state - but allocates it only under
``method == "mtp"``, so under ``method == "dflash"`` the inherited method
returns None and the boot dies in ``determine_available_memory`` with
``TypeError: 'NoneType' object is not subscriptable``, before a token is
drafted. The subclass answers that name itself, with the ordinary final hidden
states in a fixed buffer.

The image is not installed here, so the three modules the plugin imports are
stubbed and the base class is a fake that answers `forward` the way the image's
outer forward does, with what the test hands it: what is under test is this
file's own choices, not the image's. The tensor work needs torch and skips
without it.
"""
import ast
import importlib.util
import pathlib
import sys
import types
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
PLUGIN = REPO / "flash-local-dflash2"


def torch_or_none():
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


TORCH = torch_or_none()


class _IntermediateTensors:
    """Stand-in for vllm.sequence.IntermediateTensors (only isinstance is used)."""


def load_model_module(added=None):
    """Import flash_local_dflash2.model against stubbed vllm modules.

    Every module name this puts into sys.modules is recorded in ``added`` so the
    caller can take them back out: another suite in the same process asserts
    that nothing in it has ever imported vllm, and these stubs are named vllm.*.
    """
    added = [] if added is None else added

    def keep(name, module):
        if name not in sys.modules:
            sys.modules[name] = module
            added.append(name)

    vllm = types.ModuleType("vllm")
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_pp_group = lambda: types.SimpleNamespace(world_size=1)
    models = types.ModuleType("vllm.models")
    qwen = types.ModuleType("vllm.models.qwen4_exp.nvidia.model")

    class Base:
        def __init__(self, *, vllm_config, prefix="model"):
            self.vllm_config = vllm_config
            self.prefix = prefix
            # what the image's own forward would hand back; a test sets it
            self.forward_returns = None
            self.forward_kwargs = None
            self.language_model = types.SimpleNamespace(
                vllm_config=vllm_config,
                model=vllm_model(vllm_config),
                scheduler_config=vllm_config.scheduler_config,
                model_config=vllm_config.model_config,
            )

        def forward(self, input_ids, positions, intermediate_tensors=None,
                    inputs_embeds=None, **kwargs):
            # the image's outer forward: deepstack staging, the n-gram kwargs,
            # and the sampled hidden states out
            self.forward_kwargs = dict(intermediate_tensors=intermediate_tensors,
                                       inputs_embeds=inputs_embeds, **kwargs)
            return self.forward_returns

    def vllm_model(vllm_config):
        hidden = vllm_config.hidden_size
        return types.SimpleNamespace(
            config=types.SimpleNamespace(
                hidden_size=hidden, hc_count=vllm_config.hc_count
            ),
            embed_tokens=types.SimpleNamespace(
                weight=types.SimpleNamespace(device=vllm_config.device)
            ),
            layers=[],
            start_layer=0,
            end_layer=48,
            _dflash_aux_buffer=None,
            register_buffer=lambda self, name, tensor, persistent=False: setattr(
                self, name, tensor
            ),
        )

    qwen.Qwen4ExpForConditionalGeneration = Base
    sequence = types.ModuleType("vllm.sequence")
    sequence.IntermediateTensors = _IntermediateTensors
    for name, module in [
        ("vllm", vllm),
        ("vllm.distributed", distributed),
        ("vllm.models", models),
        ("vllm.models.qwen4_exp", types.ModuleType("vllm.models.qwen4_exp")),
        ("vllm.models.qwen4_exp.nvidia", types.ModuleType("vllm.models.qwen4_exp.nvidia")),
        ("vllm.models.qwen4_exp.nvidia.model", qwen),
        ("vllm.sequence", sequence),
    ]:
        keep(name, module)

    capture_path = PLUGIN / "flash_local_dflash2" / "capture.py"
    model_path = PLUGIN / "flash_local_dflash2" / "model.py"
    keep("flash_local_dflash2", _package_module())
    for name, path in [
        ("flash_local_dflash2.capture", capture_path),
        ("flash_local_dflash2.model", model_path),
    ]:
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            keep(name, module)
            spec.loader.exec_module(module)
    return sys.modules["flash_local_dflash2.model"]


def _package_module():
    package = types.ModuleType("flash_local_dflash2")
    package.__path__ = [str(PLUGIN / "flash_local_dflash2")]
    return package


def vllm_config(method="dflash", tokens=16, hidden=8, hc=4, dtype=None):
    t = TORCH
    return types.SimpleNamespace(
        speculative_config=None if method is None else types.SimpleNamespace(method=method),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=tokens),
        model_config=types.SimpleNamespace(dtype=t.float32 if dtype is None else dtype),
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1),
        hidden_size=hidden,
        hc_count=hc,
        device="cpu",
    )


@unittest.skipIf(TORCH is None, "needs torch: the plugin's capture module imports it")
class TargetFeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._stubbed = []
        cls.module = load_model_module(cls._stubbed)
        cls.target_class = cls.module.DFlashQwen4ExpForConditionalGeneration

    @classmethod
    def tearDownClass(cls):
        # The gate suite asserts, for the whole process, that nothing has
        # imported vllm; the stubs that stand in for the image are named vllm.*,
        # so they must not outlive this class.
        for name in cls._stubbed:
            sys.modules.pop(name, None)

    def setUp(self):
        # the runner's own convention: vLLM adds 1 to the drafter's
        # target_layer_ids, so the completed layers 5/15/25/35/45 arrive as
        # 6/16/26/36/46 (eagle3_utils.get_eagle3_aux_layers_from_config)
        self.aux_layers = (6, 16, 26, 36, 46)

    def _target(self, method="dflash", tokens=16, hidden=8, hc=4):
        cfg = vllm_config(method=method, tokens=tokens, hidden=hidden, hc=hc)
        target = self.target_class(vllm_config=cfg)
        # a stand-in aux buffer, as set_aux_hidden_state_layers would register
        inner = target.language_model.model
        t = TORCH
        inner._dflash_aux_buffer = t.zeros(len(self.aux_layers), tokens, hidden)
        target._dflash_capture_slots = len(self.aux_layers)
        return target, cfg

    def test_the_target_answers_get_mtp_target_hidden_states_itself(self):
        # Inheriting the image's method is what crashes: it returns None under
        # method=dflash and the runner slices the result. Defining it here is
        # the fix, so say so in the class, not only in behaviour.
        self.assertIn(
            "get_mtp_target_hidden_states",
            vars(self.target_class),
            "the subclass must define get_mtp_target_hidden_states itself",
        )

    def test_the_feed_is_allocated_for_dflash_only(self):
        # no drafter at all, or the MTP target, gets no extra buffer: MTP keeps
        # the image's own pre-HC capture untouched. Needs no tensor: those paths
        # never allocate.
        for method in (None, "mtp", "ngram"):
            quiet = self.target_class(vllm_config=vllm_config(method=method))
            self.assertIsNone(quiet._dflash_last_hidden_buffer, method)

    @unittest.skipIf(TORCH is None, "needs torch")
    def test_the_dflash_feed_is_the_full_token_window_at_target_width(self):
        tokens, hidden = 16, 8
        target, cfg = self._target(method="dflash", tokens=tokens, hidden=hidden)
        self.assertIsNotNone(target._dflash_last_hidden_buffer)
        self.assertEqual(
            tuple(target._dflash_last_hidden_buffer.shape),
            (cfg.scheduler_config.max_num_batched_tokens, cfg.hidden_size),
        )

    @unittest.skipIf(TORCH is None, "needs torch")
    def test_forward_fills_the_buffer_and_returns_the_pair(self):
        t = TORCH
        tokens, max_tokens = 5, 16
        target, _ = self._target(tokens=max_tokens)
        buffer = target._dflash_last_hidden_buffer
        buffer.fill_(float("nan"))
        hidden_states = t.arange(tokens * target.language_model.model.config.hidden_size).view(
            tokens, -1
        ).to(t.float32)
        target.forward_returns = hidden_states

        out = target.forward(None, t.arange(tokens), inputs_embeds=None)

        self.assertIsInstance(out, tuple)
        hidden, aux = out
        self.assertIs(hidden, hidden_states)
        self.assertEqual(len(aux), len(self.aux_layers))
        # the rows the runner will slice are exactly what the model returned,
        # and the rows past them are untouched: a replayed graph writes the
        # same bytes at the same address
        self.assertTrue(t.equal(buffer[:tokens], hidden_states))
        self.assertTrue(t.isnan(buffer[tokens:]).all())
        self.assertEqual(buffer[:tokens].data_ptr(), buffer.data_ptr())

    @unittest.skipIf(TORCH is None, "needs torch")
    def test_the_pair_is_not_claimed_without_capture(self):
        # The runner asserts a Tensor when it did not ask for aux outputs, so a
        # target that captured nothing must return the plain tensor.
        t = TORCH
        target, _ = self._target()
        target._dflash_capture_slots = 0
        out = target.forward(None, t.arange(3), inputs_embeds=None)
        self.assertNotIsInstance(out, tuple)

    def test_a_target_without_the_capture_says_so(self):
        # The runner calls this whenever the target answers to the name, so a
        # target built for another method fails loudly rather than handing back
        # None for the runner to subscript.
        target = self.target_class(vllm_config=vllm_config(method="mtp"))
        with self.assertRaises(RuntimeError) as ctx:
            target.get_mtp_target_hidden_states()
        self.assertIn("dflash", str(ctx.exception))


class TargetSource(unittest.TestCase):
    """What holds whatever torch is installed: the override is in the source."""

    def test_the_subclass_defines_the_override_itself(self):
        # Inheriting the image's method is what crashes the boot, so the fix has
        # to be a method of this class, not an attribute set somewhere else.
        tree = ast.parse((PLUGIN / "flash_local_dflash2" / "model.py").read_text())
        target = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and node.name == "DFlashQwen4ExpForConditionalGeneration"
        )
        defined = {
            node.name
            for node in target.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("get_mtp_target_hidden_states", defined)
        self.assertIn("forward", defined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
