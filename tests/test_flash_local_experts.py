#!/usr/bin/env python3
"""The per-expert routing counter the flash-local lane mounts (flash-local-experts/), and the
cockpit's reading of what it writes.

The count itself was run against the pinned image's own torch and router classes on the
reference box's CPU (2026-11-10): index_add_ with int32 and int64 ids, 3 us a call, the file
whole and world-readable. What is held here runs without torch: the counting logic against
fakes with torch's call shape, the file's format from the plugin's writer to the cockpit's
reader, the plugin's hook on the runner's constructor, and the unit and installer that put
the folder in front of the engine.
"""
import importlib.util
import json
import os
import pathlib
import re
import struct
import sys
import tempfile
import types
import unittest
from array import array

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[1]
PLUGIN = REPO / "flash-local-experts"
sys.path.insert(0, str(PLUGIN))
from flash_local_experts import counter, plugin, wire  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


reader = load("cockpit_experts_reader", REPO / "dashboard" / "experts.py")


# ── fakes with torch's call shape: an int32 matrix, its row views, a flat vector ────────
class Vec:
    def __init__(self, data, off=0, n=None):
        self.data, self.off = data, off
        self.n = len(data) - off if n is None else n

    def numel(self):
        return self.n

    def reshape(self, _):
        return self

    def __getitem__(self, sl):
        start, stop, _ = sl.indices(self.n)
        return Vec(self.data, self.off + start, stop - start)

    def index_add_(self, dim, ids, src):
        assert dim == 0 and src.n == ids.n, "index_add_ takes one value per id"
        for k in range(ids.n):
            i = ids.data[ids.off + k]
            assert 0 <= i < self.n, f"index_add_: index {i} out of a row of {self.n} (the GPU asserts and the engine dies)"
            self.data[self.off + i] += src.data[src.off + k]
        return self

    def copy_(self, other):
        assert other.n == self.n
        self.data[self.off:self.off + self.n] = array("i", other.tolist())
        return self

    def clamp_(self, lo, hi):
        for k in range(self.n):
            self.data[self.off + k] = min(hi, max(lo, self.data[self.off + k]))
        return self

    def tolist(self):
        return list(self.data[self.off:self.off + self.n])


class Mat:
    """A row-major int32 matrix and its views: rows, or rows and columns, as torch slices them."""

    def __init__(self, rows, cols, data=None, off=0, stride=None):
        self.shape = (rows, cols)
        self.data = data if data is not None else array("i", [0] * (rows * cols))
        self.off, self.stride = off, cols if stride is None else stride

    def __getitem__(self, i):
        rows, cols = self.shape
        rs, cs = i if isinstance(i, tuple) else (i, slice(None))
        if isinstance(rs, slice):
            r0, r1, _ = rs.indices(rows)
            c0, c1, _ = cs.indices(cols)
            return Mat(r1 - r0, c1 - c0, self.data, self.off + r0 * self.stride + c0, self.stride)
        c0, c1, _ = cs.indices(cols)
        return Vec(self.data, self.off + rs * self.stride + c0, c1 - c0)

    def rows(self):
        return [self[r].tolist() for r in range(self.shape[0])]

    def copy_(self, other, non_blocking=False):
        assert other.shape == self.shape
        for r, row in enumerate(other.rows()):
            self[r].copy_(Vec(array("i", row)))
        self.copied = True
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return array("i", [x for row in self.rows() for x in row]).tobytes()


def ids(*rows):
    return Vec(array("i", [e for r in rows for e in r]))


SPARE = 2   # the test matrix's columns past the experts (the engine's: counter.SPARE_COLUMNS)


def make(experts=8, top_k=2, capacity=3, tokens=4, spare=SPARE):
    room = tokens * (top_k + spare)
    c = counter.Counter(Mat(capacity, experts + spare), Vec(array("i", [1] * room)), Vec(array("i", [0] * room)),
                        Mat(capacity, experts + spare), experts, top_k, {"model": "fake", "num_hidden_layers": 2})
    c.path = os.path.join(tempfile.mkdtemp(prefix="experts-"), "qwen38-experts.bin")
    return c


class TheCount(unittest.TestCase):
    def test_each_runner_gets_the_next_row_and_its_calls_count_every_id(self):
        c = make()
        r0, r1 = c.add_row("model.layers.0.mlp.experts"), c.add_row("model.layers.1.mlp.experts")
        self.assertEqual((r0, r1, c.names), (0, 1, ["model.layers.0.mlp.experts", "model.layers.1.mlp.experts"]))
        f0, f1 = c.capture_fn(r0), c.capture_fn(r1)
        f0(ids([1, 5], [5, 7]))          # two tokens, top 2 each
        f1(ids([0, 0]))                  # one token twice to expert 0
        f0(ids([1, 2]))
        self.assertEqual(c.counts[0].tolist(), [0, 2, 1, 0, 0, 2, 0, 1] + [0] * SPARE)
        self.assertEqual(c.counts[1].tolist(), [2, 0, 0, 0, 0, 0, 0, 0] + [0] * SPARE)

    def test_an_id_past_the_experts_lands_in_the_spare_columns_not_in_an_assert(self):
        """The image's router appends the fused shared expert's id, num_experts, to each token's
        ids (the boot of 2026-11-10 died on it); any id is clamped into the matrix."""
        c = make()
        f = c.capture_fn(c.add_row("model.layers.0.mlp.experts"))
        f(ids([1, 8], [3, 8]))           # two tokens, one routed expert each plus the shared expert, id 8 of 8
        f(ids([5, 40], [-3, 9]))         # far past the matrix, and negative: clamped to its last and first column
        self.assertEqual(c.counts[0].tolist(), [1, 1, 0, 1, 0, 1, 0, 0] + [2, 2], "column 8: the shared expert twice; column 9: id 9 and the clamped 40")
        f(ids([0, 0]))
        d = wire.unpack(c.frame())
        self.assertEqual((d["experts"], list(d["counts"].cast("i"))), (8, [3, 1, 0, 1, 0, 1, 0, 0]), "the file carries the experts' columns alone")
        with self.assertRaises(ValueError):
            counter.Counter(Mat(3, 8), Vec(array("i", [1] * 8)), Vec(array("i", [0] * 8)), Mat(3, 8), 8, 2, {})

    def test_a_batch_wider_than_planned_is_left_uncounted_not_grown(self):
        c = make(tokens=1)   # the ones vector holds 1 x (2 + 2) = 4 ids
        f = c.capture_fn(c.add_row("model.layers.0.mlp.experts"))
        f(ids([1, 2], [3, 4]))
        f(ids([1, 2], [3, 4], [5, 6]))   # 6 ids, wider: skipped, no allocation on the hot path
        self.assertEqual(c.counts[0].tolist(), [0, 1, 1, 1, 1, 0, 0, 0] + [0] * SPARE)

    def test_the_first_rows_call_copies_the_whole_matrix_out_after_its_own_count(self):
        c = make()
        f0 = c.capture_fn(c.add_row("model.layers.0.mlp.experts"))
        f1 = c.capture_fn(c.add_row("model.layers.1.mlp.experts"))
        f1(ids([3, 3]))
        self.assertFalse(getattr(c.host, "copied", False), "only row 0 queues the copy")
        f0(ids([1, 1]))
        self.assertTrue(c.host.copied)
        self.assertEqual(c.host[1].tolist(), [0, 0, 0, 2, 0, 0, 0, 0] + [0] * SPARE)
        self.assertEqual(c.host[0].tolist(), [0, 2, 0, 0, 0, 0, 0, 0] + [0] * SPARE, "the copy carries row 0's own count of this call")

    def test_a_slot_already_taken_is_called_after_the_count(self):
        c, seen = make(), []
        f = c.capture_fn(c.add_row("model.layers.0.mlp.experts"), previous=seen.append)
        t = ids([2, 3])
        f(t)
        self.assertEqual((seen, c.counts[0].tolist()[2:4]), ([t], [1, 1]))

    def test_beyond_the_capacity_no_row_is_given(self):
        c = make(capacity=2)
        self.assertEqual([c.add_row(f"model.layers.{i}.mlp.experts") for i in range(3)], [0, 1, None])


class TheFile(unittest.TestCase):
    def test_the_plugin_writes_what_the_cockpit_reads(self):
        c = make()
        f0 = c.capture_fn(c.add_row("model.layers.0.mlp.experts"))
        c.add_row("mtp.layers.0.mlp.experts")
        f0(ids([1, 5], [5, 7]))
        f0(ids([0, 0]))                  # every row-0 call copies the matrix out, this count included
        c.write_once()
        self.assertEqual(oct(os.stat(c.path).st_mode)[-3:], "644", "the operator's cockpit reads a root-written file")
        self.assertEqual(os.listdir(os.path.dirname(c.path)), ["qwen38-experts.bin"], "no sibling left behind")
        data = pathlib.Path(c.path).read_bytes()
        head = reader.read_header(c.path)
        self.assertEqual((head["present"], head["rows"], head["experts"], head["top_k"], head["model"], head["seq"]),
                         (True, 2, 8, 2, "fake", 1))
        self.assertLess(head["age"], 5)
        frame, age = reader.read_frame(c.path)
        self.assertEqual(frame, data)
        back = wire.unpack(data)
        self.assertEqual(back["meta"]["rows"], ["model.layers.0.mlp.experts", "mtp.layers.0.mlp.experts"])
        counts = list(back["counts"].cast("i"))
        self.assertEqual(counts[:8], [2, 1, 0, 0, 0, 2, 0, 1], "the matrix as of the last copy out")
        self.assertEqual(counts[8:], [0] * 8)
        self.assertEqual(len(data) % 4, 0, "the counts stay 4-byte aligned behind the JSON block")

    def test_the_json_block_is_padded_so_the_counts_stay_aligned(self):
        for meta in ({"a": 1}, {"ab": 12}, {"abc": 123}, {"abcd": 1234}):
            data = wire.pack(meta, b"\x07\0\0\0" * 3, 1, 3, 1, 1, 0.0, 1)
            self.assertEqual(len(data) % 4, 0, meta)
            self.assertEqual(list(wire.unpack(data)["counts"].cast("i")), [7, 7, 7], meta)
            self.assertEqual(reader.whole(data)["meta"], meta)

    def test_what_is_not_a_whole_file_is_none_on_both_sides(self):
        data = wire.pack({"model": "x"}, b"\0" * 16, 2, 2, 1, 3, 1.0, 1)
        for bad in (data[:-1], data + b"\0", b"XXXX" + data[4:], data[:40], b"", struct.pack("<4sI", b"QEXP", 2) + data[8:]):
            self.assertIsNone(wire.unpack(bad))
            self.assertIsNone(reader.whole(bad))
        self.assertIsNone(reader.parse_header(wire.HEADER.pack(b"QEXP", 1, 1, 1, 1, 1, 1, 0.0, 4, 0) + b"[1] "),
                          "a JSON block that is not an object")

    def test_the_cockpit_reads_no_file_and_a_foreign_file_as_absent(self):
        d = tempfile.mkdtemp(prefix="experts-")
        self.assertEqual(reader.read_header(os.path.join(d, "none"))["present"], False)
        self.assertEqual(reader.read_frame(os.path.join(d, "none")), (None, 0.0))
        p = os.path.join(d, "foreign")
        pathlib.Path(p).write_bytes(b"not a counter file at all, " * 8)
        self.assertEqual((reader.read_header(p)["present"], reader.read_header(p).get("unreadable")), (False, True))
        self.assertEqual(reader.read_frame(p), (None, 0.0))

    def test_the_two_formats_are_one(self):
        self.assertEqual((wire.HEADER.format, wire.HEADER.size, wire.MAGIC, wire.VERSION),
                         (reader.HEADER.format, reader.HEADER.size, reader.MAGIC, reader.VERSION))


# ── the hook ───────────────────────────────────────────────────────────────────────────
class Router:
    def __init__(self, experts=8, top_k=2, slot=True):
        self.global_num_experts, self.top_k, self.capture_fn = experts, top_k, None
        if not slot:
            del self.set_capture_fn   # noqa: an instance attribute hides the method... see below

    def set_capture_fn(self, fn):
        self.capture_fn = fn


class NoSlotRouter:
    global_num_experts, top_k = 8, 2


def runner_class():
    class Runner:
        def __init__(self, layer_name, router):
            self.layer_name, self.router = layer_name, router
    return Runner


class TheHook(unittest.TestCase):
    def setUp(self):
        plugin._STATE["counter"] = None

    def test_each_runner_built_binds_its_router_to_the_next_row(self):
        Runner, made = runner_class(), []
        c = make()
        plugin.install(Runner, make=lambda router: made.append(router) or c)
        a = Runner("model.layers.0.mlp.experts", Router())
        b = Runner("model.layers.1.mlp.experts", Router())
        self.assertEqual((len(made), c.names), (1, ["model.layers.0.mlp.experts", "model.layers.1.mlp.experts"]))
        self.assertTrue(callable(a.router.capture_fn) and callable(b.router.capture_fn))
        b.router.capture_fn(ids([4, 4]))
        self.assertEqual(c.counts[1].tolist(), [0, 0, 0, 0, 2, 0, 0, 0] + [0] * SPARE)
        self.assertIsNotNone(c._thread, "the writer starts with the first runner")

    def test_a_router_without_the_slot_or_of_another_width_is_left_alone(self):
        Runner = runner_class()
        c = make()
        plugin.install(Runner, make=lambda router: c)
        Runner("model.layers.0.mlp.experts", Router())
        other = Runner("model.layers.1.mlp.experts", Router(experts=16))
        none = Runner("model.layers.2.mlp.experts", NoSlotRouter())
        self.assertEqual(c.names, ["model.layers.0.mlp.experts"])
        self.assertIsNone(other.router.capture_fn)
        self.assertFalse(hasattr(none.router, "capture_fn"))

    def test_build_puts_the_host_copy_on_the_cpu_whatever_device_the_model_loads_under(self):
        """vLLM builds the model inside a device context: a tensor made with no device lands on the
        GPU, and the GPU cannot be pinned (the boot of 2026-11-10 counted nothing for that)."""
        class T:
            def __init__(self, shape, device, pinned=False):
                self.shape, self.device, self.pinned = shape, device, pinned

            def pin_memory(self):
                if self.device != "cpu":
                    raise RuntimeError("Only dense CPU tensors can be pinned")
                return T(self.shape, self.device, True)

        def tensor(shape, dtype=None, device=None, pin_memory=False):
            dev = "cuda:0" if device is None else str(device)   # the context's default: the GPU
            if pin_memory and dev != "cpu":
                raise RuntimeError("Only dense CPU tensors can be pinned")
            return T(shape if isinstance(shape, tuple) else (shape,), dev, pin_memory)

        fake = types.ModuleType("torch")
        fake.int32 = "int32"
        fake.zeros = fake.ones = tensor
        fake.device = lambda kind, index=None: f"{kind}:{index}" if index is not None else kind
        fake.cuda = types.SimpleNamespace(is_available=lambda: True, current_device=lambda: 0)
        cfg = types.SimpleNamespace(model_config=types.SimpleNamespace(hf_text_config=types.SimpleNamespace(num_hidden_layers=2)),
                                    scheduler_config=types.SimpleNamespace(max_num_batched_tokens=16))
        saved = sys.modules.get("torch")
        sys.modules["torch"] = fake
        try:
            c = counter.build(cfg, 8, 2)
        finally:
            if saved is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = saved
        self.assertEqual((c.counts.device, c.ones.device, c.scratch.device, c.host.device, c.host.pinned), ("cuda:0", "cuda:0", "cuda:0", "cpu", True))
        self.assertEqual((c.capacity, c.width, c.counts.shape, c.host.shape, c.ones.shape), (2 + 4, 8 + 8, (6, 16), (6, 16), (16 * (2 + 8),)))

    def test_a_counter_that_cannot_be_built_leaves_the_model_built(self):
        Runner = runner_class()

        def boom(router):
            raise RuntimeError("no device")

        plugin.install(Runner, make=boom)
        r = Runner("model.layers.0.mlp.experts", Router())
        self.assertEqual((r.layer_name, r.router.capture_fn), ("model.layers.0.mlp.experts", None))

    def test_installed_twice_the_constructor_is_wrapped_once(self):
        Runner = runner_class()
        plugin.install(Runner, make=lambda router: make())
        once = Runner.__init__
        plugin.install(Runner, make=lambda router: make())
        self.assertIs(Runner.__init__, once)

    def test_register_wraps_vllms_runner_and_a_kill_switch_leaves_it(self):
        Runner = runner_class()
        mod = types.ModuleType("vllm.model_executor.layers.fused_moe.runner.moe_runner")
        mod.MoERunner = Runner
        names = ["vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.fused_moe",
                 "vllm.model_executor.layers.fused_moe.runner", mod.__name__]
        saved = {n: sys.modules.get(n) for n in names}
        for n in names[:-1]:
            sys.modules[n] = types.ModuleType(n)
        sys.modules[mod.__name__] = mod
        try:
            os.environ["QWEN38_EXPERTS"] = "0"
            plugin.register()
            self.assertFalse(getattr(Runner, plugin._MARKER, False))
            del os.environ["QWEN38_EXPERTS"]
            plugin.register()
            self.assertTrue(getattr(Runner, plugin._MARKER, False))
        finally:
            os.environ.pop("QWEN38_EXPERTS", None)
            for n, m in saved.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m

    def test_the_entry_point_names_register(self):
        text = (PLUGIN / "flash_local_experts-0.0.0.dist-info" / "entry_points.txt").read_text()
        self.assertIn("[vllm.general_plugins]\nflash_local_experts = flash_local_experts.plugin:register", text)
        self.assertTrue(callable(plugin.register))

    def test_the_models_shape_is_read_with_every_field_optional(self):
        hf = types.SimpleNamespace(model_type="qwen4_exp_text", num_hidden_layers=48, layer_types=["linear_attention"] * 3 + ["full_attention"],
                                   hidden_size=2560, num_experts=512, num_experts_per_tok=10, hc_count=4, ple_layer_ids=[2],
                                   mtp_num_hidden_layers=1, vocab_size=248320, shared_expert_intermediate_size=640)
        cfg = types.SimpleNamespace(
            model_config=types.SimpleNamespace(hf_text_config=hf, model="nvidia/Qwen3.8-Flash-Next-FP4-Local", served_model_name=["qwen3.8-flash-next-fp4"]),
            speculative_config=types.SimpleNamespace(method="dflash", num_speculative_tokens=4, draft_model_config=types.SimpleNamespace(model="incoai/Qwen3.8-Flash-Next-DFlash2")),
            scheduler_config=types.SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=4096))
        meta = counter.model_meta(cfg)
        self.assertEqual((meta["served_model_name"], meta["num_experts"], meta["spec_method"], meta["draft_model"], meta["layer_types"][3], meta["ple_layer_ids"]),
                         ("qwen3.8-flash-next-fp4", 512, "dflash", "incoai/Qwen3.8-Flash-Next-DFlash2", "full_attention", [2]))
        bare = counter.model_meta(types.SimpleNamespace(model_config=types.SimpleNamespace(), scheduler_config=types.SimpleNamespace()))
        self.assertEqual((bare["num_experts"], bare["spec_method"], bare["model"]), (None, None, None))
        json.dumps(meta)


# ── the unit and the installer ─────────────────────────────────────────────────────────
class TheLaneMountsIt(unittest.TestCase):
    def test_the_unit_mounts_the_folder_read_only_and_puts_it_on_the_path(self):
        unit = (REPO / "qwen38-flash-local.service.template").read_text()
        # the lane's installer renders the line (its dflash2 target adds a plugin of its own after it)
        installer = (REPO / "install-flash-local.sh").read_text()
        self.assertIn("-e PYTHONPATH=__PYTHONPATH__", unit)
        self.assertIn('PYTHONPATH_LINE="/opt/qwen38-auth:/opt/experts-plugin"', installer)
        self.assertIn('PYTHONPATH_LINE="/opt/qwen38-auth:/opt/experts-plugin:/opt/df2-plugin"', installer)
        self.assertIn("--mount type=bind,source=__HOME__/.config/qwen38/flash-local-experts,target=/opt/experts-plugin,readonly", unit)
        self.assertIn("--ipc=host", unit, "the file under /dev/shm reaches the host through the shared IPC namespace")

    def test_the_installer_copies_the_folder_for_the_install_and_the_switch(self):
        text = (REPO / "install-flash-local.sh").read_text()
        install = text[text.index("install_service(){"):text.index("\n}\n", text.index("install_service(){"))]
        stage = text[text.index('if [ "$MODE" = stage ]; then'):]
        self.assertIn("put_experts_plugin", install)
        self.assertIn("put_experts_plugin", stage)
        self.assertLess(install.index("put_experts_plugin"), install.index("render_unit"), "copied before the unit is rendered")
        self.assertIn('for f in $(experts_files); do starts+=("$CONFIG_DIR/flash-local-experts/$f"); done', install,
                      "its files are among what the engine reads at its start")

    def test_a_stage_puts_the_copy_in_and_renders_the_mount_from_the_home(self):
        stage_tests = load("stage_tests", REPO / "tests" / "test_flash_local_stage.py")
        box = stage_tests.Box()
        try:
            rc, out = box.stage(QWEN38_MEMORY_CLASS="128")
            self.assertEqual(rc, 0, out)
            copy = box.home / ".config" / "qwen38" / "flash-local-experts"
            self.assertEqual(sorted(p.relative_to(copy).as_posix() for p in copy.rglob("*") if p.is_file()),
                             sorted(p.relative_to(PLUGIN).as_posix() for p in PLUGIN.rglob("*") if p.is_file() and "__pycache__" not in p.parts))
            unit = box.out.read_text()
            self.assertIn(f"--mount type=bind,source={box.home}/.config/qwen38/flash-local-experts,target=/opt/experts-plugin,readonly", unit)
            self.assertIn("-e PYTHONPATH=/opt/qwen38-auth:/opt/experts-plugin", unit)
            # a copy that drifted is made again from the checkout
            (copy / "flash_local_experts" / "plugin.py").write_text("# drifted\n")
            rc, out = box.stage(QWEN38_MEMORY_CLASS="128")
            self.assertEqual(rc, 0, out)
            self.assertEqual((copy / "flash_local_experts" / "plugin.py").read_text(), (PLUGIN / "flash_local_experts" / "plugin.py").read_text())
        finally:
            import shutil
            shutil.rmtree(box.t, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
