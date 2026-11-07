#!/usr/bin/env python3
"""Library's Download: for each target a Load can switch to, whether its checkpoint is in the
caches the lanes read (whole for what the lane serves, begun, or absent) and whether its lane
is installed; and the action that fetches it, switch-model.sh <target> --fetch, which changes
nothing else. The target list is the switch's own."""
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
DASH = HERE.parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(DASH))
ENV_BEFORE = dict(os.environ)


def tearDownModule():
    os.environ.clear()
    os.environ.update(ENV_BEFORE)


class Fetch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cockpit-fetch-"))
        (cls.tmp / "config").mkdir()
        (cls.tmp / "config" / "api-key").write_text("k\n")
        closed = "http://127.0.0.1:1"
        os.environ.update(HOME=str(cls.tmp), COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(cls.tmp / "config"),
                          COCKPIT_REPO_DIR=str(REPO), COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0",
                          COCKPIT_AUTOHEAL="0", COCKPIT_ENGINE=closed, COCKPIT_PROXY=closed, COCKPIT_IMAGE=closed)
        spec = importlib.util.spec_from_file_location("cockpit_fetch_under_test", DASH / "cockpit.py")
        cls.cp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cp)
        cls.cp.run = lambda argv, timeout=5.0, merge_err=False: ""
        cls.units = cls.tmp / "units"
        cls.units.mkdir()
        # Every unit the cockpit reads, this test's: the box's own Nemotron unit, once installed, named
        # its real cache (found on the reference box, 2026-10-16), and a fixed list misses the next lane's.
        for name, value in list(vars(cls.cp).items()):
            if name.endswith("_UNIT_PATH") and isinstance(value, Path):
                setattr(cls.cp, name, cls.units / value.name)
        cls.cp.UNIT_PATHS = {"qwen38-sglang.service": cls.units / "qwen38-sglang.service",
                             "qwen38-flash.service": cls.units / "launch-flash.sh"}
        cls.pins = cls.cp.registry_snapshot(max_age=0.0)["pins"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.hub = self.tmp / ".cache" / "huggingface" / "hub"
        shutil.rmtree(self.hub, ignore_errors=True)
        self.hub.mkdir(parents=True)
        for f in self.units.iterdir():
            f.unlink()

    def snap(self, repo, rev, whole=True, part=""):
        d = self.hub / ("models--" + repo.replace("/", "--")) / "snapshots" / rev / part
        d.mkdir(parents=True)
        (d / "model.safetensors.index.json").write_text('{"weight_map": {"a": "1.safetensors", "b": "2.safetensors"}}')
        (d / "1.safetensors").write_text("w")
        if whole:
            (d / "2.safetensors").write_text("w")

    def targets(self):
        return {t["target"]: t for t in self.cp.registry_snapshot(max_age=0.0)["targets"]}

    def test_every_switch_target_is_listed_once(self):
        self.assertEqual(sorted(self.targets()), sorted(self.cp.ACTIONS["switch"]["params"]["target"]))

    def test_whole_begun_or_absent_and_whether_its_lane_is_installed(self):
        self.snap("RadixArk/Qwen3.8-27B-NVFP4", self.pins["STOCK_REV"])
        self.snap("Aleph-Alpha/Kolibri-1", self.pins["KOLIBRI_FP8_REV"], whole=False)
        (self.units / "qwen38-sglang.service").write_text("[Service]\n")
        (self.units / "qwen38-kolibri.service").write_text("[Service]\n")
        t = self.targets()
        self.assertEqual((t["stock"]["state"], t["stock"]["installed"]), ("complete", True))
        self.assertEqual((t["kolibri"]["state"], t["kolibri"]["installed"]), ("incomplete", True))
        self.assertEqual((t["kolibri-nvfp4"]["state"], t["kolibri-nvfp4"]["installed"]), ("absent", True))
        self.assertEqual((t["image"]["state"], t["image"]["installed"]), ("absent", False))
        self.assertEqual(t["flash"]["installed"], False)

    def test_the_video_lane_is_whole_for_the_partition_it_serves(self):
        rev = self.pins["VIDEO_MODEL_PIN_REV"]
        self.snap("MiniMaxAI/MiniMax-H3", rev, part="FL2VA/transformer")
        self.snap("MiniMaxAI/MiniMax-H3", rev, whole=False, part="Ref2VA/transformer")
        (self.hub / "models--MiniMaxAI--MiniMax-H3" / "snapshots" / rev / "model_index.json").write_text("{}")
        (self.units / "qwen38-video.service").write_text(
            "[Service]\nExecStart=/x/python -m sglang.multimodal_gen.runtime.launch_server "
            "--model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va\n")
        self.assertEqual(self.targets()["video"]["state"], "complete")

    def test_a_nemotron_target_is_whole_with_its_drafter_whole_too(self):
        # the two Nemotron lanes load a drafter with the checkpoint, and their Download fetches the
        # two: Bolt's read "on disk" with its drafter missing (found on the reference box, 2026-10-15)
        for target, model, draft in (
                ("nemotron", "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB", "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"),
                ("nemotron-super", "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
                 "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr")):
            pin = self.cp.TARGET_PIN[target]
            dpin = self.cp.TARGET_DRAFT_PIN[target]
            with self.subTest(target=target):
                self.setUp()
                self.assertEqual(self.targets()[target]["state"], "absent")
                self.snap(model, self.pins[pin])
                self.assertEqual(self.targets()[target]["state"], "incomplete", "its drafter is missing")
                self.snap(draft, self.pins[dpin], whole=False)
                self.assertEqual(self.targets()[target]["state"], "incomplete", "its drafter is begun")
                shutil.rmtree(self.hub / ("models--" + draft.replace("/", "--")))
                self.snap(draft, self.pins[dpin])
                self.assertEqual(self.targets()[target]["state"], "complete")
                shutil.rmtree(self.hub / ("models--" + model.replace("/", "--")))
                self.assertEqual(self.targets()[target]["state"], "incomplete", "its drafter alone")
        # a target with no drafter of its own reads its checkpoint alone, as before
        self.setUp()
        self.snap("RadixArk/Qwen3.8-27B-NVFP4", self.pins["STOCK_REV"])
        self.assertEqual(self.targets()["stock"]["state"], "complete")

    def test_the_flash_local_targets_read_their_checkpoints_their_own_way(self):
        # the lane's two targets share the checkpoint: flash-local reads it alone,
        # flash-local-dflash2 reads it with its DFlash2 drafter, whole with its drafter whole,
        # begun with either begun, as the Nemotron lanes do
        model, draft = "nvidia/Qwen3.8-Flash-Next-FP4-Local", "incoai/Qwen3.8-Flash-Next-DFlash2"
        pin, dpin = self.cp.TARGET_PIN["flash-local"], self.cp.TARGET_DRAFT_PIN["flash-local-dflash2"]
        self.assertEqual(self.cp.TARGET_PIN["flash-local-dflash2"], pin, "one checkpoint for the lane's two targets")
        self.setUp()
        self.assertEqual(self.targets()["flash-local"]["state"], "absent")
        self.assertEqual(self.targets()["flash-local-dflash2"]["state"], "absent")
        self.snap(model, self.pins[pin])
        self.assertEqual(self.targets()["flash-local"]["state"], "complete")
        self.assertEqual(self.targets()["flash-local-dflash2"]["state"], "incomplete", "its drafter is missing")
        self.snap(draft, self.pins[dpin], whole=False)
        self.assertEqual(self.targets()["flash-local-dflash2"]["state"], "incomplete", "its drafter is begun")
        shutil.rmtree(self.hub / ("models--" + draft.replace("/", "--")))
        self.snap(draft, self.pins[dpin])
        self.assertEqual(self.targets()["flash-local-dflash2"]["state"], "complete")
        shutil.rmtree(self.hub / ("models--" + model.replace("/", "--")))
        self.assertEqual(self.targets()["flash-local"]["state"], "absent")
        self.assertEqual(self.targets()["flash-local-dflash2"]["state"], "incomplete",
                         "its checkpoint is gone, its drafter alone")

    def test_the_action_runs_the_switch_with_fetch_and_nothing_else(self):
        code, body = self.cp.start_action("fetch", {"target": "kolibri-nvfp4"})
        self.assertEqual(code, 202, body)
        self.assertEqual(body["argv"], ["bash", str(REPO / "switch-model.sh"), "kolibri-nvfp4", "--fetch"])
        for _ in range(100):
            if self.cp.JOBS[body["job"]].status != "running":
                break
            __import__("time").sleep(0.05)
        self.assertEqual(self.cp.start_action("fetch", {"target": "nope"})[0], 400)
        self.assertIs(self.cp.ACTIONS["fetch"]["params"]["target"], self.cp.ACTIONS["switch"]["params"]["target"])


if __name__ == "__main__":
    unittest.main()
