#!/usr/bin/env python3
"""The flash-local-dflash2 target of the Flash FP4-Local lane: the lane's second target.

flash-local serves the checkpoint's own MTP head, drafting 4 tokens as the card gives it, and
renders the unit the release before this change wrote, byte for byte. flash-local-dflash2 serves
the same checkpoint under the same unit, port and served model name, drafted by incoai's DFlash2
drafter at its pinned revision: the drafter fetched with the checkpoint into the same cache,
named in the unit's speculative config in the dotted flags the MTP target already uses (no JSON,
no quotes inside the unit's single-quoted ExecStart), FLASH_LOCAL_DRAFT_TOKENS draft tokens (3 by
default, 1 to 7, the length the installed unit names when the run gives none), the drafter's
loader as one more dotted flag only when the installer's FLASH_LOCAL_DRAFT_LOAD_FORMAT line names
one, the capture plugin this repo carries in flash-local-dflash2/ copied into the config directory
and mounted read-only at /opt/df2-plugin on the engine's PYTHONPATH, and the build in the target's
own caches, flash-local-cache-dflash2/, the kernel choices of one drafter's run not being the
other's. The target is the unit's own record, as the Kolibri lane's two checkpoints are: a plain
run keeps the target the installed unit names, FLASH_LOCAL_VARIANT says the other, and a value it
is not is refused by its name before anything is fetched or rendered. A checkout without the
plugin directory (a partial copy) stops the dflash2 install, saying so. These run the installer as
written, in a throwaway home; the stop runs it out of a throwaway repo copy with no plugin."""
import getpass
import grp
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
import test_flash_local_stage as fls  # noqa: E402  (its fake cache and the unit's ExecStart, as bash runs it)
import test_engine_inputs as ein  # noqa: E402  (its fakes: systemctl, docker, the images)

INSTALLER = (REPO / "install-flash-local.sh").read_text()
PINS = {k: v.strip('"') for k, v in (line.split("=", 1) for line in INSTALLER.splitlines()
                                     if line.startswith("FLASH_LOCAL_") and "=" in line)}
DRAFT_ID = PINS["FLASH_LOCAL_DRAFT_REPO"].replace("/", "--")
DRAFT_REV = PINS["FLASH_LOCAL_DRAFT_REV"]
# what the release before this change served: the template as it stood, rendered with the values
# its installer gave it for flash-local (the test's home and cache)
PRE = (REPO / "tests" / "fixtures" / "flash-local-template-pre-dflash2.service").read_text()
# the files the stage of the installer takes from the repo it runs out of
REPO_FILES = ("install-flash-local.sh", "lanes.sh", "boxmem.py", "vllm-lane.sh", "ple-table.py",
              "patch-template.py", "seccomp-io-uring.json", "flash-local-files.sh",
              "qwen38-flash-local.service.template")
PLUGIN_DIR = REPO / "flash-local-dflash2"
# the dflash2 target's speculative config, as the unit's own line gives it, for N draft tokens
DFLASH2_FLAGS = ("--speculative-config.method dflash "
                 f"--speculative-config.model {PINS['FLASH_LOCAL_DRAFT_REPO']} "
                 f"--speculative-config.revision {DRAFT_REV} "
                 "--speculative-config.num_speculative_tokens {n}")
# the drafter's own loader, one more dotted key on the same line of the unit: the checkpoint
# keeps b12x, the drafter's plain safetensors goes through vLLM's own loader
DRAFT_LOAD_FLAG = " --speculative-config.draft_load_config.load_format auto"


def pre_dflash2_render(home, hf):
    """The flash-local unit as the release before this change wrote it: the fixture template, its
    placeholders filled with this box's values, its empty continuation lines dropped, as its
    sed did."""
    index = f"{hf}/hub/models--{PINS['FLASH_LOCAL_REPO'].replace('/', '--')}/snapshots/{PINS['FLASH_LOCAL_REV']}/model.safetensors.index.json"
    values = {"__CHECKPOINT_INDEX__": index, "__TABLE_SOURCES__": "", "__USER__": getpass.getuser(),
              "__GROUP__": grp.getgrgid(os.getgid()).gr_name, "__HOME__": str(home),
              "__IMAGE__": PINS["FLASH_LOCAL_IMAGE"], "__MODEL__": PINS["FLASH_LOCAL_REPO"],
              "__REV__": PINS["FLASH_LOCAL_REV"], "__GMU__": PINS["FLASH_LOCAL_GMU"],
              "__SEQS__": PINS["FLASH_LOCAL_SEQS"], "__HF_CACHE__": str(hf),
              "__ENGINE_BIND__": "127.0.0.1", "__PORT__": "30000", "__TABLE_MOUNTS__": ""}
    text = PRE
    for ph, val in values.items():
        text = text.replace(ph, val)
    lines = [ln for ln in text.splitlines() if not re.fullmatch(r"\s*\\", ln)]
    return "\n".join(lines) + "\n"


class Box:
    """A home and a cache, and the installer to run: the repo's own, or a throwaway copy of its
    files without the plugin directory (without_plugin), the partial checkout the stop is for."""

    def __init__(self, without_plugin=False, installed_unit=None):
        base = fls.Box()   # the fake home and the checkpoint's cache, as the stage test lays them out
        self.t = base.t
        self.home = base.home
        self.units = base.units
        self.hf = base.hf
        self.draft_snap = self.hf / "hub" / f"models--{DRAFT_ID}" / "snapshots" / DRAFT_REV
        self.draft_snap.mkdir(parents=True)
        (self.draft_snap / "config.json").write_text('{"architectures": ["DFlash2DraftModel"]}\n')
        (self.draft_snap / "model.safetensors").write_bytes(b"draft-weights" * 64)
        self.out = self.t / "stage.service"
        if installed_unit is not None:
            (self.units / "qwen38-flash-local.service").write_text(installed_unit)
        if without_plugin:
            self.repo = self.t / "repo"
            self.repo.mkdir()
            for f in REPO_FILES:
                (self.repo / f).write_text((REPO / f).read_text())
        else:
            self.repo = REPO

    def stage(self, **env):
        e = {"PATH": "/usr/bin:/bin", "HOME": str(self.home), "HF_CACHE": str(self.hf),
             "LANES_UNIT_DIR": str(self.units), "QWEN38_MEMORY_CLASS": "128", **env}
        r = subprocess.run(["bash", str(self.repo / "install-flash-local.sh"), "--stage", str(self.out)],
                           capture_output=True, text=True, env=e, timeout=120)
        return r.returncode, r.stdout + r.stderr

    def cfg(self):
        return self.home / ".config" / "qwen38"

    def plugin_marker(self):
        # the file the unit's check names: the entry point the engine finds the plugin by
        return self.cfg() / "flash-local-dflash2" / "flash_local_dflash2-0.0.0.dist-info" / "entry_points.txt"

    def guard(self, unit: str, tries: int = 0) -> list:
        """The check at the head of the unit's ExecStart, as bash runs it, with its tries of 3 s set.
        What it has to look like is asserted where it runs: this box is no test case."""
        cmd = fls.exec_command(unit)
        head = cmd[:cmd.index("exec /usr/bin/docker run")].strip()
        return ["env", f"FLASH_LOCAL_FILES_TRIES={tries}", "bash", "-c", head + " exit 0"]


class TheFlashLocalUnitStaysTodays(unittest.TestCase):
    """flash-local, and no variant given: the unit the release before this change wrote, byte for
    byte, and the installer's behaviour with it, what it keeps and what it makes, unchanged too."""

    def setUp(self):
        self.box = Box()

    def tearDown(self):
        shutil.rmtree(self.box.t, ignore_errors=True)

    def fresh(self):
        shutil.rmtree(self.box.t, ignore_errors=True)
        self.box = Box()

    def test_no_variant_and_mtp_render_the_pre_dflash2_unit_byte_for_byte(self):
        for variant in (None, "mtp"):
            with self.subTest(variant=variant):
                self.fresh()
                env = {} if variant is None else {"FLASH_LOCAL_VARIANT": variant}
                rc, out = self.box.stage(FLASH_LOCAL_PLE_DIR="none", **env)
                self.assertEqual(rc, 0, out)
                got = self.box.out.read_bytes()
                want = pre_dflash2_render(self.box.home, self.box.hf).encode()
                self.assertEqual(got, want,
                                 "the flash-local unit is not what the release before this change wrote")
                self.assertFalse((self.box.cfg() / "flash-local-cache-dflash2").exists())
                self.assertFalse((self.box.cfg() / "flash-local-dflash2").exists())
                for cache in ("vllm", "flashinfer", "b12x", "triton"):
                    self.assertTrue((self.box.cfg() / "flash-local-cache" / cache).is_dir(), cache)

    def test_the_flash_local_unit_keeps_its_card_flags(self):
        rc, out = self.box.stage(FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        unit = self.box.out.read_text()
        self.assertIn("--speculative-config.method mtp --speculative-config.num_speculative_tokens 4", unit)
        self.assertIn("-e VLLM_MXFP8_LM_HEAD=1", unit)
        self.assertIn("-e PYTHONPATH=/opt/qwen38-auth -e QWEN38_KEY_FILE", unit)
        self.assertNotIn("df2-plugin", unit)
        self.assertNotIn("DFlash2", unit)


class TheDflash2Target(unittest.TestCase):
    def setUp(self):
        self.box = Box()

    def tearDown(self):
        shutil.rmtree(self.box.t, ignore_errors=True)

    def fresh(self):
        shutil.rmtree(self.box.t, ignore_errors=True)
        self.box = Box()

    def test_it_names_the_drafter_in_its_speculative_config(self):
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="dflash2", FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        unit = self.box.out.read_text()
        self.assertIn("    " + DFLASH2_FLAGS.format(n=3) + DRAFT_LOAD_FLAG + " \\\n", unit)
        self.assertNotIn("--speculative-config.method mtp", unit)
        self.assertNotIn("num_speculative_tokens 4", unit)
        # the drafter goes through vLLM's own loader, the one flag the Nemotron lane's dflash
        # drafter carries too: the checkpoint keeps b12x, the drafter's plain BF16 file does
        # not want the loader built for the checkpoint's quantized shards
        self.assertIn("--speculative-config.draft_load_config.load_format auto", unit)
        # no JSON and no quote inside the unit's one single-quoted ExecStart: what reaches the
        # engine is what the line says, word for word, the dotted keys the parser folds into one
        # speculative config
        cmd = fls.exec_command(unit)
        serve = cmd[cmd.index("vllm serve "):]
        self.assertNotIn('"', serve)
        self.assertNotIn("'", serve)
        self.assertNotIn("{", serve)
        self.assertNotIn("\\", serve)
        self.assertEqual(serve.count("--speculative-config"), 5)
        r = subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        # and the MTP target, which loads no drafter at all, keeps the unit it had
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="mtp", FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        self.assertNotIn("draft_load_config", self.box.out.read_text(),
                         "the mtp unit has no drafter to load")

    def test_the_drafter_goes_through_vllms_own_loader_and_the_installer_line_names_it(self):
        # the drafter is one plain BF16 model.safetensors with no index of its own: it goes
        # through vLLM's loader, as the LiLiCorr drafter of the Nemotron lane does
        # (qwen38-nemotron.service.template:81). The b12x loader the checkpoint reads is an
        # O_DIRECT/GDS read of that checkpoint's quantized shards, keyed on its
        # model.safetensors.index.json and on an initial-load session the drafter has no part
        # in (b12x_loader.py:224, :280 of the pinned wheel). The one line of the installer is
        # the knob: naming b12x there hands the drafter to the target's loader instead.
        text = (REPO / "install-flash-local.sh").read_text()
        self.assertEqual(text.count('\nFLASH_LOCAL_DRAFT_LOAD_FORMAT="auto"\n'), 1)
        box = Box(without_plugin=True)
        try:
            shutil.copytree(PLUGIN_DIR, box.repo / "flash-local-dflash2")
            inst = box.repo / "install-flash-local.sh"
            inst.write_text(inst.read_text().replace('\nFLASH_LOCAL_DRAFT_LOAD_FORMAT="auto"\n',
                                                     '\nFLASH_LOCAL_DRAFT_LOAD_FORMAT="b12x"\n'))
            rc, out = box.stage(FLASH_LOCAL_VARIANT="dflash2", FLASH_LOCAL_PLE_DIR="none")
            self.assertEqual(rc, 0, out)
            self.assertIn("    " + DFLASH2_FLAGS.format(n=3)
                          + " --speculative-config.draft_load_config.load_format b12x \\\n",
                          box.out.read_text())
        finally:
            shutil.rmtree(box.t, ignore_errors=True)

    def test_the_drafts_length_is_the_tokens_and_holds_its_range(self):
        for n in ("1", "7"):
            with self.subTest(n=n):
                self.fresh()
                rc, out = self.box.stage(FLASH_LOCAL_VARIANT="dflash2", FLASH_LOCAL_DRAFT_TOKENS=n,
                                         FLASH_LOCAL_PLE_DIR="none")
                self.assertEqual(rc, 0, out)
                self.assertIn(DFLASH2_FLAGS.format(n=n) + " ", self.box.out.read_text())
        for bad in ("0", "8", "3x"):
            with self.subTest(bad=bad):
                self.fresh()
                rc, out = self.box.stage(FLASH_LOCAL_VARIANT="dflash2", FLASH_LOCAL_DRAFT_TOKENS=bad,
                                         FLASH_LOCAL_PLE_DIR="none")
                self.assertNotEqual(rc, 0, out)
                self.assertIn(f"got: {bad}", out)

    def test_the_length_the_unit_names_holds_for_the_runs_after(self):
        # as the target itself is the unit's own record: a run with no token of its own reads the
        # draft's length off the installed unit, and gives 3 where the unit names none
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="dflash2", FLASH_LOCAL_DRAFT_TOKENS="5",
                                 FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        df2_unit = self.box.out.read_text()
        self.box = Box(installed_unit=df2_unit)
        rc, out = self.box.stage(FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        self.assertIn(DFLASH2_FLAGS.format(n=5) + " ", self.box.out.read_text())
        self.fresh()
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="dflash2", FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        self.assertIn(DFLASH2_FLAGS.format(n=3) + " ", self.box.out.read_text())
        # the mtp unit's 4 is the card's, never read as the dflash2 draft's length
        self.fresh()
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="mtp", FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        self.box = Box(installed_unit=self.box.out.read_text())
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="dflash2", FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        self.assertIn(DFLASH2_FLAGS.format(n=3) + " ", self.box.out.read_text())

    def test_it_mounts_the_plugin_and_puts_it_on_the_pythonpath(self):
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="dflash2", FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        unit = self.box.out.read_text()
        self.assertIn(f"--mount type=bind,source={self.box.home}/.config/qwen38/flash-local-dflash2,"
                      "target=/opt/df2-plugin,readonly \\", unit)
        self.assertIn("-e PYTHONPATH=/opt/qwen38-auth:/opt/df2-plugin", unit)
        # the copy, whole folder, what the mount reads at the start: the repo's plugin, file
        # for file, and nothing else
        self.assertTrue(self.box.plugin_marker().is_file())
        copied = sorted(p.relative_to(self.box.cfg() / "flash-local-dflash2")
                        for p in (self.box.cfg() / "flash-local-dflash2").rglob("*") if p.is_file())
        want = sorted(p.relative_to(PLUGIN_DIR) for p in PLUGIN_DIR.rglob("*")
                      if p.is_file() and "__pycache__" not in p.parts)
        self.assertEqual(copied, want)
        for rel in want:
            self.assertEqual((self.box.cfg() / "flash-local-dflash2" / rel).read_bytes(),
                             (PLUGIN_DIR / rel).read_bytes(), str(rel))
        self.assertIn("flash_local_dflash2 = flash_local_dflash2.plugin:register",
                      self.box.plugin_marker().read_text())

    def test_the_drafter_group_gets_a_layer_compact_kv_layout(self):
        """The drafter's five-layer group is only viewable under a layout that keeps a manager
        block one dense page; the mtp target names no layout at all and vLLM resolves it."""
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="dflash2", FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        lines = [ln for ln in self.box.out.read_text().splitlines() if "VLLM_KV_CACHE_LAYOUT" in ln]
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].strip().startswith("-e VLLM_KV_CACHE_LAYOUT=LBNHC"), lines)
        self.assertTrue(lines[0].endswith("\\"), lines)
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="mtp", FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        self.assertNotIn("VLLM_KV_CACHE_LAYOUT", self.box.out.read_text())

    def test_it_builds_in_its_own_caches(self):
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="dflash2", FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        unit = self.box.out.read_text()
        for what, where in (("vllm", "/root/.cache/vllm"), ("flashinfer", "/root/.cache/flashinfer"),
                            ("b12x", "/root/.cache/b12x"), ("triton", "/root/.triton")):
            self.assertIn(f"-v {self.box.home}/.config/qwen38/flash-local-cache-dflash2/{what}:{where} \\", unit)
        for cache in ("vllm", "flashinfer", "b12x", "triton"):
            self.assertTrue((self.box.cfg() / "flash-local-cache-dflash2" / cache).is_dir(), cache)
        self.assertFalse((self.box.cfg() / "flash-local-cache" / "vllm").exists(),
                         "the flash-local target's caches are not this target's")

    def test_the_image_is_the_lanes_pin_and_the_head_line_is_the_mtps(self):
        # the drafter's experiment settled its two open values without the target (df2-super, its
        # phase 0): the dflash2 target runs the lane's own image pin, so that the MTP comparison
        # runs on the image it measured, and the checkpoint's output head, which it stores in
        # BF16, serves without the card's MXFP8 head: the mtp target's line, and no head line at
        # all under dflash2
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="dflash2", FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        unit = self.box.out.read_text()
        self.assertIn(f"  {PINS['FLASH_LOCAL_IMAGE']} \\\n", unit)
        self.assertNotIn("VLLM_MXFP8_LM_HEAD", unit)
        self.fresh()
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="mtp", FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        mtp = self.box.out.read_text()
        self.assertIn(f"  {PINS['FLASH_LOCAL_IMAGE']} \\\n", mtp)
        self.assertIn("-e VLLM_MXFP8_LM_HEAD=1 -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13", mtp)

    def test_its_check_guards_the_drafter_and_the_plugin(self):
        # the unit's check of its files asks, besides the checkpoint and the table's copies, for
        # the drafter's snapshot, its config and its one weights file, and for the plugin's file
        # its mount names: missing or empty, the start stays failed, with the line the cockpit
        # reads
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="dflash2", FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        unit = self.box.out.read_text()
        cmd = fls.exec_command(unit)
        check_call = cmd[cmd.index('bash "$check" '):cmd.index("|| exit $?")]
        for what in (str(self.box.draft_snap / "config.json"),
                     str(self.box.draft_snap / "model.safetensors"),
                     str(self.box.plugin_marker())):
            self.assertIn(what + " ", check_call, what)
        head = cmd[:cmd.index("exec /usr/bin/docker run")].strip()
        script = f"{self.box.home}/.config/qwen38/flash-local-files.sh"
        self.assertTrue(head.startswith(f"check={script}; ") and 'bash "$check" ' in head, head)
        self.assertTrue(head.endswith("|| exit $?;"), head)
        self.assertTrue(os.access(script, os.X_OK), "the stage put no check where the unit runs it")
        sys.path.insert(0, str(REPO / "dashboard"))
        import lifecycle

        def start():
            return subprocess.run(self.box.guard(unit), capture_output=True, text=True, timeout=30)
        self.assertEqual(start().returncode, 0)
        cases = ((self.box.draft_snap / "config.json", "its DFlash2 drafter's snapshot"),
                 (self.box.draft_snap / "model.safetensors", "its DFlash2 drafter's snapshot"),
                 (self.box.plugin_marker(), "its capture plugin"))
        for path, word in cases:
            with self.subTest(missing=path.name):
                gone = path.with_name(path.name + ".gone")
                path.rename(gone)
                r = start()
                self.assertEqual(r.returncode, 78, r.stderr)
                self.assertIn(str(path), r.stderr)
                self.assertIn(word, r.stderr)
                said = lifecycle.parse_preflight(r.stderr.splitlines())
                self.assertEqual((said or {}).get("verdict"), "lacks its files", r.stderr)
                self.assertIn(str(path), said["reason"])
                gone.rename(path)
        self.assertEqual(start().returncode, 0)


class TheVariantIsTheUnitsOwnRecord(unittest.TestCase):
    """As the Kolibri lane's two checkpoints: a run with no variant of its own keeps the target
    the installed unit names, and a value the lane does not serve is refused before anything is
    fetched or rendered."""

    def setUp(self):
        self.box = Box()

    def tearDown(self):
        shutil.rmtree(self.box.t, ignore_errors=True)

    def fresh(self):
        shutil.rmtree(self.box.t, ignore_errors=True)
        self.box = Box()

    def test_a_dflash2_unit_keeps_its_target_on_a_run_with_no_variant(self):
        # the target is the unit's own record: a run with no variant of its own renders the target
        # the installed unit names, the dflash2 one where the drafter is in its speculative config
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="dflash2", FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        unit = self.box.out.read_text()
        self.assertIn("df2-plugin", unit)
        self.box = Box(installed_unit=unit)
        rc, out = self.box.stage(FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        again = self.box.out.read_text()
        self.assertIn("df2-plugin", again)
        self.assertIn(DFLASH2_FLAGS.format(n=3) + " ", again,
                      "a run with no variant of its own re-rendered another target")
        self.assertNotIn("--speculative-config.method mtp", again)

    def test_a_flash_local_unit_keeps_its_target_too(self):
        rc, out = self.box.stage(FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        mtp = self.box.out.read_text()
        self.assertNotIn("df2-plugin", mtp)
        self.box = Box(installed_unit=mtp)
        rc, out = self.box.stage(FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        again = self.box.out.read_text()
        self.assertIn("--speculative-config.method mtp --speculative-config.num_speculative_tokens 4", again)
        self.assertNotIn("df2-plugin", again)

    def test_a_variant_the_lane_serves_not_is_refused(self):
        for bad in ("DFlash2", "mtp2", "dflash"):
            with self.subTest(bad=bad):
                self.fresh()
                rc, out = self.box.stage(FLASH_LOCAL_VARIANT=bad, FLASH_LOCAL_PLE_DIR="none")
                self.assertNotEqual(rc, 0, out)
                self.assertIn(f"got: {bad}", out)
                self.assertIn("mtp", out)
                self.assertIn("dflash2", out)
                self.assertFalse(self.box.out.exists() and self.box.out.read_text().strip(),
                                 "a unit rendered with a variant the lane does not serve")


class ThePluginNotInTheCheckout(unittest.TestCase):
    """A checkout without flash-local-dflash2/ (a partial copy of the repo) stops the dflash2
    target's install, saying so, rather than stage a unit whose mount has no source (docker
    would make a root-owned folder in its place, and the engine would start without its
    plugin)."""

    def test_a_dflash2_stage_without_the_plugin_stops_with_its_line(self):
        box = Box(without_plugin=True)
        try:
            rc, out = box.stage(FLASH_LOCAL_VARIANT="dflash2", FLASH_LOCAL_PLE_DIR="none")
            self.assertNotEqual(rc, 0, out)
            self.assertIn("flash-local-dflash2/", out)
            self.assertIn("the lane's flash-local-dflash2 target cannot be installed", out)
            self.assertFalse(box.out.exists() and box.out.read_text().strip(),
                             "a dflash2 unit staged with no plugin to mount")
            self.assertFalse((box.cfg() / "flash-local-dflash2").exists())
            # the lane's own target goes on, as today
            rc, out = box.stage(FLASH_LOCAL_PLE_DIR="none")
            self.assertEqual(rc, 0, out)
        finally:
            shutil.rmtree(box.t, ignore_errors=True)


class TheUninstallTakesTheDflash2Half(unittest.TestCase):
    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp(prefix="fl-dflash2-uninstall-"))
        self.home = self.d / "home"
        self.cfg = self.home / ".config" / "qwen38"
        self.cfg.mkdir(parents=True)
        (self.cfg / "vllm-auth").mkdir()
        for f in ("chat-template-flash-local.jinja", "seccomp-io-uring.json", "flash-local-files.sh"):
            (self.cfg / f).write_text("x\n")
        (self.cfg / "flash-local-cache").mkdir()
        (self.cfg / "flash-local-cache-dflash2").mkdir()
        (self.cfg / "flash-local-dflash2" / "flash_local_dflash2-0.0.0.dist-info").mkdir(parents=True)
        (self.cfg / "flash-local-dflash2" / "flash_local_dflash2-0.0.0.dist-info" / "entry_points.txt").write_text("x\n")
        self.sd = self.d / "systemd"
        self.sd.mkdir()
        self.bin = self.d / "bin"
        self.bin.mkdir()
        self.log = self.d / "calls.log"
        for tool in ("sudo", "docker", "systemctl"):
            (self.bin / tool).write_text(f'#!/bin/bash\necho "{tool} $*" >> "{self.log}"\n'
                                         + ('[ "$1" = is-enabled ] && exit 1\n' if tool == "systemctl" else "")
                                         + "exit 0\n")
            (self.bin / tool).chmod(0o755)
        self.unit = self.sd / "qwen38-flash-local.service"
        # the two checkpoints' cache directories: what the uninstall says is kept, like the
        # lane's own checkpoint
        self.hf = self.d / "hf"
        (self.hf / "hub" / f"models--{PINS['FLASH_LOCAL_REPO'].replace('/', '--')}").mkdir(parents=True)
        (self.hf / "hub" / f"models--{DRAFT_ID}").mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def run_block(self):
        self.unit.write_text("ExecStart=/bin/bash -c 'exec /usr/bin/docker run x'\n")
        text = (REPO / "install-flash-local.sh").read_text()
        start = text.index('if [ "$ACTION" = uninstall ]; then')
        block = text[start:text.index("\n  exit 0\nfi\n", start) + len("\n  exit 0\nfi\n")]
        pins = [ln for ln in text.splitlines()
                if ln.startswith(("FLASH_LOCAL_REPO=", "FLASH_LOCAL_IMAGE=", "FLASH_LOCAL_DRAFT_REPO=",
                                  "VLANE="))]
        script = "\n".join(["set -euo pipefail", "step(){ :; }", f'LANES_UNIT_DIR="{self.sd}"',
                            f'. "{REPO}/lanes.sh"', f'. "{REPO}/vllm-lane.sh"', *pins,
                            f'HERE="{REPO}"', "ACTION=uninstall", "UNIT=qwen38-flash-local.service",
                            f'INSTALLED="{self.unit}"', "CONTAINER=qwen38-flash-local",
                            f'CONFIG_DIR="{self.cfg}"', f'HF_CACHE="{self.d}/hf"',
                            block])
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                           env=dict(os.environ, PATH=f"{self.bin}:/usr/bin:/bin", HOME=str(self.home)))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout

    def test_the_dflash2_caches_and_the_plugin_copy_go_and_both_checkpoints_are_said_to_stay(self):
        out = self.run_block()
        self.assertFalse((self.cfg / "flash-local-cache-dflash2").exists())
        self.assertFalse((self.cfg / "flash-local-dflash2").exists())
        self.assertIn("checkpoint kept", out)
        self.assertIn(f"models--{PINS['FLASH_LOCAL_REPO'].replace('/', '--')}", out)
        self.assertIn(f"models--{DRAFT_ID}", out,
                      "the drafter, said to stay like the lane's own checkpoint")
        for gone in ("chat-template-flash-local.jinja", "seccomp-io-uring.json", "flash-local-files.sh"):
            self.assertFalse((self.cfg / gone).exists(), gone)


class TheEngineInputsCountTheChange(unittest.TestCase):
    """What a change of the target, or of the plugin's files, should restart: the target is the
    unit itself, whose content is the first input of every lane (a dflash2 unit and a flash-local
    unit differ in their speculative config, their caches, their mount, their image and their
    head, and the fingerprint over the unit's content changes with them); the plugin's files are
    counted the way the lane's other inputs of the config directory that its unit does not name
    are: as the extra of engine-inputs.py, the vLLM lanes' key middleware being the one of
    them (vllm-lane.sh's LANE_FILES, the cockpit's VLLM_LANE_FILES), and the drafter's
    checkpoint the way the lane's checkpoints are: not counted by name, its snapshot's files
    guarded by the unit's check of its files instead, as the checkpoint's are."""

    def setUp(self):
        self.box = Box()

    def tearDown(self):
        shutil.rmtree(self.box.t, ignore_errors=True)

    def unit_box(self, unit_text):
        box = ein.Box()
        box.unit.write_text(unit_text)
        images = box.images.read_text()
        digest = PINS["FLASH_LOCAL_IMAGE"]
        images += f"\n{digest}=sha256:img1\ncontainer:qwen38-flash-local=sha256:img1\n"
        box.images.write_text(images)
        return box

    def test_a_change_of_the_target_moves_the_fingerprint_of_the_unit_it_renders(self):
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="mtp", FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        mtp = self.box.out.read_text()
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="dflash2", FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        df2 = self.box.out.read_text()
        self.assertNotEqual(mtp, df2)
        box = self.unit_box(mtp)
        fp_mtp = box.run("fingerprint")
        box.unit.write_text(df2)
        fp_df2 = box.run("fingerprint")
        self.assertNotEqual(fp_mtp, fp_df2, "the target's change is the unit's content changing")

    def test_a_change_of_the_plugin_is_one_of_the_extras_that_counts(self):
        rc, out = self.box.stage(FLASH_LOCAL_VARIANT="dflash2", FLASH_LOCAL_PLE_DIR="none")
        self.assertEqual(rc, 0, out)
        box = self.unit_box(self.box.out.read_text())
        t = int(time.time()) - 100
        for f in (box.unit, box.cfg / "api-key"):
            os.utime(f, (t, t))
        for f in box.cfg.rglob("*"):
            if f.is_file():
                os.utime(f, (t, t))
        box.started = t
        marker = box.cfg / "flash-local-dflash2" / "flash_local_dflash2-0.0.0.dist-info" / "entry_points.txt"
        # written now, after the start: an input the engine reads at its start, the cockpit's
        # extra names it, and running() says what it says of a change of it
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("new capture\n")
        import importlib.util
        spec = importlib.util.spec_from_file_location("engine_inputs_count", REPO / "engine-inputs.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        before = dict(os.environ)
        os.environ.update(PATH=f"{box.bin}:/usr/bin:/bin", FAKE_IMAGES=str(box.images),
                          FAKE_ACTIVE="active", FAKE_STARTED=str(box.started))
        try:
            self.assertEqual(mod.running(str(box.unit), str(box.cfg), str(box.hf), [],
                                         extra=[str(marker)]),
                             f"no: {marker} changed after {box.unit.name} started")
            # without it named, the plugin is none of the unit's inputs (the directory it mounts
            # is not a file, and no line of the unit names the marker)
            self.assertEqual(mod.running(str(box.unit), str(box.cfg), str(box.hf), []), "yes")
        finally:
            os.environ.clear()
            os.environ.update(before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
