#!/usr/bin/env python3
"""The Experts view: the counter's header on the stream, the counts by the view's own poll,
and the page's reading of both.

The counts are 98 KB a second for 48 layers x 512 experts: the stream carries the header
alone (the experts collector, every 3 s), and the view fetches /api/experts whole once a
second while it is open. The route serves the file as the plugin wrote it, behind the
session like every /api route; the page parses it, keeps the previous frame and draws the
difference. The rail entry appears for a Qwen Next model or a counter that writes, and for
nothing else.
"""
import base64
import http.client
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile
import threading
import time
import unittest
from array import array
from pathlib import Path

HERE = Path(__file__).resolve()
DASH = HERE.parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "flash-local-experts"))
import pagejs  # noqa: E402
from flash_local_experts import wire  # noqa: E402
from test_page_behaviour import HELPERS  # noqa: E402

API_KEY = "test-key-not-a-real-one"
CLOSED = "http://127.0.0.1:9"
ENV_BEFORE = {}


def setUpModule():
    ENV_BEFORE.update(os.environ)


def tearDownModule():
    for name in set(os.environ) - set(ENV_BEFORE):
        del os.environ[name]
    os.environ.update(ENV_BEFORE)


def load_cockpit(config_dir: Path, experts_file: Path):
    os.environ.update(
        COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(config_dir), COCKPIT_REPO_DIR=str(REPO),
        COCKPIT_PORT="0", COCKPIT_AGENT_PORT="0", COCKPIT_AUTOHEAL="0",
        COCKPIT_ENGINE=CLOSED, COCKPIT_PROXY=CLOSED, COCKPIT_IMAGE=CLOSED, HOME=str(config_dir),
        COCKPIT_EXPERTS_FILE=str(experts_file),
    )
    sys.path.insert(0, str(DASH))
    spec = importlib.util.spec_from_file_location("cockpit_experts_under_test", DASH / "cockpit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


META = {"served_model_name": "qwen3.8-flash-next-fp4", "model_type": "qwen4_exp_text", "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "hidden_size": 2560, "num_experts": 8, "num_experts_per_tok": 2, "hc_count": 4, "ple_layer_ids": [2],
        "vocab_size": 248320, "spec_method": "mtp", "num_speculative_tokens": 4,
        "rows": ["model.layers.0.mlp.experts", "model.layers.1.mlp.experts", "model.layers.2.mlp.experts",
                 "model.layers.3.mlp.experts", "mtp.layers.0.mlp.experts"]}


def frame(counts, seq=1, wall=1000.0, pid=4242, meta=META):
    data = array("i", counts).tobytes()
    return wire.pack(meta, data, len(counts) // 8, 8, 2, seq, wall, pid)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cockpit-experts-"))
        (cls.tmp / "api-key").write_text(API_KEY + "\n")
        cls.file = cls.tmp / "qwen38-experts.bin"
        cls.ck = load_cockpit(cls.tmp, cls.file)
        cls.ck.run = lambda argv, timeout=5.0, merge_err=False, errors="strict": ""
        cls.srv = cls.ck.Server(("127.0.0.1", 0), cls.ck.Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.ck.LOGIN_FAILS.clear()
        if self.file.exists():
            self.file.unlink()

    def req(self, method, path, body=None, cookie=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = {"Content-Type": "application/json"} if body is not None else {}
        if cookie:
            h["Cookie"] = cookie
        c.request(method, path, body=json.dumps(body) if body is not None else None, headers=h)
        r = c.getresponse()
        data = r.read()
        hdrs = {k: v for k, v in r.getheaders()}
        c.close()
        return r.status, hdrs, data

    def login(self):
        st, hdrs, _ = self.req("POST", "/api/login", {"key": API_KEY})
        self.assertEqual(st, 200)
        return hdrs.get("Set-Cookie", "").split(";")[0]


class TheCollector(Base):
    def test_no_file_is_no_counter(self):
        d = self.ck.collect_experts()
        self.assertEqual((d["present"], d["live"]), (False, False))

    def test_a_counter_that_writes_is_live_with_the_models_shape(self):
        self.file.write_bytes(frame([0] * 40, seq=7))
        d = self.ck.collect_experts()
        self.assertEqual((d["present"], d["live"], d["rows"], d["experts"], d["top_k"], d["model"], d["model_type"], d["seq"], d["spec_method"]),
                         (True, True, 5, 8, 2, "qwen3.8-flash-next-fp4", "qwen4_exp_text", 7, "mtp"))

    def test_a_file_the_writer_left_behind_is_present_but_not_live(self):
        self.file.write_bytes(frame([0] * 40))
        old = time.time() - 60
        os.utime(self.file, (old, old))
        d = self.ck.collect_experts()
        self.assertEqual((d["present"], d["live"]), (True, False))
        self.assertGreater(d["age"], 50)

    def test_without_a_counter_the_shape_is_the_lanes_checkpoint_and_the_units_drafter(self):
        rev = "b" * 40
        snap = self.tmp / "hf/hub/models--nvidia--Qwen-Test/snapshots" / rev
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": {
            k: v for k, v in META.items() if k not in ("rows", "spec_method", "num_speculative_tokens", "served_model_name")}}))
        unit = self.tmp / "qwen38-flash-local.service"
        unit.write_text(f"ExecStart=docker run -v {self.tmp}/hf:/root/.cache/huggingface x vllm serve nvidia/Qwen-Test --revision {rev} "
                        "--speculative-config.method dflash --speculative-config.model incoai/Draft --speculative-config.num_speculative_tokens 4\n")
        before = self.ck.FLASH_LOCAL_UNIT_PATH
        self.ck.FLASH_LOCAL_UNIT_PATH = unit
        try:
            self.ck._SHAPE_CACHE.clear()
            d = self.ck.collect_experts()
            sh = d["shape"]
            self.assertEqual((d["present"], sh["num_hidden_layers"], sh["layer_types"], sh["num_experts"], sh["num_experts_per_tok"], sh["ple_layer_ids"], sh["hc_count"]),
                             (False, 4, META["layer_types"], 8, 2, [2], 4))
            self.assertEqual((sh["spec_method"], sh["num_speculative_tokens"], sh["draft_model"]), ("dflash", 4, "incoai/Draft"))
            self.assertIs(self.ck.collect_experts()["shape"], sh, "read once per config file and unit")
            unit.write_text(unit.read_text().replace("method dflash", "method mtp"))
            self.assertEqual(self.ck.collect_experts()["shape"]["spec_method"], "mtp", "a new unit text is read again")
            self.file.write_bytes(frame([0] * 40))
            self.assertNotIn("shape", self.ck.collect_experts(), "a counter that writes carries the shape itself")
        finally:
            self.ck.FLASH_LOCAL_UNIT_PATH = before
            self.ck._SHAPE_CACHE.clear()

    def test_without_the_lanes_unit_there_is_no_shape_and_no_error(self):
        before = self.ck.FLASH_LOCAL_UNIT_PATH
        self.ck.FLASH_LOCAL_UNIT_PATH = self.tmp / "no-such-unit"
        try:
            d = self.ck.collect_experts()
        finally:
            self.ck.FLASH_LOCAL_UNIT_PATH = before
        self.assertEqual((d["present"], d["shape"]), (False, None))

    def test_a_foreign_file_is_no_counter_and_no_error(self):
        self.file.write_bytes(b"\xff" * 300)
        d = self.ck.collect_experts()
        self.assertEqual((d["present"], d.get("unreadable"), "error" in d), (False, True, False))

    def test_it_is_sampled_every_three_seconds_and_the_page_knows_the_period(self):
        tier = next(p for p, c in self.ck.TIERS if "experts" in c)
        self.assertEqual(tier, 3.0)
        self.assertEqual(self.ck.PERIODS.get("experts"), 3.0)


class TheRoute(Base):
    def test_it_is_behind_the_session(self):
        self.file.write_bytes(frame([1] * 40))
        st, _, _ = self.req("GET", "/api/experts")
        self.assertEqual(st, 401)

    def test_no_file_is_a_404_the_page_can_read(self):
        st, hdrs, body = self.req("GET", "/api/experts", cookie=self.login())
        self.assertEqual(st, 404)
        self.assertIn("no routing counter", json.loads(body)["error"])

    def test_the_file_is_served_whole_as_written_with_its_age(self):
        data = frame(list(range(40)), seq=3)
        self.file.write_bytes(data)
        st, hdrs, body = self.req("GET", "/api/experts", cookie=self.login())
        self.assertEqual(st, 200)
        self.assertEqual(body, data)
        self.assertEqual(hdrs.get("Content-Type"), "application/octet-stream")
        self.assertEqual(hdrs.get("Cache-Control"), "no-store")
        self.assertLess(float(hdrs.get("X-Experts-Age")), 5.0)
        self.assertIn("default-src 'self'", hdrs.get("Content-Security-Policy", ""))

    def test_a_file_cut_short_is_not_served(self):
        self.file.write_bytes(frame([1] * 40)[:-3])
        st, _, _ = self.req("GET", "/api/experts", cookie=self.login())
        self.assertEqual(st, 404)


class ThePage(unittest.TestCase):
    """The page's own reading, run in node against the real scripts."""

    def run_page(self, body, **kw):
        return pagejs.run(self, HELPERS + r"""
        const bytes = s => { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u.buffer; };
        """ + body, **kw)

    def test_the_rail_entry_appears_for_a_qwen_next_model_or_a_counter_and_for_nothing_else(self):
        r = self.run_page(r"""
        const seen = {};
        feed({config: CONFIG, units: UNITS, engine_info: {info: {served_model_name: 'qwen3.8-27b'}}, experts: {present: false, live: false}});
        seen.other = {hidden: $('nav-experts').hidden, cap: txt('ex-cap'), note: txt('ex-empty')};
        feed({experts: {present: true, live: true, age: 0.4, rows: 49, experts: 512, top_k: 10, model: 'qwen3.8-flash-next-fp4', model_type: 'qwen4_exp_text'}});
        seen.live = {hidden: $('nav-experts').hidden, cap: txt('ex-cap'), note: txt('ex-empty')};
        feed({experts: {present: true, live: false, age: 120, rows: 49, experts: 512, top_k: 10, model: 'qwen3.8-flash-next-fp4'}});
        seen.stale = {hidden: $('nav-experts').hidden, cap: txt('ex-cap')};
        feed({engine_info: {info: {served_model_name: 'qwen3.8-flash-next-fp4'}}, experts: {present: false, live: false}});
        seen.qwenNoCounter = {hidden: $('nav-experts').hidden, cap: txt('ex-cap'), note: txt('ex-empty')};
        report(seen);
        """)
        self.assertEqual((r["other"]["hidden"], r["other"]["cap"]), (True, "not this model"))
        self.assertIn("not a Qwen Next model", r["other"]["note"])
        self.assertEqual((r["live"]["hidden"], r["live"]["cap"]), (False, "live"))
        self.assertIn("49 layers x 512 experts, top 10", r["live"]["note"])
        self.assertEqual(r["stale"]["hidden"], False)
        self.assertTrue(r["stale"]["cap"].startswith("stale"), r["stale"])
        self.assertEqual((r["qwenNoCounter"]["hidden"], r["qwenNoCounter"]["cap"]), (False, "no counter"))
        self.assertIn("mounts one", r["qwenNoCounter"]["note"])

    def test_it_parses_a_frame_lays_the_rows_in_the_stack_and_draws_the_difference(self):
        first = base64.b64encode(frame([0] * 40, seq=1, wall=1000.0)).decode()
        counts = [0] * 40
        counts[0 * 8 + 1] = 6; counts[0 * 8 + 5] = 2      # layer 0: expert 1 six, expert 5 two
        counts[3 * 8 + 7] = 4                               # layer 3 (QSA): expert 7 four
        counts[4 * 8 + 0] = 10                              # the MTP head's row
        second = base64.b64encode(frame(counts, seq=2, wall=1001.0)).decode()
        r = self.run_page(r"""
        const a = exParse(bytes(%r)), b = exParse(bytes(%r));
        const layout = exLayout(b);
        EX.prev = a; EX.cur = b; EX.layout = layout; EX.mode = 'live';
        const v = exValues(), s = exStats(v, layout);
        const bad = exParse(bytes(%r).slice(0, 60));
        report({rows: b.rows, experts: b.experts, topk: b.topk, seq: b.seq, pid: b.pid, wall: b.wall, model: b.meta.served_model_name,
                main: layout.main.map(L => [L.idx, L.kind, L.ple, L.row]), extra: layout.extra.map(L => [L.kind, L.row]),
                dt: v.dt, row0: Array.from(v.v.slice(0, 8)), row3: Array.from(v.v.slice(24, 32)),
                routes: s.routes, tokens: s.tokens, tps: s.tps, used: s.used, cells: s.cells, hot: s.hot && [s.hot.layer.idx, s.hot.expert, s.hot.ratio], bad});
        """ % (first, second, second))
        self.assertEqual((r["rows"], r["experts"], r["topk"], r["seq"], r["pid"], r["wall"], r["model"]), (5, 8, 2, 2, 4242, 1001.0, "qwen3.8-flash-next-fp4"))
        self.assertEqual(r["main"], [[0, "GDN", False, 0], [1, "GDN", False, 1], [2, "GDN", True, 2], [3, "QSA", False, 3]])
        self.assertEqual(r["extra"], [["MTP head", 4]])
        self.assertEqual((r["dt"], r["row0"], r["row3"]), (1.0, [0, 6, 0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 4]))
        # 12 routes over 4 decoder layers at top 2: 1.5 tokens a second through the stack; the MTP row is not a decoder layer
        self.assertEqual((r["routes"], r["tokens"], r["tps"], r["used"], r["cells"]), (12, 1.5, 1.5, 3, 32))
        self.assertEqual(r["hot"], [3, 7, 8.0], "one expert with all of a layer's routes is 8x an even share of 8")
        self.assertIsNone(r["bad"])

    def test_before_the_counter_the_tiles_are_the_checkpoints_layers_unlit(self):
        shape = {k: v for k, v in META.items() if k != "rows"}
        r = self.run_page(r"""
        const seen = {};
        feed({config: CONFIG, units: UNITS, engine_info: {info: {served_model_name: 'qwen3.8-flash-next-fp4'}}, experts: {present: false, live: false, shape: %s}});
        seen.mtp = {panel: $('ex-panel').hidden, main: EX.layout.main.length, extra: EX.layout.extra.map(L => L.kind), tps: txt('ex-tps'), note: txt('ex-empty')};
        feed({experts: {present: false, live: false, shape: Object.assign({}, %s, {spec_method: 'dflash', draft_model: 'incoai/Draft'})}});
        seen.dflash = {panel: $('ex-panel').hidden, extra: EX.layout.extra.length};
        feed({experts: {present: false, live: false}});
        seen.noShape = {panel: $('ex-panel').hidden};
        feed({engine_info: {info: {served_model_name: 'qwen3.8-27b'}}, experts: {present: false, live: false, shape: %s}});
        seen.otherModel = {panel: $('ex-panel').hidden};
        report(seen);
        """ % (json.dumps(shape), json.dumps(shape), json.dumps(shape)))
        self.assertEqual(r["mtp"], {"panel": False, "main": 4, "extra": ["MTP head"], "tps": "…", "note": r["mtp"]["note"]})
        self.assertIn("unlit until", r["mtp"]["note"])
        self.assertEqual(r["dflash"], {"panel": False, "extra": 0}, "another drafter has no experts of its own")
        self.assertEqual(r["noShape"]["panel"], True)
        self.assertEqual(r["otherModel"]["panel"], True)

    def test_the_tiles_give_each_expert_a_fixed_place(self):
        r = self.run_page(r"""
        const g = exGeom(1000, 512, 48, 1), n = exGeom(320, 512, 48, 0);
        EX.geom = g; EX.layout = exSkeleton({num_hidden_layers: 48, num_experts: 512, spec_method: 'mtp'});
        // tile 9 (second row, second column), cell column 5 of row 2: expert 2 x 32 + 5
        const at = exCellAt(108 + 5 * 3 + 1, 26 + 72 + 14 + 2 * 3 + 1);
        report({wide: [g.cols, g.cs, g.tileW, g.tileH, g.pos(47), g.pos(48), g.H, g.gridW], narrow: [n.cols, n.cs, n.H],
                at, gap: exCellAt(108 - 1, 26 + 72 + 20), head: exCellAt(5, 26 + 5)});
        """)
        self.assertEqual(r["wide"], [8, 3, 96, 62, {"x": 756, "y": 386}, {"x": 0, "y": 482}, 588, 852])
        self.assertEqual(r["narrow"], [2, 4, 26 + 24 * 88 + 34])
        self.assertEqual(r["at"], {"i": 9, "e": 69})
        self.assertIsNone(r["gap"], "between two tiles")
        self.assertIsNone(r["head"], "a tile's head is not a cell")

    def test_the_poll_runs_while_the_view_is_open_and_stops_when_it_is_left(self):
        r = self.run_page(r"""
        feed({config: CONFIG, units: UNITS, experts: {present: true, live: true, age: 0.2, rows: 5, experts: 8, top_k: 2, model: 'x', model_type: 'qwen4_exp_text'}});
        // the cockpit answers 404 (no file): the poll ends each request and asks again a second later
        window.__fetch = async () => ({ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0)});
        const calls = () => __fetches.filter(f => f.url === '/api/experts').length;
        const before = calls();
        showView('experts'); await __advance(2100);
        const open = calls();
        showView('now'); await __advance(3000);
        report({before, open, after: calls()});
        """)
        self.assertEqual(r["before"], 0, "no poll off the view")
        self.assertGreaterEqual(r["open"], 2, "one fetch at once, then one a second")
        self.assertEqual(r["after"], r["open"], "left, the poll stops")


if __name__ == "__main__":
    unittest.main(verbosity=2)
