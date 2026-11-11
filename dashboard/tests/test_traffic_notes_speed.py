#!/usr/bin/env python3
"""Three changes of the Traffic view and the notes strip (2026-11-10):

1. A note that informs can be hidden, and stays hidden across a reload for as long as its words
   do not change (the hash of its words kept in localStorage).
2. The speed tiles: prefill, decode and aggregate tokens per second, smoothed over a sliding
   window (engine_metrics.Rate) and shown like the running-connections counter.
3. On a narrow screen the list of connections is the last block of the Traffic view.
"""
import importlib.util
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pagejs  # noqa: E402
from test_page_behaviour import run, css_rules  # noqa: E402

DASH = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("engine_metrics_rate", DASH / "engine_metrics.py")
em = importlib.util.module_from_spec(spec)
spec.loader.exec_module(em)

CLONE = ("{installed: 'v1.46.0', installed_commit: 'aaaaaaa', clone_commit: '%s', commits_count: 1, "
         "behind_clone: true, behind_clone_why: 'commit'}")
FEED = r"""
const upd = c => feed({config: CONFIG, units: UNITS, lifecycle: life({}), update: %s});
"""


class HiddenNotes(unittest.TestCase):
    def go(self, body, **kw):
        return run(self, FEED % "c" + body, **kw)

    def test_hide_survives_a_reload_and_a_changed_message_shows_again(self):
        r = self.go(r"""
        upd(%s); const before = txt('banners');
        const hide = document.querySelector('#banners .banner.orange .nhide'); hide.click();
        const hidden = txt('banners'), stored = localStorage.getItem('cockpit.hidden-notes');
        hiddenMem = null; $('banners').dataset.sig = '';               // a reload: memory gone, storage kept
        upd(%s); const reloaded = txt('banners');
        upd(%s); const changed = txt('banners');
        report({before, hidden, stored, reloaded, changed});
        """ % (CLONE % "bbbbbbb", CLONE % "bbbbbbb", CLONE % "ccccccc"))
        self.assertIn("The clone moved on", r["before"])
        self.assertNotIn("The clone moved on", r["hidden"])
        self.assertIn("1 hidden", r["hidden"])
        self.assertEqual(len(json.loads(r["stored"])), 1)
        self.assertNotIn("The clone moved on", r["reloaded"])
        self.assertIn("The clone moved on", r["changed"])        # a new commit: a new hash

    def test_the_count_shows_them_again_and_clears_the_storage(self):
        r = self.go(r"""
        upd(%s); document.querySelector('#banners .banner.orange .nhide').click();
        const row = document.querySelector('#banners .banner-hidden .nshow'); row.click();
        report({text: txt('banners'), stored: localStorage.getItem('cockpit.hidden-notes')});
        """ % (CLONE % "bbbbbbb"))
        self.assertIn("The clone moved on", r["text"])
        self.assertNotIn("hidden", r["text"].replace("Hide", ""))
        self.assertIsNone(r["stored"])

    def test_an_error_and_the_running_installer_have_no_hide(self):
        r = self.go(r"""
        feed({config: CONFIG, units: UNITS, lifecycle: life({}),
              update: {install_running: {to: 'v9.9.9', started: Date.now() / 1000 - 60}}});
        const run = document.querySelectorAll('#banners .nhide').length;
        feed({config: CONFIG, units: UNITS, lifecycle: life({'qwen38-sglang.service': eng('wedged')})});
        report({run, err: document.querySelectorAll('#banners .banner.err .nhide').length,
                errs: document.querySelectorAll('#banners .banner.err').length});
        """)
        self.assertEqual(r["run"], 0)
        self.assertEqual(r["errs"], 1)
        self.assertEqual(r["err"], 0)

    def test_blocked_storage_still_works_for_the_page_view(self):
        r = self.go(r"""
        upd(%s); document.querySelector('#banners .banner.orange .nhide').click();
        report({text: txt('banners')});
        """ % (CLONE % "bbbbbbb"), opts={"noStorage": True})
        self.assertNotIn("The clone moved on", r["text"])

    def test_a_countdown_is_no_part_of_the_hash(self):
        r = self.go(r"""
        report({a: msgHash('Title', 'answers again in 2 min'), b: msgHash('Title', 'answers again in 35 s'),
                c: msgHash('Title', 'another sentence in 2 min')});
        """)
        self.assertEqual(r["a"], r["b"])
        self.assertNotEqual(r["a"], r["c"])


class SpeedRate(unittest.TestCase):
    def feed(self, rate, rows, key="u"):
        return [rate.add(t, key, p, g) for t, p, g in rows]

    def test_none_until_two_reads_then_the_mean_over_the_window(self):
        r = em.Rate(10.0)
        out = self.feed(r, [(0, 0, 0), (1, 0, 80), (2, 0, 160)])
        self.assertIsNone(out[0])
        self.assertEqual(out[2], {"prefill": 0.0, "decode": 80.0, "total": 80.0})

    def test_idle_reads_zero(self):
        r = em.Rate(10.0)
        out = self.feed(r, [(t, 500, 900) for t in range(5)])
        self.assertEqual(out[-1], {"prefill": 0.0, "decode": 0.0, "total": 0.0})

    def test_a_bursty_prefill_is_smoothed_over_the_window_not_chunk_by_chunk(self):
        r = em.Rate(10.0)
        # 4096-token chunks every other second: the raw one-second rate flips 4096 / 0
        rows, p = [], 0
        for t in range(0, 21):
            p += 4096 if t % 2 == 0 else 0
            rows.append((t, p, 0))
        out = self.feed(r, rows)
        tail = [o["prefill"] for o in out[12:]]
        self.assertTrue(all(1800 <= v <= 2300 for v in tail), tail)    # about 2048, never 0 or 4096

    def test_the_window_forgets_old_reads(self):
        r = em.Rate(10.0)
        out = self.feed(r, [(0, 0, 0), (1, 0, 1000), (30, 0, 1000), (31, 0, 1000)])
        self.assertEqual(out[-1]["decode"], 0.0)

    def test_counters_going_back_or_another_engine_start_again(self):
        r = em.Rate(10.0)
        self.feed(r, [(0, 0, 0), (1, 100, 100)])
        self.assertIsNone(r.add(2, "u", 5, 5))                     # restarted: no negative rate
        self.assertIsNone(r.add(3, "other", 9, 9))
        self.assertEqual(r.add(4, "other", 9, 29)["decode"], 20.0)

    def test_the_tiles_say_a_dash_with_no_engine_and_a_zero_when_idle(self):
        body = r"""
        feed({engine_fast: {load: [{num_reqs: 0}], speed: %s}});
        const idle = [txt('tr-pre'), txt('tr-dec'), txt('tr-tps')];
        feed({engine_fast: {load: null, speed: null}});
        const none = [txt('tr-pre'), txt('tr-dec'), txt('tr-tps')];
        feed({engine_fast: {load: [{num_reqs: 1}], speed: {prefill: 1234.5, decode: 73.2, total: 1307.7}}});
        report({idle, none, live: [txt('tr-pre'), txt('tr-dec'), txt('tr-tps')]});
        """ % '{prefill: 0, decode: 0, total: 0}'
        r = run(self, body)
        self.assertEqual(r["idle"], ["0", "0", "0"])
        self.assertEqual(r["none"], ["-", "-", "-"])
        self.assertEqual(r["live"], ["1,235", "73", "1,308"])


class MobileOrder(unittest.TestCase):
    def test_the_list_of_connections_is_last_on_a_narrow_screen_only(self):
        rules = [x for x in css_rules() if "#tr-feed" in x[0]]
        self.assertEqual(len(rules), 1)
        sel, media, decl = rules[0]
        self.assertIn("order:2", decl.replace(" ", ""))
        self.assertTrue(media and "max-width:980px" in media[0].replace(" ", ""), media)

    def test_the_markup_keeps_the_desktop_order(self):
        html = (pagejs.STATIC / "index.html").read_text()
        a, b, c = html.index('id="tr-speed"'), html.index('id="tr-feed"'), html.index('data-src="metrics"')
        self.assertLess(a, b)
        self.assertLess(b, c)          # the list stays the first column of the wide grid


if __name__ == "__main__":
    unittest.main()
