"""The reverse proxy's secret header as a way in (dashboard/proxy.md).

nginx proxy manager does the Basic-auth at the edge and injects a header carrying a
shared secret; a request that presents it reads as signed in, on both the cockpit and
the agent relay, while the direct path keeps asking for the session cookie exactly as
before. What a browser cannot do is set a custom header from another origin (the
preflight would need a CORS answer this server never gives), so possession of the
secret is its own unforgeable proof, and mutating POSTs so admitted owe no CSRF token.

Two cockpits are loaded side by side: one with COCKPIT_MAGIC_SECRET set, one without
(the installed default), because "off by default" is a security property and only the
second one can show it.
"""
import http.client
import http.server
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
DASH = HERE.parents[1]
REPO = HERE.parents[2]

API_KEY = "test-key-not-a-real-one"
SECRET = "s3cr3t-from-the-proxy"
HEADER = "X-Alzeo-Auth"
CLOSED = "http://127.0.0.1:1"


def load_cockpit(config_dir: Path, tag: str, magic: bool, agent_port: str = "0"):
    """Import cockpit.py with its environment at a throwaway box; magic=False leaves
    the secret unset, which is the installed default. The relay thread only starts in
    main(), so a nonzero COCKPIT_AGENT_PORT here just shapes the CSP and the state."""
    env = dict(COCKPIT_DRY_RUN="1", COCKPIT_CONFIG_DIR=str(config_dir),
               COCKPIT_REPO_DIR=str(REPO), COCKPIT_PORT="0", COCKPIT_AGENT_PORT=agent_port,
               COCKPIT_AUTOHEAL="0", COCKPIT_ENGINE=CLOSED, COCKPIT_PROXY=CLOSED,
               COCKPIT_IMAGE=CLOSED, HOME=str(config_dir))
    if magic:
        env.update(COCKPIT_MAGIC_SECRET=SECRET,
                   COCKPIT_PUBLIC_URL="https://dash.example.com",
                   COCKPIT_AGENT_PUBLIC_URL="https://agent.example.com")
    else:
        env.update(COCKPIT_MAGIC_SECRET="", COCKPIT_PUBLIC_URL="",
                   COCKPIT_AGENT_PUBLIC_URL="")
    os.environ.update(env)
    sys.path.insert(0, str(DASH))
    spec = importlib.util.spec_from_file_location(f"cockpit_{tag}_under_test",
                                                  DASH / "cockpit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeOpencode(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        FakeOpencode.seen.append((self.path, dict(self.headers)))
        out = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


class CockpitBase(unittest.TestCase):
    """Boots one cockpit on an ephemeral port."""
    magic = True
    tag = "on"
    agent_port = "0"

    @classmethod
    def setUpClass(cls):
        cls.env_before = dict(os.environ)
        cls.path_before = list(sys.path)
        cls.tmp = Path(tempfile.mkdtemp(prefix=f"cockpit-magic-{cls.tag}-"))
        (cls.tmp / "api-key").write_text(API_KEY + "\n")
        cls.cp = load_cockpit(cls.tmp, cls.tag, cls.magic, cls.agent_port)
        cls.srv = cls.cp.Server(("127.0.0.1", 0), cls.cp.Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]
        cls.addClassCleanup(cls.srv.shutdown)
        cls.addClassCleanup(cls.srv.server_close)
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        # module-level: the loaded cockpit shares os.environ/sys.path with the process
        def restore():
            for name in set(os.environ) - set(cls.env_before):
                del os.environ[name]
            os.environ.clear()
            os.environ.update(cls.env_before)
            sys.path[:] = cls.path_before
        cls.addClassCleanup(restore)

    def req(self, method, path, body=None, headers=None, cookie=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = dict(headers or {})
        if cookie:
            h["Cookie"] = cookie
        payload = None
        if body is not None:
            payload = json.dumps(body).encode()
            h.setdefault("Content-Type", "application/json")
        try:
            c.request(method, path, payload, h)
            r = c.getresponse()
            return r.status, dict(r.getheaders()), r.read()
        finally:
            c.close()

    def login(self):
        st, hdrs, _ = self.req("POST", "/api/login", {"key": API_KEY})
        self.assertEqual(st, 200)
        return hdrs.get("Set-Cookie", "").split(";")[0]


class MagicOn(CockpitBase):
    magic, tag = True, "on"
    agent_port = "30091"      # the CSP names the relay only when a relay is configured

    # ---- the gate ----------------------------------------------------------
    def test_state_opens_with_the_secret_header(self):
        st, _, data = self.req("GET", "/api/state", headers={HEADER: SECRET})
        self.assertEqual(st, 200)
        self.assertIn("config", json.loads(data))

    def test_state_refuses_a_wrong_secret(self):
        st, _, data = self.req("GET", "/api/state", headers={HEADER: SECRET + "x"})
        self.assertEqual(st, 401)
        self.assertEqual(json.loads(data)["error"], "auth")

    def test_state_refuses_the_secret_under_another_header(self):
        st, _, _ = self.req("GET", "/api/state", headers={"X-Other": SECRET})
        self.assertEqual(st, 401)

    def test_index_is_served_not_the_login_with_the_secret(self):
        st, _, data = self.req("GET", "/", headers={HEADER: SECRET})
        self.assertEqual(st, 200)
        self.assertIn(b'id="spine"', data)        # the real page, not login.html

    def test_login_page_still_answers_without_it(self):
        st, _, data = self.req("GET", "/")
        self.assertEqual(st, 200)
        self.assertNotIn(b'id="spine"', data)     # the direct path still logs in by key

    def test_empty_header_value_is_not_the_secret(self):
        st, _, _ = self.req("GET", "/api/state", headers={HEADER: ""})
        self.assertEqual(st, 401)

    # ---- mutating POSTs ----------------------------------------------------
    def test_post_with_the_secret_owes_no_csrf_token(self):
        # unknown action proves the request passed auth AND the CSRF gate
        st, _, data = self.req("POST", "/api/action", {"name": "nope"},
                               headers={HEADER: SECRET})
        self.assertEqual(st, 404)
        self.assertEqual(json.loads(data)["error"], "unknown action")

    def test_post_with_a_cookie_still_owes_the_csrf_token(self):
        cookie = self.login()
        st, _, data = self.req("POST", "/api/action", {"name": "nope"}, cookie=cookie)
        self.assertEqual(st, 403)
        self.assertEqual(json.loads(data)["error"], "csrf")

    def test_post_without_either_is_refused_before_the_body_matters(self):
        st, _, _ = self.req("POST", "/api/action", {"name": "nope"})
        self.assertEqual(st, 401)

    def test_cookie_session_still_works_alongside_the_secret(self):
        cookie = self.login()
        st, _, data = self.req("POST", "/api/csrf", {}, cookie=cookie)
        self.assertEqual(st, 200)
        tok = json.loads(data)["token"]
        st, _, data = self.req("POST", "/api/action", {"name": "nope", "csrf": tok},
                               cookie=cookie)
        self.assertEqual(st, 404)                 # reached the action registry

    # ---- CSP and the state the page frames the relay from -------------------
    def test_frame_src_allows_the_public_relay(self):
        st, hdrs, _ = self.req("GET", "/api/state", headers={HEADER: SECRET})
        self.assertEqual(st, 200)
        self.assertIn("https://agent.example.com", hdrs.get("Content-Security-Policy", ""))

    def test_state_carries_the_public_url_for_the_agent_tab(self):
        # the page frames F.agent.relay.public_url (static/js/agent.js): the value the
        # unit passes as COCKPIT_AGENT_PUBLIC_URL has to reach the AGENT dict
        self.assertEqual(self.cp.AGENT["public_url"], "https://agent.example.com")
        st, _, data = self.req("GET", "/api/state", headers={HEADER: SECRET})
        self.assertEqual(json.loads(data)["config"]["data"]["agent_port"], 30091)


class MagicOff(CockpitBase):
    """The installed default: no secret configured, the header is worth nothing."""
    magic, tag = False, "off"

    def test_the_secret_is_worth_nothing_when_unset(self):
        for value in (SECRET, "", "anything"):
            st, _, _ = self.req("GET", "/api/state", headers={HEADER: value})
            self.assertEqual(st, 401)

    def test_no_public_urls_in_the_csp(self):
        st, hdrs, _ = self.req("GET", "/api/state", cookie="cockpit=" + self.cp.make_token("sess"))
        self.assertEqual(st, 200)
        self.assertNotIn("agent.example.com", hdrs.get("Content-Security-Policy", ""))


class ProxyPortBase(CockpitBase):
    """The dedicated listener (COCKPIT_PROXY_PORT): the same page, no cookie path,
    no login page, dead without the secret."""
    tag = "pp"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.psrv = cls.cp.Server(("127.0.0.1", 0), cls.cp.ProxyHandler)
        threading.Thread(target=cls.psrv.serve_forever, daemon=True).start()
        cls.pport = cls.psrv.server_address[1]
        cls.addClassCleanup(cls.psrv.shutdown)
        cls.addClassCleanup(cls.psrv.server_close)

    def preq(self, method, path, **kw):
        c = http.client.HTTPConnection("127.0.0.1", self.pport, timeout=10)
        h = dict(kw.get("headers") or {})
        if kw.get("cookie"):
            h["Cookie"] = kw["cookie"]
        payload = None
        if kw.get("body") is not None:
            payload = json.dumps(kw["body"]).encode()
            h.setdefault("Content-Type", "application/json")
        try:
            c.request(method, path, payload, h)
            r = c.getresponse()
            return r.status, dict(r.getheaders()), r.read()
        finally:
            c.close()


class ProxyPortOn(ProxyPortBase):
    magic, tag = True, "pp-on"

    def test_health_stays_public_for_the_proxy(self):
        st, _, data = self.preq("GET", "/api/health")
        self.assertEqual(st, 200)
        self.assertTrue(json.loads(data)["ok"])

    def test_everything_else_is_401_without_the_header(self):
        for path in ("/", "/login", "/api/state", "/static/js/agent.js", "/api/stream"):
            st, _, data = self.preq("GET", path)
            self.assertEqual(st, 401, path)
            self.assertEqual(json.loads(data)["error"], "auth", path)

    def test_the_main_ports_cookie_is_worth_nothing_here(self):
        cookie = self.login()                        # minted on the main listener
        st, _, _ = self.preq("GET", "/api/state", cookie=cookie)
        self.assertEqual(st, 401)
        st, _, _ = self.req("GET", "/api/state", cookie=cookie)
        self.assertEqual(st, 200)                    # and still works there

    def test_the_login_route_is_dead_here(self):
        # even the API key cannot sign in on the proxy port: the edge holds credentials
        st, _, _ = self.preq("POST", "/api/login", body={"key": API_KEY})
        self.assertEqual(st, 401)

    def test_the_header_opens_the_page_the_state_and_the_actions(self):
        st, _, data = self.preq("GET", "/", headers={HEADER: SECRET})
        self.assertEqual(st, 200)
        self.assertIn(b'id="spine"', data)
        st, _, data = self.preq("GET", "/api/state", headers={HEADER: SECRET})
        self.assertEqual(st, 200)
        self.assertIn("config", json.loads(data))
        st, _, data = self.preq("POST", "/api/action", headers={HEADER: SECRET},
                                body={"name": "nope"})
        self.assertEqual(st, 404)                    # past auth and past csrf
        self.assertEqual(json.loads(data)["error"], "unknown action")

    def test_a_wrong_secret_is_worth_a_401(self):
        st, _, _ = self.preq("GET", "/api/state", headers={HEADER: SECRET[:-1]})
        self.assertEqual(st, 401)

    def test_the_sse_stream_runs_on_it_too(self):
        c = http.client.HTTPConnection("127.0.0.1", self.pport, timeout=10)
        try:
            c.request("GET", "/api/stream", headers={HEADER: SECRET})
            r = c.getresponse()
            self.assertEqual(r.status, 200)
            self.assertIn("text/event-stream", r.getheader("Content-Type", ""))
            self.assertEqual(r.getheader("X-Accel-Buffering"), "no")
        finally:
            c.close()


class ProxyPortOff(ProxyPortBase):
    """A proxy port enabled without the secret: dead, never open."""
    magic, tag = False, "pp-off"

    def test_answers_401_to_everything_including_the_right_header(self):
        st, _, _ = self.preq("GET", "/api/state")
        self.assertEqual(st, 401)
        st, _, _ = self.preq("GET", "/api/state", headers={HEADER: SECRET})
        self.assertEqual(st, 401)
        st, _, _ = self.preq("POST", "/api/login", body={"key": API_KEY},
                             headers={HEADER: SECRET})
        self.assertEqual(st, 401)                    # off: no secret, no way in
        st, _, data = self.preq("GET", "/api/health")
        self.assertEqual(st, 200)                    # health stays


class HeaderHygiene(unittest.TestCase):
    """The secret must never reach opencode, plain or through the WebSocket tunnel."""

    def test_forward_drops_the_named_header_case_insensitively(self):
        import agent_relay as ar
        items = [("Cookie", "cockpit=x"), (HEADER, SECRET), ("x-alzeo-auth", SECRET),
                 ("Accept", "*/*")]
        out = dict(ar.forward_request_headers(items, "up:1", "Basic x", drop=(HEADER,)))
        self.assertNotIn(HEADER, out)
        self.assertNotIn("x-alzeo-auth", out)
        self.assertEqual(out.get("Accept"), "*/*")
        self.assertEqual(out.get("Authorization"), "Basic x")

    def test_forward_without_a_drop_forwards_it(self):
        # pins that the drop is the config's doing, not a blanket filter someone
        # could lean on: no magic configured, nothing extra leaves
        import agent_relay as ar
        out = dict(ar.forward_request_headers([("Accept", "*/*")], "up:1", "Basic x"))
        self.assertEqual(out.get("Accept"), "*/*")

    def test_frame_policy_names_the_public_cockpit(self):
        import agent_relay as ar
        base = ar.frame_policy("agent.example.com", 30090)
        self.assertEqual(base, "frame-ancestors 'self' http://agent.example.com:30090")
        wide = ar.frame_policy("agent.example.com", 30090, "https://dash.example.com")
        self.assertEqual(wide, base + " https://dash.example.com")

    def test_invalid_header_name_falls_back(self):
        # the cockpit validates COCKPIT_MAGIC_HEADER at import: a name with a space or
        # colon in it could never be sent as one header
        tmp = Path(tempfile.mkdtemp(prefix="cockpit-magic-name-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "api-key").write_text(API_KEY + "\n")
        before = dict(os.environ)
        try:
            cp = load_cockpit(tmp, "name", magic=True)
            os.environ["COCKPIT_MAGIC_HEADER"] = "Bad Name:"
            cp2 = load_cockpit(tmp, "name2", magic=True)
            self.assertEqual(cp.MAGIC_HEADER, HEADER)      # the default, kept
            self.assertEqual(cp2.MAGIC_HEADER, HEADER)      # the invalid one replaced
        finally:
            for name in set(os.environ) - set(before):
                del os.environ[name]
            os.environ.clear()
            os.environ.update(before)


class TheCockpitsRelay(unittest.TestCase):
    """The relay the cockpit itself builds (agent_config) opens opencode to the
    session cookie, to the proxy's secret header, and to nothing else."""

    @classmethod
    def setUpClass(cls):
        cls.env_before = dict(os.environ)
        cls.path_before = list(sys.path)
        cls.tmp = Path(tempfile.mkdtemp(prefix="relay-magic-"))
        (cls.tmp / "api-key").write_text(API_KEY + "\n")
        (cls.tmp / "opencode-web.env").write_text(
            "OPENCODE_SERVER_USERNAME=opencode\nOPENCODE_SERVER_PASSWORD=pw-test\n")
        cls.upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOpencode)
        cls.upstream.daemon_threads = True
        threading.Thread(target=cls.upstream.serve_forever, daemon=True).start()
        os.environ.update(COCKPIT_AGENT_UPSTREAM=f"http://127.0.0.1:{cls.upstream.server_address[1]}")
        cls.cp = load_cockpit(cls.tmp, "relay", magic=True)
        cls.relay = cls.cp.ar.serve(cls.cp.agent_config(), "127.0.0.1", 0)
        threading.Thread(target=cls.relay.serve_forever, daemon=True).start()
        cls.addClassCleanup(cls.relay.server_close)
        cls.addClassCleanup(cls.relay.shutdown)
        cls.addClassCleanup(cls.upstream.server_close)
        cls.addClassCleanup(cls.upstream.shutdown)
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)

        def restore():
            for name in set(os.environ) - set(cls.env_before):
                del os.environ[name]
            os.environ.clear()
            os.environ.update(cls.env_before)
            sys.path[:] = cls.path_before
        cls.addClassCleanup(restore)

    def setUp(self):
        FakeOpencode.seen.clear()

    def get(self, headers=None, path="/session"):
        c = http.client.HTTPConnection("127.0.0.1", self.relay.server_address[1], timeout=10)
        try:
            c.request("GET", path, headers=dict(headers or {}))
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def test_relay_refuses_an_anonymous_request(self):
        st, _ = self.get()
        self.assertEqual(st, 401)

    def test_relay_refuses_a_wrong_secret(self):
        st, _ = self.get({HEADER: SECRET + "x"})
        self.assertEqual(st, 401)

    def test_relay_admits_the_secret_and_hides_it_upstream(self):
        st, body = self.get({HEADER: SECRET})
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(body), {"ok": True})
        self.assertEqual(len(FakeOpencode.seen), 1)
        path, hdrs = FakeOpencode.seen[0]
        self.assertEqual(path, "/session")
        self.assertTrue(hdrs.get("Authorization", "").startswith("Basic "))
        self.assertIsNone(hdrs.get(HEADER))          # the secret never reaches opencode

    def test_session_cookie_still_opens_it(self):
        tok = self.cp.make_token("sess")
        st, _ = self.get({"Cookie": f"cockpit={tok}"})
        self.assertEqual(st, 200)

    def test_relay_pages_point_at_the_public_cockpit(self):
        st, body = self.get({"Accept": "text/html"}, path="/session")
        self.assertEqual(st, 401)
        self.assertIn(b"https://dash.example.com/", body)   # cfg.cockpit_public, not host:port

    def test_frame_ancestors_admits_the_public_cockpit(self):
        c = http.client.HTTPConnection("127.0.0.1", self.relay.server_address[1], timeout=10)
        try:
            c.request("GET", "/session", headers={HEADER: SECRET,
                                                  "Host": "agent.example.com"})
            r = c.getresponse()
            r.read()
            csp = r.getheader("Content-Security-Policy", "")
        finally:
            c.close()
        self.assertIn("https://dash.example.com", csp)


if __name__ == "__main__":
    unittest.main()
