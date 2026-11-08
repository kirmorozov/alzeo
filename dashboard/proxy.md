# Opening the cockpit through nginx proxy manager

Two ways to reach the cockpit, both live at once:

```
browser ──(Basic-auth)──▶ nginx proxy manager ──(injects the secret header)──▶ cockpit :30090 / relay :30091
browser ──(API key login, as always)────────────────────────────────────────▶ cockpit :30090 / relay :30091
```

The cockpit and the relay each accept a request when **either** holds: it carries a
valid session cookie (the direct path, the API key at the login prompt), or it carries
a header the reverse proxy injects after its Basic-auth accepted the browser. The
secret never appears in the browser, and a page on another origin cannot set a custom
header of its own — the preflight would need a CORS answer this server never gives —
so possession of the header proves the proxy let you through. That is why a mutating
POST so admitted also owes no CSRF token: the token exists because cookies ride along
on any request; the header cannot.

Nothing changes without a secret set: with `COCKPIT_MAGIC_SECRET` empty (the default,
and what a plain install leaves), the header is worth nothing and everything behaves
exactly as before.

## One proxy is enough for the status view

The browser only ever talks to the cockpit's own port. The engine (`:30000`) and the
keepalive proxy (`:30001`) are read **server-side**, by the cockpit process itself
(`/metrics`, `/health`, `/proxy/*`), and shipped to the page over the cockpit's port.
So one NPM host — pointed at the proxy listener below, `30095` — shows the whole box:
what serves, the pool, traffic, logs, the works. A second host, `:30091`, is only for
the Agent view (opencode's own web UI and its terminal, which frame directly into the
browser and need their own host).

## Install the secret, and the dedicated proxy listener

```bash
openssl rand -hex 32    # the secret, once
```

Then, on the box (idempotent, keeps everything else the unit says):

```bash
DASH_MAGIC_SECRET=<that-random> \
DASH_PROXY_PORT=30095 \
DASH_PUBLIC_URL=https://spark.home.morozov.cloud \
DASH_AGENT_PUBLIC_URL=https://spark-agent.example.com \
dashboard/install-dashboard.sh
```

- `DASH_MAGIC_SECRET` goes into `~/.config/qwen38/dashboard.env`, mode 0600 — never
  into the unit file, which is world-readable. A re-run without it keeps the installed
  one; with it, the file is replaced and the cockpit restarted.
- `DASH_PROXY_PORT` opens the **dedicated proxy-facing listener**: the same page and
  action surface, but it opens only to requests carrying the secret header. No cookie
  path, no login page, and `/api/health` stays public for the proxy's checks. With no
  secret set the port is dead (401 to everything), never open. **Point nginx at this
  one** (`:30095`) and leave the main port (`:30090`) to the LAN.
- The two public URLs are only for the content-security policy (the cockpit's page may
  frame the proxied relay, and the relay accepts a frame from the proxied cockpit).
  Skip `DASH_AGENT_PUBLIC_URL` if you are not doing the Agent view. The page frames
  the public relay only when it is itself loaded over https; over plain http it frames
  `host:30091` as always, so LAN access keeps working untouched.

The env file may also be edited by hand and the unit restarted:

```bash
printf 'COCKPIT_MAGIC_SECRET=%s\n' <that-random> > ~/.config/qwen38/dashboard.env
chmod 600 ~/.config/qwen38/dashboard.env
sudo systemctl restart qwen38-dashboard
```

## nginx proxy manager

**Proxy Host 1 — the cockpit** (`spark.home.morozov.cloud` → `192.168.1.187:30095`
— the dedicated proxy listener, `DASH_PROXY_PORT` above — scheme HTTP, TLS
certificate on, Force HTTPS on). Under the **Access** tab, enable Basic-Auth and add
your user. Then under **Custom** → **nginx Configuration**:

```nginx
location / {
    proxy_set_header X-Alzeo-Auth "<the same long secret>";
    include /config/nginx/proxy.conf;
    proxy_pass http://192.168.1.187:30095;
}
```

`proxy_set_header` **replaces** any value the client sent under that name, so a
browser cannot hand itself the header — only nginx's value reaches the cockpit. The
**Websockets Support** toggle is not needed here (the live view is a server-sent
event stream, and the cockpit already tells nginx not to buffer it with
`X-Accel-Buffering: no`); leaving it on is harmless. If you would rather not open
the extra port, pointing NPM at the main `:30090` works too — that port accepts the
header alongside its cookie path — but the dedicated listener keeps the surfaces
apart: a session cookie is worth nothing on the proxy listener, and its 401s never
touch the login flow of the LAN port. (The secret itself opens both ports — on the
LAN, anyone holding it could log in with the API key anyway.)

**Proxy Host 2 — the Agent relay** (optional: `spark-agent.example.com` →
`192.168.1.187:30091`). Same recipe, plus the **Websockets Support** toggle (the
terminal panel is one WebSocket):

```nginx
location / {
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection $http_connection;
    proxy_set_header X-Alzeo-Auth "<the same long secret>";
    include /config/nginx/proxy.conf;
    proxy_pass http://192.168.1.187:30091;
}
```

The relay never forwards the header to opencode, along with the session cookie and the
client's own Authorization (see `REQUEST_DROP` in `agent_relay.py`).

## What each path looks like

| | LAN / tailnet | through the proxy |
|---|---|---|
| cockpit `:30090` | API key at the login prompt, as always | (or use the listener below; it accepts the header too) |
| proxy listener `:30095` | answers 401 to everything without the header | Basic-auth prompt from nginx, then straight into the page |
| relay `:30091` | needs the cockpit session (same host, cookie rides over) | the same Basic-auth, then the header carries it |
| opencode `:4096` | never reachable: only the relay talks to it, on loopback | unchanged |

The cockpit still binds a LAN/tailnet address only (`COCKPIT_BIND`); nginx reaches it
on that address. If the box should become reachable **only** through the proxy, bind
it to loopback (`DASH_BIND=127.0.0.1`) and point nginx at `127.0.0.1` if nginx runs on
the same box — with one exception: the Agent relay binds its own address
(`COCKPIT_AGENT_BIND`), so give that host the header too, or it stays closed.

## Knobs

| env (unit) / install variable | meaning | default |
|---|---|---|
| `COCKPIT_MAGIC_SECRET` / `DASH_MAGIC_SECRET` | the header's value; empty = feature off | off |
| `COCKPIT_MAGIC_HEADER` / `DASH_MAGIC_HEADER` | which header the proxy must inject | `X-Alzeo-Auth` |
| `COCKPIT_PROXY_PORT` / `DASH_PROXY_PORT` | dedicated proxy-facing listener, header-only; 0 = off | off |
| `COCKPIT_PUBLIC_URL` / `DASH_PUBLIC_URL` | the cockpit's public URL, for the relay's frame policy and its sign-in page | unset |
| `COCKPIT_AGENT_PUBLIC_URL` / `DASH_AGENT_PUBLIC_URL` | the relay's public URL, for the cockpit's CSP and the Agent tab's frame | unset |

Pick a header name nginx will not mangle (underscores in header names are dropped by
nginx unless `underscores_in_headers on`; dashes are safe). If you change
`COCKPIT_MAGIC_HEADER`, change `proxy_set_header` to match — a request under any other
name is an anonymous request.

## Belt notes

- The secret compares in constant time, and a missing header is compared rather than
  short-circuited, so the answer's timing says nothing about the secret.
- Anyone on the LAN who can reach `:30090` and knows the secret can walk in without
  the API key — they could already do that with the API key itself. What the secret
  adds is the *proxy* path, and the proxy path is gated by nginx's Basic-auth.
- Rotate by re-running the install command with a new `DASH_MAGIC_SECRET`: the file
  lands, the unit sees the file is newer than the running process and restarts. Update
  both nginx `proxy_set_header` lines, or the proxy path logs you out with 401s.
- If you expose the relay host without the cockpit host, remember the relay accepts
  the header or a cockpit session cookie: with the secret as its only key, treat the
  Basic-auth on that host as the lock it is.
