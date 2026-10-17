"""The run report: best lap, split comparison against your demo lap, totals and
learning-stability curves, written to runs/<name>/report/ as a self-contained
report.html (opens offline in any browser) and summary.json.

Inputs are the files a run already leaves behind: episodes.csv (one row per
attempt, written by the trainer), best_laps.jsonl (written by ShutdownHook at
every new best lap) and the track's demo laps. Works on a crashed or running
run too: `tmai report` rebuilds it at any time."""

from __future__ import annotations

import csv
import html
import json
import math
import time
from pathlib import Path

import numpy as np

from tmai.run import atomic

CURVES = (  # (episodes.csv column, title, what it tells you)
    ("reward", "Episode reward", "Total reward per attempt; should trend up."),
    ("progress", "Track progress", "Share of the reference line covered per attempt."),
    ("critic_loss", "Critic loss", "Q-network error; spikes that keep growing mean instability."),
    ("actor_loss", "Actor loss", "Policy objective; drifts as Q-values change, should not explode."),
    ("q", "Mean Q-value", "Predicted return; steady growth is normal, runaway growth is divergence."),
    ("alpha", "Entropy temperature", "SAC's exploration weight; usually falls then settles."),
    ("entropy", "Policy entropy", "How random the actions are; falls as the policy commits."),
)
WINDOW = 50          # episodes per rolling window
SESSION_GAP_S = 600  # rows further apart than this belong to different sessions


# -- loading --------------------------------------------------------------------
def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_episodes(path: Path) -> list[dict]:
    """episodes.csv rows with numbers parsed. A row cut short by a crash is skipped."""
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if None in row or any(v is None for v in row.values()):
                continue
            rows.append({k: (v if k == "end" else _num(v)) for k, v in row.items()})
    return rows


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue   # last line cut short by a crash
    return out


def demo_splits(track_dir: Path | None) -> dict | None:
    """Checkpoint times of your fastest demo lap, or None."""
    if track_dir is None or not (track_dir / "demos").is_dir():
        return None
    best = None
    for f in sorted((track_dir / "demos").glob("*.npz")):
        try:
            with np.load(f) as d:
                lap = int(d["lap_time_ms"])
                cps = [int(x) for x in d["cp_times"]] if "cp_times" in d else []
        except (OSError, KeyError, ValueError):
            continue
        if best is None or lap < best["lap_ms"]:
            best = {"lap_ms": lap, "splits": cps, "file": f.name}
    return best


# -- statistics -----------------------------------------------------------------
def rolling(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling mean and std over the last `window` finite values at each point."""
    mean = np.full(len(values), np.nan)
    std = np.full(len(values), np.nan)
    for i in range(len(values)):
        w = values[max(0, i - window + 1): i + 1]
        w = w[np.isfinite(w)]
        if len(w):
            mean[i], std[i] = w.mean(), w.std()
    return mean, std


def stability(values: np.ndarray, window: int = WINDOW) -> dict:
    """Plain-language health of one curve over the latest window."""
    finite = values[np.isfinite(values)]
    out = {"n": int(len(values)), "nan": int(len(values) - len(finite))}
    if len(finite) < 2:
        return {**out, "verdict": "not enough data"}
    last = finite[-window:]
    std = float(last.std())
    if std < 1e-9 * max(1.0, abs(float(last.mean()))):
        std = 0.0   # float noise on a constant curve
    out.update(last_mean=float(last.mean()), last_std=std)
    out["cv"] = float(last.std() / abs(last.mean())) if last.mean() else None
    x = np.arange(len(last))
    out["slope_per_100"] = float(np.polyfit(x, last, 1)[0] * 100) if len(last) > 2 and std else 0.0
    verdict = "stable"
    if out["nan"]:
        verdict = "NaN values: learning broke at some point"
    elif len(finite) >= 2 * window:
        earlier = finite[:-window]
        base = np.median(np.abs(earlier)) or 1e-9
        if np.abs(last).mean() > 5 * base:
            verdict = "much larger than earlier: possible divergence"
        elif last.std() > 3 * (np.std(earlier[-window:]) or 1e-9):
            verdict = "noisier than earlier"
    out["verdict"] = verdict
    return out


def active_hours(wall: np.ndarray) -> float:
    wall = wall[np.isfinite(wall)]
    if len(wall) < 2:
        return 0.0
    gaps = np.diff(np.sort(wall))
    return float(gaps[gaps < SESSION_GAP_S].sum() / 3600)


def split_table(ai: list[int], demo: list[int] | None) -> list[dict]:
    """Per checkpoint: cumulative and sector times for the AI's best lap and your demo."""
    rows = []
    demo = demo or []
    for i, t in enumerate(ai):
        row = {"cp": i + 1, "ai_ms": t, "ai_sector_ms": t - (ai[i - 1] if i else 0)}
        if i < len(demo):
            row["demo_ms"] = demo[i]
            row["demo_sector_ms"] = demo[i] - (demo[i - 1] if i else 0)
            row["delta_ms"] = t - demo[i]
            row["sector_delta_ms"] = row["ai_sector_ms"] - row["demo_sector_ms"]
        rows.append(row)
    return rows


def summarize(run_dir: Path, track_dir: Path | None = None, extra: dict | None = None) -> dict:
    episodes = read_episodes(run_dir / "episodes.csv")
    bests = read_jsonl(run_dir / "best_laps.jsonl")
    demo = demo_splits(track_dir)
    col = {k: np.array([r.get(k, math.nan) for r in episodes], dtype=float)
           for k in ("episode", "wall_time", "env_steps", "updates", "race_time_s",
                     *(c for c, _, _ in CURVES))}
    finished = [r for r in episodes if r.get("end") == "finished"]
    ends: dict[str, int] = {}
    for r in episodes:
        ends[r.get("end", "?")] = ends.get(r.get("end", "?"), 0) + 1

    best = bests[-1] if bests else None
    if best is None and finished:   # no best_laps.jsonl: fall back to episodes.csv (0.01 s resolution)
        r = min(finished, key=lambda r: r["race_time_s"])
        best = {"lap_ms": int(round(r["race_time_s"] * 1000)), "episode": int(r["episode"]), "splits": []}
    s = {
        "run": run_dir.name, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": len(episodes), "finished": len(finished),
        "finish_rate": len(finished) / len(episodes) if episodes else 0.0,
        "env_steps": int(np.nanmax(col["env_steps"])) if episodes else 0,
        "updates": int(np.nanmax(col["updates"])) if episodes else 0,
        "training_hours": active_hours(col["wall_time"]),
        "ends": ends, "best": best, "best_history": bests, "demo": demo,
        "first_finish": ({"episode": int(finished[0]["episode"]),
                          "lap_ms": int(round(finished[0]["race_time_s"] * 1000))} if finished else None),
        "splits": split_table(best["splits"], demo["splits"] if demo else None) if best and best.get("splits") else [],
        "stability": {c: stability(col[c]) for c, _, _ in CURVES},
    }
    if best and demo:
        s["vs_demo_ms"] = best["lap_ms"] - demo["lap_ms"]
    s.update(extra or {})
    s["_series"] = col
    s["_ends"] = [r.get("end") for r in episodes]
    return s


# -- rendering ------------------------------------------------------------------
def _fmt_ms(ms) -> str:
    if ms is None or (isinstance(ms, float) and not math.isfinite(ms)):
        return "–"
    return f"{ms / 1000:.3f} s"


def _fmt_delta(ms) -> str:
    if ms is None:
        return "–"
    return f"{'+' if ms > 0 else '−' if ms < 0 else '±'}{abs(ms) / 1000:.3f} s"


def _path(xs, ys, sx, sy) -> str:
    parts, pen = [], "M"
    for x, y in zip(xs, ys):
        if not math.isfinite(y):
            pen = "M"
            continue
        parts.append(f"{pen}{sx(x):.1f},{sy(y):.1f}")
        pen = "L"
    return " ".join(parts)


def _nice(lo: float, hi: float) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    step = 10 ** math.floor(math.log10((hi - lo) / 4))
    for m in (1, 2, 5, 10):
        if (hi - lo) / (step * m) <= 5:
            step *= m
            break
    start = math.ceil(lo / step) * step
    return [start + i * step for i in range(int((hi - start) / step) + 1)]


def svg_chart(x, layers, title: str, chart_id: str, y_label: str = "", w: int = 640, h: int = 220) -> str:
    """One-axis line chart. layers: dicts with y, kind (line|dots|band), cls, name, lo/hi for bands."""
    pad_l, pad_r, pad_t, pad_b = 56, 12, 12, 28
    x = np.asarray(x, dtype=float)
    ys = [v for L in layers for v in np.concatenate([np.asarray(L.get(k, []), float) for k in ("y", "lo", "hi")
                                                      if k in L])]
    ys = np.asarray(ys, dtype=float)
    ys = ys[np.isfinite(ys)]
    if len(x) == 0 or len(ys) == 0:
        return f'<p class="muted">No data for {html.escape(title)} yet.</p>'
    x0, x1 = float(np.nanmin(x)), float(np.nanmax(x))
    y0, y1 = float(ys.min()), float(ys.max())
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y0, y1 = y0 - 1, y1 + 1
    m = (y1 - y0) * 0.05
    y0, y1 = y0 - m, y1 + m

    def sx(v):
        return pad_l + (v - x0) / (x1 - x0) * (w - pad_l - pad_r)

    def sy(v):
        return pad_t + (1 - (v - y0) / (y1 - y0)) * (h - pad_t - pad_b)

    out = [f'<svg class="chart" id="{chart_id}" viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">']
    for t in _nice(y0, y1):
        out.append(f'<line class="grid" x1="{pad_l}" x2="{w - pad_r}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
                   f'<text class="tick" x="{pad_l - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    for t in _nice(x0, x1):
        out.append(f'<text class="tick" x="{sx(t):.1f}" y="{h - 8}" text-anchor="middle">{t:g}</text>')
    out.append(f'<text class="tick" x="4" y="{h - 8}" text-anchor="start">attempt</text>')
    out.append(f'<line class="axis" x1="{pad_l}" x2="{w - pad_r}" y1="{h - pad_b}" y2="{h - pad_b}"/>')
    for L in layers:
        cls = L.get("cls", "s1")
        if L["kind"] == "band":
            top = [(sx(a), sy(b)) for a, b in zip(x, L["hi"]) if math.isfinite(b)]
            bot = [(sx(a), sy(b)) for a, b in zip(x, L["lo"]) if math.isfinite(b)]
            if top:
                pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in top + bot[::-1])
                out.append(f'<polygon class="band {cls}" points="{pts}"/>')
        elif L["kind"] == "dots":
            for a, b in zip(x, L["y"]):
                if math.isfinite(b):
                    out.append(f'<circle class="dot {cls}" cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="2.5"/>')
        else:
            out.append(f'<path class="line {cls}" d="{_path(x, L["y"], sx, sy)}"/>')
    out.append(f'<line class="cross" x1="0" x2="0" y1="{pad_t}" y2="{h - pad_b}" visibility="hidden"/>')
    out.append("</svg>")
    # Data for the hover tooltip.
    data = {"x": [float(v) for v in x], "x0": x0, "x1": x1, "padL": pad_l, "padR": pad_r, "w": w,
            "label": y_label,
            "series": [{"name": L["name"], "y": [None if not math.isfinite(v) else round(float(v), 4)
                                                 for v in L["y"]]} for L in layers if "y" in L and L.get("name")]}
    legend = "".join(f'<span class="key"><i class="sw {L.get("cls", "s1")} {L["kind"]}"></i>{html.escape(L["name"])}</span>'
                     for L in layers if L.get("name"))
    return (f'<div class="figure"><div class="legend">{legend}</div>{"".join(out)}'
            f'<script type="application/json" data-for="{chart_id}">{json.dumps(data)}</script></div>')


CSS = """
:root{--surface:#fcfcfb;--panel:#ffffff;--text:#0b0b0b;--text2:#52514e;--muted:#7a7974;--grid:#e6e5e1;
--s1:#2a78d6;--s2:#eb6834;--good:#008300;--bad:#e34948;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--panel:#222221;--text:#f0efec;
--text2:#c3c2b7;--muted:#97968e;--grid:#353532;--s1:#3987e5;--s2:#d95926;--good:#3aa63a;--bad:#e66767;color-scheme:dark}}
:root[data-theme="dark"]{--surface:#1a1a19;--panel:#222221;--text:#f0efec;--text2:#c3c2b7;--muted:#97968e;--grid:#353532;
--s1:#3987e5;--s2:#d95926;--good:#3aa63a;--bad:#e66767;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--text);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:960px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 8px}h3{font-size:15px;margin:20px 0 2px}
.muted{color:var(--muted)}.sub{color:var(--text2);margin:0 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile{background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:12px 14px}
.tile b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.tile span{color:var(--text2);font-size:13px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}
.scroll{overflow-x:auto}th,td{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--text2);font-weight:600}
.faster{color:var(--good)}.slower{color:var(--bad)}
.figure{position:relative;background:var(--panel);border:1px solid var(--grid);border-radius:10px;padding:8px 8px 0;margin:6px 0}
.chart{width:100%;height:auto;display:block}.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--muted);stroke-width:1}
.tick{fill:var(--muted);font-size:11px}.line{fill:none;stroke-width:2;stroke-linejoin:round}
.line.s1{stroke:var(--s1)}.line.s2{stroke:var(--s2)}.dot.s1{fill:var(--s1);opacity:.35}.dot.s2{fill:var(--s2)}
.band.s1{fill:var(--s1);opacity:.14}.cross{stroke:var(--muted);stroke-dasharray:3 3}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--text2);padding:2px 4px}
.key{display:inline-flex;align-items:center;gap:6px}.sw{display:inline-block;width:14px;height:3px;border-radius:2px}
.sw.s1{background:var(--s1)}.sw.s2{background:var(--s2)}.sw.dots{width:8px;height:8px;border-radius:50%}
.sw.dots.s1{opacity:.5}.sw.band{height:10px;opacity:.3}
.tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--grid);border-radius:6px;
padding:4px 8px;font-size:12px;color:var(--text);white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,.15)}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;border:1px solid var(--grid)}
code{font-size:13px}
"""

JS = """
document.querySelectorAll('script[data-for]').forEach(function(s){
  var d=JSON.parse(s.textContent),svg=document.getElementById(s.dataset.for),fig=svg.parentNode;
  var tip=document.createElement('div');tip.className='tip';tip.hidden=true;fig.appendChild(tip);
  var cross=svg.querySelector('.cross');
  svg.addEventListener('mousemove',function(e){
    var r=svg.getBoundingClientRect(),px=(e.clientX-r.left)/r.width*d.w;
    var xv=d.x0+(px-d.padL)/(d.w-d.padL-d.padR)*(d.x1-d.x0),best=0,bd=Infinity;
    for(var i=0;i<d.x.length;i++){var dd=Math.abs(d.x[i]-xv);if(dd<bd){bd=dd;best=i}}
    var cx=d.padL+(d.x[best]-d.x0)/(d.x1-d.x0)*(d.w-d.padL-d.padR);
    cross.setAttribute('x1',cx);cross.setAttribute('x2',cx);cross.setAttribute('visibility','visible');
    var lines=['Episode '+d.x[best]];
    d.series.forEach(function(se){var v=se.y[best];if(v!==null)lines.push(se.name+': '+v)});
    tip.innerHTML=lines.join('<br>');tip.hidden=false;
    var left=e.clientX-fig.getBoundingClientRect().left+12;
    if(left>fig.clientWidth-160)left-=180;tip.style.left=left+'px';tip.style.top='28px';
  });
  svg.addEventListener('mouseleave',function(){tip.hidden=true;cross.setAttribute('visibility','hidden')});
});
"""


def render_html(s: dict) -> str:
    e = html.escape
    col = s["_series"]
    ep = col["episode"]
    best = s.get("best")
    demo = s.get("demo")
    tiles = [
        (_fmt_ms(best["lap_ms"]) if best else "no finish yet", "best lap"),
        (_fmt_delta(s["vs_demo_ms"]) if "vs_demo_ms" in s else "–", "best lap vs your demo"),
        (f'{s["env_steps"]:,}', "environment steps"),
        (f'{s["updates"]:,}', "learner updates"),
        (f'{s["training_hours"]:.1f} h', "training time"),
        (f'{s["episodes"]:,} ({s["finish_rate"] * 100:.0f}% finished)', "attempts"),
    ]
    parts = [f"<h1>Run report: {e(s['run'])}</h1>",
             f'<p class="sub">Track {e(str(s.get("track", "?")))} · generated {e(s["generated"])}'
             f'{" · stopped by " + e(str(s["stop_reason"])) if s.get("stop_reason") else ""}</p>',
             '<div class="tiles">' + "".join(f'<div class="tile"><b>{e(a)}</b><span>{e(b)}</span></div>'
                                             for a, b in tiles) + "</div>"]

    # Lap times.
    parts.append("<h2>Lap times</h2>")
    lap = np.where([end == "finished" for end in s["_ends"]], col["race_time_s"], np.nan)
    running = np.fmin.accumulate(np.where(np.isfinite(lap), lap, np.inf)) if len(lap) else lap
    running = np.where(np.isfinite(running), running, np.nan)
    parts.append(svg_chart(ep, [{"kind": "dots", "y": lap, "cls": "s1", "name": "finished lap (s)"},
                                {"kind": "line", "y": running, "cls": "s2", "name": "best so far (s)"}],
                           "Lap times", "c-lap"))
    if s.get("best_history"):
        rows = "".join(f'<tr><td>{int(b.get("episode", 0))}</td><td>{e(b.get("time", ""))}</td>'
                       f'<td>{_fmt_ms(b["lap_ms"])}</td><td>{b.get("env_steps", "–")}</td></tr>'
                       for b in s["best_history"][-15:])
        parts.append('<h3>New best laps</h3><div class="scroll"><table><tr><th>Episode</th><th>When</th>'
                     f'<th>Lap</th><th>Env steps</th></tr>{rows}</table></div>')

    # Splits.
    parts.append("<h2>Checkpoint splits</h2>")
    if s["splits"]:
        def cls(d):
            return "" if d is None else ("faster" if d < 0 else "slower" if d > 0 else "")
        rows = ""
        for r in s["splits"]:
            d, sd = r.get("delta_ms"), r.get("sector_delta_ms")
            rows += (f'<tr><td>{"Finish" if r is s["splits"][-1] else "CP " + str(r["cp"])}</td>'
                     f'<td>{_fmt_ms(r["ai_ms"])}</td><td>{_fmt_ms(r.get("demo_ms"))}</td>'
                     f'<td class="{cls(d)}">{_fmt_delta(d)}</td><td>{_fmt_ms(r["ai_sector_ms"])}</td>'
                     f'<td>{_fmt_ms(r.get("demo_sector_ms"))}</td><td class="{cls(sd)}">{_fmt_delta(sd)}</td></tr>')
        parts.append('<p class="muted">The agent\'s best lap against your fastest demo lap'
                     f'{" (" + e(demo["file"]) + ")" if demo else ""}. Negative = the agent is faster.</p>'
                     '<div class="scroll"><table><tr><th>Checkpoint</th><th>Agent</th><th>Demo</th><th>Gap</th>'
                     '<th>Agent sector</th><th>Demo sector</th><th>Sector gap</th></tr>' + rows + "</table></div>")
    else:
        parts.append('<p class="muted">No finished lap with splits yet.</p>')

    # Stability.
    parts.append("<h2>Learning stability</h2>"
                 f'<p class="muted">Rolling mean and ±1 standard deviation over {WINDOW} attempts. '
                 'The verdict compares the latest window with the rest of the run.</p>')
    rows = ""
    for c, title, _ in CURVES:
        st = s["stability"][c]
        rows += (f'<tr><td>{e(title)}</td><td>{st.get("last_mean", math.nan):.4g}</td>'
                 f'<td>{st.get("last_std", math.nan):.3g}</td><td>{st.get("slope_per_100", math.nan):+.3g}</td>'
                 f'<td>{e(st["verdict"])}</td></tr>')
    parts.append('<div class="scroll"><table><tr><th>Curve</th><th>Latest mean</th><th>Latest std</th>'
                 f'<th>Trend / 100 attempts</th><th>Verdict</th></tr>{rows}</table></div>')
    for c, title, what in CURVES:
        y = col[c]
        mean, std = rolling(y, WINDOW)
        parts.append(f"<h3>{e(title)}</h3><p class=\"muted\">{e(what)}</p>")
        parts.append(svg_chart(ep, [{"kind": "band", "lo": mean - std, "hi": mean + std, "cls": "s1"},
                                    {"kind": "dots", "y": y, "cls": "s1", "name": "per attempt"},
                                    {"kind": "line", "y": mean, "cls": "s1", "name": f"rolling mean ({WINDOW})"}],
                               title, f"c-{c}"))

    # Endings, files, replay.
    parts.append("<h2>How attempts ended</h2>")
    rows = "".join(f"<tr><td>{e(str(k))}</td><td>{v:,}</td></tr>"
                   for k, v in sorted(s["ends"].items(), key=lambda kv: -kv[1]))
    parts.append(f'<div class="scroll"><table><tr><th>End</th><th>Attempts</th></tr>{rows}</table></div>')

    parts.append("<h2>Best lap replay</h2>")
    g = s.get("replay")
    if g and g.get("found"):
        parts.append(f'<p>Saved as <code>{e(Path(g["copy"]).name)}</code> next to this report '
                     f'({_fmt_ms(g["lap_ms"])}). {e(g["how_to_watch"])}</p>')
        if g.get("note"):
            parts.append(f'<p class="muted">{e(g["note"])}</p>')
    else:
        parts.append(f'<p class="muted">{e((g or {}).get("detail", "Not looked up for this report."))}</p>')

    if s.get("shutdown"):
        rows = "".join(f'<tr><td>{e(st["name"])}</td><td>{"done" if st["ok"] else "FAILED"}</td>'
                       f'<td>{st["seconds"]:.1f} s</td><td>{e(st.get("detail", ""))}</td></tr>'
                       for st in s["shutdown"])
        parts.append("<h2>Shutdown</h2><div class=\"scroll\"><table><tr><th>Step</th><th>Result</th><th>Time</th>"
                     f"<th>Detail</th></tr>{rows}</table></div>")
    body = "\n".join(parts)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Run report {e(s['run'])}</title><style>{CSS}</style></head>"
            f"<body><main>{body}</main><script>{JS}</script></body></html>")


def build_report(run_dir: str | Path, track_dir: str | Path | None = None, extra: dict | None = None) -> Path:
    """Write report/report.html and report/summary.json; returns the HTML path."""
    run_dir = Path(run_dir)
    s = summarize(run_dir, Path(track_dir) if track_dir else None, extra)
    out = run_dir / "report"
    with atomic.atomic_open(out / "report.html", "w") as f:
        f.write(render_html(s))
    public = {k: v for k, v in s.items() if not k.startswith("_")}
    atomic.write_json(out / "summary.json", public)
    return out / "report.html"
