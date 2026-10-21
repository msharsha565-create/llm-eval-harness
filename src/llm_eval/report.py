from __future__ import annotations

from html import escape
from typing import Any

from llm_eval.runner import compare

CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e3e3e3;--card:#f7f7f8;--acc:#2457d6;--ok:#12803a;--bad:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#16171a;--fg:#ececec;--mut:#9a9a9a;--line:#33353a;--card:#1f2024;--acc:#7ea2ff;--ok:#5fd08a;--bad:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 18px}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}.card{flex:1;min-width:200px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card b{font-size:26px;display:block}.ok{color:var(--ok)}.bad{color:var(--bad)}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut)}code{font:12.5px ui-monospace,monospace;word-break:break-word}
.tag{font-size:12px;padding:1px 8px;border-radius:999px;border:1px solid var(--line)}
form{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:0 0 6px}select,button{padding:8px 10px;font-size:14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
button{background:var(--acc);color:#fff;border:0;cursor:pointer}
"""


def _cell(r: dict[str, Any] | None) -> str:
    if r is None:
        return "<td>-</td>"
    mark = '<b class="ok">PASS</b>' if r["passed"] else '<b class="bad">FAIL</b>'
    detail = f'<br><span class="sub">{escape(r["detail"])}</span>' if r["detail"] else ""
    return f"<td>{mark} <code>{escape(r['output'][:160])}</code>{detail}</td>"


def render_report(a: dict[str, Any], b: dict[str, Any] | None = None, controls: str = "") -> str:
    by_a = {r["id"]: r for r in a["results"]}
    by_b = {r["id"]: r for r in b["results"]} if b else {}
    cmp = compare(a, b) if b else None
    cards = (f'<div class="card"><b>{a["pass_rate"]:.0%}</b>{escape(a["model"])} '
             f'({a["passed"]}/{a["total"]} passed)</div>')
    if b and cmp:
        cards += (f'<div class="card"><b>{b["pass_rate"]:.0%}</b>{escape(b["model"])} '
                  f'({b["passed"]}/{b["total"]} passed)</div>')
        cls = "bad" if cmp["regressions"] else "ok"
        cards += (f'<div class="card"><b class="{cls}">{len(cmp["regressions"])} regression(s)'
                  f'</b>{len(cmp["improvements"])} improvement(s), '
                  f'delta {cmp["delta"]:+.0%}</div>')
    head = f"<th>{escape(a['model'])}</th>" + (f"<th>{escape(b['model'])}</th><th>Change</th>" if b else "")
    rows = []
    for case_id, ra in by_a.items():
        rb = by_b.get(case_id)
        change = ""
        if b and cmp:
            if case_id in cmp["regressions"]:
                change = '<td><span class="tag bad">regression</span></td>'
            elif case_id in cmp["improvements"]:
                change = '<td><span class="tag ok">improved</span></td>'
            else:
                change = "<td></td>"
        extra = (_cell(rb) + change) if b else ""
        rows.append(f"<tr><td><code>{escape(case_id)}</code><br><span class='sub'>"
                    f"{escape(ra['input'][:110])}</span></td>{_cell(ra)}{extra}</tr>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>llm-eval report</title><style>{CSS}</style></head><body><main>"
            f"<h1>llm-eval</h1><p class='sub'>Regression report over {a['total']} cases.</p>"
            f"{controls}<div class='cards'>{cards}</div>"
            f"<table><thead><tr><th>Case</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
            f"</main></body></html>")
